#!/usr/bin/env python3
"""
Unified LARM inference runner with four mutually exclusive modes:

  --random    : Sample random target extrinsics per step and render for a provided qpos list
  --multilink : Like --random, but target views are randomized ONCE from the FIRST JSON
                (treated as the "first joint") and REUSED for all subsequent JSONs
                (other joints) in this run.
  --view      : Target views — render one image per provided target pose across all qpos
  --video     : Render a ring of target views and stitch them into an MP4

Examples
--------
# Random mode 
python inference.py --random \
  --model_ckpt weight/larm/model_198000.pth \
  --datalist_path data_sample/random_metadata/data.txt \
  --save_dir output_random \
  --resolution 512 --batch_size 4 --num_input_views 6 \
  --num_target_views 32 \
  --qpos_list "0.00,0.25,0.50,0.75,1.00" \
  --qpos_in_a 0.00 --qpos_in_b 1.00

# Multilink mode (views sampled once on FIRST JSON, then reused for all others)
python inference.py --multilink \
  --model_ckpt weight/larm/model_198000.pth \
  --datalist_path data_sample/random_metadata/data.txt \
  --save_dir output_multilink \
  --resolution 512 --batch_size 4 --num_input_views 6 \
  --num_target_views 32 \
  --qpos_list "0.00,0.25,0.50,0.75,1.00" \
  --qpos_in_a 0.00 --qpos_in_b 1.00

# View mode
python inference.py --view \
  --model_ckpt weight/larm/model_198000.pth \
  --datalist_path data_sample/view_metadata/data.txt \
  --save_dir output_view \
  --resolution 512 --batch_size 4 --num_input_views 6 \
  --qpos_in_a 0.00 --qpos_in_b 1.00

# Video mode
python inference.py --video \
  --model_ckpt weight/larm/model_198000.pth \
  --datalist_path data_sample/random_metadata/data.txt \
  --save_dir output_video \
  --resolution 512 --batch_size 4 --num_input_views 6 \
  --num_target_views 64 --fps 25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.larm_mask import LARM
from mathutils import Vector
import calibur
from calibur import CC

_EVAL_ID_RX = re.compile(r'(eval_\d+)_joint_(\d+)')
_OBJ_RX = re.compile(r'(eval_\d+)')
_JOINT_RX = re.compile(r'joint_(\d+)')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def read_datalist(datalist_path: str) -> List[str]:
    paths: List[str] = []
    with open(datalist_path, "r") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                paths.append(s)
    return paths

def sort_input_frame_keys(d: Dict[str, Any]) -> List[str]:
    def idx(k: str) -> int:
        try:
            return int(k.split("input_frame_")[1])
        except Exception:
            return 1 << 30
    return sorted([k for k in d.keys() if k.startswith("input_frame_")], key=idx)

def sort_target_frame_keys(d: Dict[str, Any]) -> List[str]:
    def idx(k: str) -> int:
        try:
            return int(k.split("target_frame_")[1])
        except Exception:
            return 1 << 30
    return sorted([k for k in d.keys() if k.startswith("target_frame_")], key=idx)

def rgba_to_rgb_white(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")

def numeric_qpos_keys(frames_by_qpos: Dict[str, Any]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for k in frames_by_qpos.keys():
        out.append((k, float(k)))
    return sorted(out, key=lambda kv: kv[1])

def sample_extrinsics_ring(
    intrinsic_mat: np.ndarray,
    fovy_rad: float,
    azimuth: Optional[float] = None,
    elevation_deg: float = 30.0,
    circ_rad: float = 0.5,
    radius_scale: float = 1.2,
    resolution: int = 512,
) -> np.ndarray:
    blender2opencv = np.array([[1, 0, 0, 0],
                               [0,-1, 0, 0],
                               [0, 0,-1, 0],
                               [0, 0, 0, 1]], dtype=np.float64)

    if azimuth is None:
        azimuth = np.random.uniform(0.0, 2.0*np.pi)
    elevation = np.radians(elevation_deg)

    x = np.cos(azimuth) * np.sin(elevation)
    y = np.sin(azimuth) * np.sin(elevation)
    z = np.cos(elevation)
    t_unit_vec = np.array([x, y, z], dtype=np.float64)

    poi = np.zeros((3,), dtype=np.float64)

    R_blender = np.array(
        Vector(poi - t_unit_vec).to_track_quat('-Z', 'Y').to_matrix(),
        dtype=np.float64
    )

    sample_radius = (circ_rad / np.tan(fovy_rad / 2.0)) * radius_scale

    viewport_center = np.array([resolution / 2, resolution / 2, 1]) * sample_radius

    viewspace_center = np.linalg.inv(intrinsic_mat) @ viewport_center

    rot_cv = np.eye(4, dtype=np.float64)
    rot_cv[:3, :3] = R_blender
    rot_cv_cv = calibur.convert_pose(rot_cv, CC.Blender, CC.CV)

    t = poi - calibur.transform_point(viewspace_center, rot_cv_cv)

    cam2world_matrix = np.eye(4, dtype=np.float64)
    cam2world_matrix[:3, :3] = R_blender
    cam2world_matrix[:3, 3]  = t

    cam2world = cam2world_matrix @ blender2opencv
    return cam2world


def sample_extrinsics_simple(
    intrinsic_mat: np.ndarray,
    fovy_rad: float,
    azimuth: Optional[float] = None,
    elevation_deg_if_fixed: float = 45.0,
    circ_rad: float = 0.5,
    radius_scale: float = 1.2,
    elevation_avoid_poles_deg: float = 2,
    resolution = 512
) -> np.ndarray:
    blender2opencv = np.array([[1, 0, 0, 0],
                               [0,-1, 0, 0],
                               [0, 0,-1, 0],
                               [0, 0, 0, 1]], dtype=np.float64)

    if azimuth is None:
        azimuth = np.random.uniform(0.0, 2.0*np.pi)
        eps = np.radians(elevation_avoid_poles_deg)
        u = np.random.uniform(0.0, 1.0)
        elevation = np.arccos(2.0*u - 1.0)
        elevation = np.clip(elevation, eps, np.pi - eps)
    else:
        elevation = np.radians(elevation_deg_if_fixed)

    x = np.cos(azimuth) * np.sin(elevation)
    y = np.sin(azimuth) * np.sin(elevation)
    z = np.cos(elevation)
    t_unit_vec = np.array([x, y, z], dtype=np.float64)

    poi = np.zeros((3,), dtype=np.float64)

    R_blender = np.array(
        Vector(poi - t_unit_vec).to_track_quat('-Z', 'Y').to_matrix(),
        dtype=np.float64
    )

    sample_radius = (circ_rad / np.tan(fovy_rad / 2.0)) * radius_scale

    viewport_center = np.array([resolution / 2, resolution / 2, 1]) * sample_radius

    viewspace_center = np.linalg.inv(intrinsic_mat) @ viewport_center

    rot_cv = np.eye(4, dtype=np.float64)
    rot_cv[:3, :3] = R_blender
    rot_cv_cv = calibur.convert_pose(rot_cv, CC.Blender, CC.CV)

    t = poi - calibur.transform_point(viewspace_center, rot_cv_cv)

    cam2world_matrix = np.eye(4, dtype=np.float64)
    cam2world_matrix[:3, :3] = R_blender
    cam2world_matrix[:3, 3]  = t

    cam2world = cam2world_matrix @ blender2opencv
    return cam2world


def build_model(ckpt_path: str, resolution: int, batch_size: int, num_input_views: int, device: torch.device) -> LARM:
    config = {"hidden": 768, "num_layers": 12, "patch_size": 8, "linear_dim": 4096, "resolution": resolution}
    model = LARM(
        config["hidden"],
        config["num_layers"],
        config["patch_size"] * config["patch_size"] * 9 + 1,
        config["patch_size"] * config["patch_size"] * 6 + 1,
        config["patch_size"],
        config["linear_dim"],
        config["resolution"],
        batch_size,
        num_input_views,
        1,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    clean = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean, strict=True)
    return model.to(device).eval()

def collect_positions_all(frames_dict: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    keys = sort_input_frame_keys(frames_dict)
    pts: List[np.ndarray] = []
    valid_keys: List[str] = []
    for k in keys:
        rec = frames_dict[k]
        tm = np.array(rec["transform_matrix"], dtype=np.float64)
        pts.append(tm[:3, 3])
        valid_keys.append(k)
    if not valid_keys:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.stack(pts, axis=0), valid_keys

def _get_inputs_map(meta: Dict[str, Any]) -> Dict[str, Any]:
    if "inputs" in meta and isinstance(meta["inputs"], dict):
        return meta["inputs"]
    return meta["frames"]

def _get_targets_map(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    t = meta.get("targets", None)
    if isinstance(t, dict):
        return t
    return None

def _choose_keys_for_qpos(frames_at_qpos: Dict[str, Any], desired_keys: List[str], need: int) -> List[str]:
    if len(desired_keys) < need:
        raise ValueError(f"Requested {need} keys but only {len(desired_keys)} provided.")
    missing = [k for k in desired_keys[:need] if k not in frames_at_qpos]
    if missing:
        raise ValueError(f"Missing expected input keys at this qpos: {missing}")
    return desired_keys[:need]

def _prepare_inputs_from_metadata_inputs(
    inputs_by_qpos: Dict[str, Any],
    json_path: str,
    image_root: Optional[str],
    num_input_views: int,
    qpos_in_a: Optional[str],
    qpos_in_b: Optional[str],
) -> Tuple[str, str, List[str], List[str]]:
    if num_input_views != 6:
        raise ValueError(
            f"num_input_views must be 6 (3 from min qpos + 3 from max qpos). Got {num_input_views}."
        )

    q_sorted = numeric_qpos_keys(inputs_by_qpos)
    if not q_sorted:
        raise ValueError("No inputs found in metadata.")
    if len(q_sorted) < 2 or float(q_sorted[0][1]) == float(q_sorted[-1][1]):
        raise ValueError("Need at least two distinct qposes to take min and max.")

    qpos_min_tok = q_sorted[0][0]
    qpos_max_tok = q_sorted[-1][0]

    def first_three_keys(frames_at_qpos: Dict[str, Any], qtok: str) -> List[str]:
        ks = sort_input_frame_keys(frames_at_qpos)
        if len(ks) < 3:
            raise ValueError(
                f"qpos={qtok} has only {len(ks)} input frames; need at least 3."
            )
        return ks[:3]

    sel_keys_min = first_three_keys(inputs_by_qpos[qpos_min_tok], qpos_min_tok)
    sel_keys_max = first_three_keys(inputs_by_qpos[qpos_max_tok], qpos_max_tok)

    sel_keys_min = _choose_keys_for_qpos(inputs_by_qpos[qpos_min_tok], sel_keys_min, 3)
    sel_keys_max = _choose_keys_for_qpos(inputs_by_qpos[qpos_max_tok], sel_keys_max, 3)

    return qpos_min_tok, qpos_max_tok, sel_keys_min, sel_keys_max

def _fill_inputs_tensors(
    image: torch.Tensor,
    extrinsic_in: torch.Tensor,
    inputs_by_qpos: Dict[str, Any],
    qpos_in_a_tok: str,
    qpos_in_b_tok: str,
    sel_keys_a: List[str],
    sel_keys_b: List[str],
    json_path: str,
    image_root: Optional[str],
    device: torch.device,
) -> None:
    blender2opencv = np.array([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]], dtype=np.float64)

    def _strict_fill(side_keys: List[str], qtok: str, start_slot: int) -> None:
        for offset, key in enumerate(side_keys):
            rec = inputs_by_qpos[qtok][key]
            img_path = rec.get("image_path", "")
            if not img_path:
                raise ValueError(f"Empty image_path for qpos={qtok}, key={key}")
            try:
                img = rgba_to_rgb_white(Image.open(img_path))
            except Exception as e:
                raise ValueError(f"Failed to open image at qpos={qtok}, key={key}, path='{img_path}': {e}")
            arr = (np.array(img).astype(np.float32) / 255.0)
            if arr.ndim != 3 or arr.shape[2] < 3:
                raise ValueError(f"Image not RGB for qpos={qtok}, key={key}, path='{img_path}' (shape {arr.shape})")
            image[0, start_slot + offset] = torch.from_numpy(arr).permute(2, 0, 1).to(device)[:3]
            e = np.array(rec["transform_matrix"], dtype=np.float64) @ blender2opencv
            extrinsic_in[0, start_slot + offset] = torch.tensor(e, dtype=torch.float32, device=device)

    n_a = len(sel_keys_a)
    _strict_fill(sel_keys_a, qpos_in_a_tok, 0)
    _strict_fill(sel_keys_b, qpos_in_b_tok, n_a)

def _idx_from_frame_path(p: str) -> int:
    base = os.path.basename(p)
    num = base.split("_")[0]
    return int(num) if num.isdigit() else 1 << 30

def _extract_obj_id(path: str) -> Optional[str]:
    m = _OBJ_RX.search(path)
    return m.group(1) if m else None

def _joint_idx_from_path(p: str) -> int:
    m = _JOINT_RX.search(p)
    return int(m.group(1)) if m else 1 << 30

def _random_run_one(
    model: LARM,
    device: torch.device,
    json_path: str,
    save_dir: str,
    resolution: int,
    batch_size: int,
    num_input_views: int,
    num_target_views: int,
    qpos_list_str: str,
    qpos_in_a: Optional[str],
    qpos_in_b: Optional[str],
    image_root: Optional[str],
) -> None: 
    data = load_json(json_path)
    if "intrinsics" not in data or ("inputs" not in data and "frames" not in data):
        raise ValueError(f"{json_path}: expected 'intrinsics' and 'inputs' (or legacy 'frames').")

    K = np.array(data["intrinsics"], dtype=np.float64)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    fovy = calibur.focal_to_fov(fx, resolution)
    intrinsic = torch.tensor([[[fx, fy, cx, cy]]], dtype=torch.float32, device=device)
    intrinsic_mat = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]], dtype=np.float64)

    joint_type = data.get("joint_type", "revolute")
    inputs_by_qpos: Dict[str, Any] = _get_inputs_map(data)

    qpos_in_a_tok, qpos_in_b_tok, sel_keys_a, sel_keys_b = _prepare_inputs_from_metadata_inputs(
        inputs_by_qpos, json_path, image_root, num_input_views, qpos_in_a, qpos_in_b
    )

    n_a = len(sel_keys_a)
    n_b = len(sel_keys_b)

    image = torch.zeros((1, num_input_views + batch_size, 3, resolution, resolution), dtype=torch.float32, device=device)
    extrinsic_in = torch.zeros((1, num_input_views, 4, 4), dtype=torch.float32, device=device)
    _fill_inputs_tensors(image, extrinsic_in, inputs_by_qpos, qpos_in_a_tok, qpos_in_b_tok, sel_keys_a, sel_keys_b, json_path, image_root, device)

    qa = float(qpos_in_a_tok); qb = float(qpos_in_b_tok)
    qpos_in_vec = torch.tensor([[qa] * n_a + [qb] * n_b], dtype=torch.float32, device=device)

    q_tokens_cli = [s.strip() for s in qpos_list_str.split(",") if s.strip() != ""]
    q_list = [float(s) for s in q_tokens_cli]
    q_block_size = num_target_views
    q_counters = [0] * len(q_list)

    name = os.path.splitext(os.path.basename(json_path))[0]
    out_dir = os.path.join(save_dir, f"eval_{name}")
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    images_dir = os.path.join(out_dir, "images")

    meta = {
        "camera_angle_x": fovy,
        "frames": [],
        "cx": float(cx), "cy": float(cy), "fx": float(fx), "fy": float(fy),
        "w": resolution, "h": resolution,
        "qpos_list": q_list,
        "joint_type": joint_type,
    }

    steps = (num_target_views + batch_size - 1) // batch_size
    for j in tqdm(range(steps), desc=f"{name} | sampling {num_target_views}"):
        extrinsic_out = np.zeros((1, batch_size, 4, 4), dtype=np.float32)
        for i in range(batch_size):
            extrinsic_out[0, i] = sample_extrinsics_simple(intrinsic_mat, fovy, resolution=resolution).astype(np.float32)
        extrinsic_out_t = torch.from_numpy(extrinsic_out).to(device)

        for qi, q in enumerate(q_list):
            qpos_out = torch.full((1, batch_size), float(q), dtype=torch.float32, device=device)
            with torch.no_grad():
                output = model(
                    intrinsic.repeat(1, image.shape[1], 1),
                    extrinsic_in,
                    extrinsic_out_t,
                    image,
                    qpos_in=qpos_in_vec,
                    qpos_out=qpos_out,
                ).permute(0, 1, 4, 2, 3)
                out_sel = output[:, num_input_views::num_input_views + 1]

            true_items = min(batch_size, num_target_views - j * batch_size)
            for m in range(true_items):
                local_idx = q_counters[qi]
                global_index = qi * q_block_size + local_idx
                q_counters[qi] += 1
                q_token_for_name = q_tokens_cli[qi]
                stem = f"{global_index}_{float(q_token_for_name):.02f}"

                rgb = (out_sel[0, m, :3].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                
                Image.fromarray(rgb).resize((resolution, resolution)).save(os.path.join(images_dir, f"{stem}.png"))

                partmask = (out_sel[0, m, 3:4].detach().cpu().squeeze().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(partmask).resize((resolution, resolution)).save(os.path.join(images_dir, f"{stem}_partmask.png"))

                mask = (out_sel[0, m, 4:5].detach().cpu().squeeze().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(mask).resize((resolution, resolution)).save(os.path.join(images_dir, f"{stem}_mask.png"))

                depth = (out_sel[0, m, 5:6].detach().cpu().squeeze().numpy())
                depth_u8 = (np.clip(depth * 255.0, 0, 255)).astype(np.uint8)
                Image.fromarray(depth_u8).resize((resolution, resolution)).save(os.path.join(images_dir, f"{stem}_depth.png"))
                np.save(os.path.join(images_dir, f"{stem}_depth.npy"), depth)

                frame = {
                    "file_path": f"./images/{stem}",
                    "qpos": float(q),
                    "transform_matrix": calibur.convert_pose(
                        extrinsic_out_t[0, m].detach().cpu().numpy(), ("X", "-Y", "-Z"), ("X", "Y", "Z")
                    ).tolist()
                }
                meta["frames"].append(frame)

    meta["frames"].sort(key=lambda f: (f.get("qpos", 0.0), _idx_from_frame_path(f["file_path"])))
    with open(os.path.join(out_dir, "transforms.json"), "w") as f:
        json.dump(meta, f, indent=2)

def _multilink_run_one(
    model: LARM,
    device: torch.device,
    json_path: str,
    save_dir: str,
    resolution: int,
    batch_size: int,
    num_input_views: int,
    num_target_views: int,
    qpos_list_str: str,
    qpos_in_a: Optional[str],
    qpos_in_b: Optional[str],
    image_root: Optional[str],
    override_extrinsics: Optional[np.ndarray] = None,
) -> np.ndarray:
    data = load_json(json_path)
    if "intrinsics" not in data or ("inputs" not in data and "frames" not in data):
        raise ValueError(f"{json_path}: expected 'intrinsics' and 'inputs' (or legacy 'frames').")

    K = np.array(data["intrinsics"], dtype=np.float64)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    fovy = calibur.focal_to_fov(fx, resolution)
    intrinsic = torch.tensor([[[fx, fy, cx, cy]]], dtype=torch.float32, device=device)
    intrinsic_mat = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]], dtype=np.float64)

    joint_type = data.get("joint_type", "revolute")
    inputs_by_qpos: Dict[str, Any] = _get_inputs_map(data)

    qpos_in_a_tok, qpos_in_b_tok, sel_keys_a, sel_keys_b = _prepare_inputs_from_metadata_inputs(
        inputs_by_qpos, json_path, image_root, num_input_views, qpos_in_a, qpos_in_b
    )

    n_a = len(sel_keys_a)
    n_b = len(sel_keys_b)

    image = torch.zeros((1, num_input_views + batch_size, 3, resolution, resolution), dtype=torch.float32, device=device)
    extrinsic_in = torch.zeros((1, num_input_views, 4, 4), dtype=torch.float32, device=device)
    _fill_inputs_tensors(image, extrinsic_in, inputs_by_qpos, qpos_in_a_tok, qpos_in_b_tok, sel_keys_a, sel_keys_b, json_path, image_root, device)

    qa = float(qpos_in_a_tok); qb = float(qpos_in_b_tok)
    qpos_in_vec = torch.tensor([[qa] * n_a + [qb] * n_b], dtype=torch.float32, device=device)

    q_tokens_cli = [s.strip() for s in qpos_list_str.split(",") if s.strip() != ""]
    q_list = [float(s) for s in q_tokens_cli]
    q_block_size = num_target_views
    q_counters = [0] * len(q_list)

    name = os.path.splitext(os.path.basename(json_path))[0]   # e.g., "44817_joint_0"
    out_dir = os.path.join(save_dir, f"eval_{name}")          # e.g., ".../eval_44817_joint_0"
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    images_dir = os.path.join(out_dir, "images")

    meta = {
        "camera_angle_x": fovy,
        "frames": [],
        "cx": float(cx), "cy": float(cy), "fx": float(fx), "fy": float(fy),
        "w": resolution, "h": resolution,
        "qpos_list": q_list,
        "joint_type": joint_type,
        "multilink_extrinsics_source": "override" if override_extrinsics is not None else "sampled_first_json"
    }

    if override_extrinsics is not None:
        if override_extrinsics.shape != (num_target_views, 4, 4):
            raise ValueError(
                f"override_extrinsics has shape {override_extrinsics.shape}, "
                f"expected ({num_target_views}, 4, 4)"
            )
        extrinsics_all = override_extrinsics.astype(np.float32, copy=False)
    else:
        extrinsics_all = np.zeros((num_target_views, 4, 4), dtype=np.float32)
        for i in range(num_target_views):
            extrinsics_all[i] = sample_extrinsics_simple(intrinsic_mat, fovy, resolution=resolution).astype(np.float32)

    steps = (num_target_views + batch_size - 1) // batch_size
    for j in tqdm(range(steps), desc=f"{name} | sampling {num_target_views} (multilink)"):
        start = j * batch_size
        end = min(start + batch_size, num_target_views)

        extrinsic_out = np.zeros((1, batch_size, 4, 4), dtype=np.float32)
        for i in range(batch_size):
            idx = min(start + i, end - 1)
            extrinsic_out[0, i] = extrinsics_all[idx]
        extrinsic_out_t = torch.from_numpy(extrinsic_out).to(device)

        for qi, q in enumerate(q_list):
            qpos_out = torch.full((1, batch_size), float(q), dtype=torch.float32, device=device)
            with torch.no_grad():
                output = model(
                    intrinsic.repeat(1, image.shape[1], 1),
                    extrinsic_in,
                    extrinsic_out_t,
                    image,
                    qpos_in=qpos_in_vec,
                    qpos_out=qpos_out,
                ).permute(0, 1, 4, 2, 3)
                out_sel = output[:, num_input_views::num_input_views + 1]

            true_items = end - start
            for m in range(true_items):
                local_idx = q_counters[qi]
                global_index = qi * q_block_size + local_idx
                q_counters[qi] += 1
                q_token_for_name = q_tokens_cli[qi]
                stem = f"{global_index}_{float(q_token_for_name):.02f}"

                rgb = (out_sel[0, m, :3].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(rgb).resize((resolution, resolution)).save(os.path.join(images_dir, f"{stem}.png"))

                partmask = (out_sel[0, m, 3:4].detach().cpu().squeeze().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(partmask).resize((resolution, resolution)).save(os.path.join(images_dir, f"{stem}_partmask.png"))

                mask = (out_sel[0, m, 4:5].detach().cpu().squeeze().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(mask).resize((resolution, resolution)).save(os.path.join(images_dir, f"{stem}_mask.png"))

                depth = (out_sel[0, m, 5:6].detach().cpu().squeeze().numpy())
                depth_u8 = (np.clip(depth * 255.0, 0, 255)).astype(np.uint8)
                Image.fromarray(depth_u8).resize((resolution, resolution)).save(os.path.join(images_dir, f"{stem}_depth.png"))
                np.save(os.path.join(images_dir, f"{stem}_depth.npy"), depth)

                frame = {
                    "file_path": f"./images/{stem}",
                    "qpos": float(q),
                    "transform_matrix": calibur.convert_pose(
                        extrinsic_out_t[0, m].detach().cpu().numpy(), ("X", "-Y", "-Z"), ("X", "Y", "Z")
                    ).tolist()
                }
                meta["frames"].append(frame)

    meta["frames"].sort(key=lambda f: (f.get("qpos", 0.0), _idx_from_frame_path(f["file_path"])))
    with open(os.path.join(out_dir, "transforms.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return extrinsics_all


def _view_run_one_json(
    model: LARM,
    device: torch.device,
    input_json: str,
    save_dir: str,
    resolution: int = 512,
    batch_size: int = 4,
    num_input_views: int = 6,
    qpos_in_a: Optional[str] = None,
    qpos_in_b: Optional[str] = None,
    image_root: Optional[str] = None,
) -> None:
    data = load_json(input_json)
    if "intrinsics" not in data or ("inputs" not in data and "frames" not in data):
        raise ValueError(f"{input_json}: JSON must contain 'intrinsics' and 'inputs' (or legacy 'frames').")
    inputs_by_qpos: Dict[str, Any] = _get_inputs_map(data)
    targets_by_qpos: Optional[Dict[str, Any]] = _get_targets_map(data)
    if not targets_by_qpos:
        raise ValueError(f"{input_json}: 'targets' missing or empty for --view mode.")

    joint_type = data.get("joint_type", "revolute")
    qpos_tokens_sorted = numeric_qpos_keys(inputs_by_qpos)
    all_qpos_floats = [val for _, val in qpos_tokens_sorted]

    qpos_in_a_tok, qpos_in_b_tok, sel_keys_a, sel_keys_b = _prepare_inputs_from_metadata_inputs(
        inputs_by_qpos, input_json, image_root, num_input_views, qpos_in_a, qpos_in_b
    )
    n_a = len(sel_keys_a)
    n_b = len(sel_keys_b)

    K = np.array(data["intrinsics"], dtype=np.float64)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    fovy = calibur.focal_to_fov(fx, resolution)
    intrinsic = torch.tensor([[[fx, fy, cx, cy]]], dtype=torch.float32, device=device)

    name = os.path.splitext(os.path.basename(input_json))[0]
    out_dir = os.path.join(save_dir, f"eval_{name}")
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    image = torch.zeros((1, num_input_views + batch_size, 3, resolution, resolution), dtype=torch.float32, device=device)
    extrinsic_in = torch.zeros((1, num_input_views, 4, 4), dtype=torch.float32, device=device)
    _fill_inputs_tensors(image, extrinsic_in, inputs_by_qpos, qpos_in_a_tok, qpos_in_b_tok, sel_keys_a, sel_keys_b, input_json, image_root, device)

    qa = float(qpos_in_a_tok)
    qb = float(qpos_in_b_tok)
    qpos_in_vec = torch.tensor([[qa] * n_a + [qb] * n_b], dtype=torch.float32, device=device)

    blender2opencv = np.array([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]], dtype=np.float64)

    target_list: List[Tuple[str, str, np.ndarray]] = []
    for tok in sorted(targets_by_qpos.keys(), key=lambda s: float(s)):
        tframes = targets_by_qpos[tok]
        tkeys = sort_target_frame_keys(tframes)
        for tk in tkeys:
            rec = tframes[tk]
            E_raw = np.array(rec["transform_matrix"], dtype=np.float64)
            E_cv = calibur.convert_pose(E_raw, CC.Blender, CC.CV)
            target_list.append((tok, tk, E_cv))

    N = len(target_list)
    if N == 0:
        raise ValueError(f"{input_json}: no target frames available under 'targets'.")

    meta_save: Dict[str, Any] = {
        "camera_angle_x": fovy,
        "frames": [],
        "cx": cx, "cy": cy, "fx": fx, "fy": fy,
        "w": resolution, "h": resolution,
        "qpos_list": all_qpos_floats,
        "joint_type": joint_type,
    }

    steps = (N + batch_size - 1) // batch_size
    for j in tqdm(range(steps), desc=f"{name} | {N} targets"):
        start = j * batch_size
        end = min(start + batch_size, N)
        batch_targets = target_list[start:end]

        extrinsic_out = torch.zeros((1, batch_size, 4, 4), dtype=torch.float32, device=device)
        qpos_out_vec = torch.zeros((1, batch_size), dtype=torch.float32, device=device)
        valid_indices: List[int] = []  # track exact target indices for saving

        # Fill extrinsics & qpos using the *real* JSON target indices
        for i in range(batch_size):
            if start + i < end:
                tok, tk, E = batch_targets[i]
                try:
                    target_idx = int(tk.split("target_frame_")[1])
                except Exception:
                    target_idx = -1
            else:
                tok, tk, E = batch_targets[-1]
                target_idx = -1

            extrinsic_out[0, i] = torch.tensor(E, dtype=torch.float32, device=device)
            qpos_out_vec[0, i] = float(tok)
            valid_indices.append(target_idx)

        # Run inference
        with torch.no_grad():
            output = model(
                intrinsic.repeat(1, image.shape[1], 1),
                extrinsic_in,
                extrinsic_out,
                image,
                qpos_in=qpos_in_vec,
                qpos_out=qpos_out_vec,
            ).permute(0, 1, 4, 2, 3)
            out_sel = output[:, num_input_views::num_input_views + 1]

        # === Save results exactly matching target indices ===
        true_batch = end - start
        for m in range(true_batch):
            tok, tk, _ = batch_targets[m]
            target_idx = valid_indices[m]
            if target_idx < 0:
                continue  # skip padded entries (if any)

            q_float = float(tok)
            stem = f"{target_idx}_{q_float:.02f}"  # e.g. "17_0.50"

            # RGB / masks / depth saving
            rgb = (out_sel[0, m, :3].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(rgb).resize((resolution, resolution)).save(os.path.join(images_dir, f"{stem}.png"))

            partmask = (out_sel[0, m, 3:4].detach().cpu().squeeze().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(partmask).resize((resolution, resolution)).save(os.path.join(images_dir, f"{stem}_partmask.png"))

            mask = (out_sel[0, m, 4:5].detach().cpu().squeeze().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(mask).resize((resolution, resolution)).save(os.path.join(images_dir, f"{stem}_mask.png"))

            depth = (out_sel[0, m, 5:6].detach().cpu().squeeze().numpy())
            depth_u8 = (np.clip(depth * 255.0, 0, 255)).astype(np.uint8)
            Image.fromarray(depth_u8).resize((resolution, resolution)).save(os.path.join(images_dir, f"{stem}_depth.png"))
            np.save(os.path.join(images_dir, f"{stem}_depth.npy"), depth)

            # Frame metadata (preserving index)
            frame = {
                "file_path": f"./images/{stem}",
                "qpos": q_float,
                "transform_matrix": calibur.convert_pose(
                    extrinsic_out[0, m].detach().cpu().numpy(), ("X", "-Y", "-Z"), ("X", "Y", "Z")
                ).tolist()
            }
            meta_save["frames"].append(frame)

    with open(os.path.join(out_dir, "transforms.json"), "w") as f:
        json.dump(meta_save, f, indent=2)

def _video_run_one_json(
    model: LARM,
    device: torch.device,
    json_path: str,
    save_root: str,
    *,
    resolution: int = 512,
    num_target_views: int = 64,
    batch_size: int = 4,
    num_input_views: int = 6,
    fps: int = 25,
    image_root: Optional[str] = None,
) -> None:
    data = load_json(json_path)
    if "intrinsics" not in data or ("inputs" not in data and "frames" not in data):
        raise ValueError(f"{json_path}: expected 'intrinsics' and 'inputs' (or legacy 'frames').")

    K = np.array(data["intrinsics"], dtype=np.float64)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    intrinsic = torch.tensor([[[fx, fy, cx, cy]]], dtype=torch.float32, device=device)
    intrinsic_mat = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]], dtype=np.float64)
    fovy_default = calibur.focal_to_fov(fx, resolution)

    joint_type = data.get("joint_type", "revolute")
    inputs_by_qpos = _get_inputs_map(data)

    qpos_in_a_tok, qpos_in_b_tok, sel_keys_0, sel_keys_1 = _prepare_inputs_from_metadata_inputs(
        inputs_by_qpos, json_path, image_root, num_input_views, None, None
    )

    blender2opencv = np.array([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]], dtype=np.float64)

    image = torch.zeros((1, num_input_views + batch_size, 3, resolution, resolution), dtype=torch.float32, device=device)
    extrinsic_in = torch.zeros((1, num_input_views, 4, 4), dtype=torch.float32, device=device)

    filled0 = 0
    for idx_i, key in enumerate(sel_keys_0):
        rec = inputs_by_qpos[qpos_in_a_tok][key]
        img_path = rec.get("image_path", "")
        if not img_path:
            raise ValueError(f"Empty image_path for qpos={qpos_in_a_tok}, key={key}")
        try:
            in_image = rgba_to_rgb_white(Image.open(img_path))
        except Exception as e:
            raise ValueError(f"Failed to open image at qpos={qpos_in_a_tok}, key={key}, path='{img_path}': {e}")
        arr = (np.array(in_image).astype(np.float32) / 255.0)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"Image not RGB for qpos={qpos_in_a_tok}, key={key}, path='{img_path}' (shape {arr.shape})")
        image[0, idx_i] = torch.from_numpy(arr).permute(2, 0, 1).to(device)[:3]
        e = np.array(rec['transform_matrix'], dtype=np.float64) @ blender2opencv
        extrinsic_in[0, idx_i] = torch.tensor(e, dtype=torch.float32, device=device)
        filled0 += 1
    if filled0 != len(sel_keys_0):
        raise ValueError(f"Failed to load all images for qpos={qpos_in_a_tok}. Loaded {filled0}/{len(sel_keys_0)}.")

    n_half = len(sel_keys_0)
    filled1 = 0
    for idx_i, key in enumerate(sel_keys_1):
        rec = inputs_by_qpos[qpos_in_b_tok][key]
        img_path = rec.get("image_path", "")
        if not img_path:
            raise ValueError(f"Empty image_path for qpos={qpos_in_b_tok}, key={key}")
        try:
            in_image = rgba_to_rgb_white(Image.open(img_path))
        except Exception as e:
            raise ValueError(f"Failed to open image at qpos={qpos_in_b_tok}, key={key}, path='{img_path}': {e}")
        arr = (np.array(in_image).astype(np.float32) / 255.0)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"Image not RGB for qpos={qpos_in_b_tok}, key={key}, path='{img_path}' (shape {arr.shape})")
        image[0, idx_i + n_half] = torch.from_numpy(arr).permute(2, 0, 1).to(device)[:3]
        e = np.array(rec['transform_matrix'], dtype=np.float64) @ blender2opencv
        extrinsic_in[0, idx_i + n_half] = torch.tensor(e, dtype=torch.float32, device=device)
        filled1 += 1
    if filled1 != len(sel_keys_1):
        raise ValueError(f"Failed to load all images for qpos={qpos_in_b_tok}. Loaded {filled1}/{len(sel_keys_1)}.")

    qa = float(qpos_in_a_tok)
    qb = float(qpos_in_b_tok)
    qpos_in_vec = torch.tensor([[qa] * n_half + [qb] * (num_input_views - n_half)], dtype=torch.float32, device=device)

    name = os.path.splitext(os.path.basename(json_path))[0]
    seq_dir = os.path.join(save_root, f"eval_{name}")
    images_dir = os.path.join(seq_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    transforms = {
        "camera_angle_x": calibur.focal_to_fov(float(fx), resolution),
        "frames": [],
        "cx": float(cx), "cy": float(cy), "fx": float(fx), "fy": float(fy),
        "w": resolution, "h": resolution,
        "joint_type": joint_type
    }

    for kk in range(num_input_views):
        arr = (image[0, kk].detach().permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(seq_dir, f"input_{kk}.png"))

    steps = (num_target_views + batch_size - 1) // batch_size
    for j in tqdm(range(steps), desc=f"{name} | {num_target_views} views"):
        extrinsic_out = np.zeros((1, batch_size, 4, 4), dtype=np.float32)
        for i in range(batch_size):
            num_im = batch_size * j + i
            if num_im >= num_target_views:
                extrinsic_out[0, i] = extrinsic_out[0, max(i - 1, 0)]
            else:
                az = np.pi * 2 / max(1, num_target_views) * num_im
                extrinsic_out[0, i] = sample_extrinsics_ring(intrinsic_mat, fovy_default, azimuth=az).astype(np.float32)
        extrinsic_out_t = torch.tensor(extrinsic_out, dtype=torch.float32, device=device)

        qpos_out_vals = np.linspace(0.0, 1.0, num_target_views)[batch_size * j: batch_size * (j + 1)]
        if len(qpos_out_vals) < batch_size:
            qpos_out_vals = np.pad(qpos_out_vals, (0, batch_size - len(qpos_out_vals)), mode="edge")
        qpos_out = torch.tensor(qpos_out_vals, dtype=torch.float32, device=device).reshape(1, -1)

        with torch.no_grad():
            output = model(intrinsic.repeat(1, image.shape[1], 1), extrinsic_in, extrinsic_out_t, image,
                           qpos_in=qpos_in_vec, qpos_out=qpos_out)
            output = output.permute(0, 1, 4, 2, 3)
            output = output[:, num_input_views::num_input_views + 1]

        for m in range(min(batch_size, num_target_views - j * batch_size)):
            num_im = batch_size * j + m
            frame = {
                "file_path": f"./images/{num_im}",
                "transform_matrix": calibur.convert_pose(
                    extrinsic_out_t[0, m].detach().cpu().numpy(), ("X", "-Y", "-Z"), ("X", "Y", "Z")
                ).tolist()
            }
            transforms["frames"].append(frame)

            rgb = (output[0, m, :3].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            partmask = (output[0, m, 3:4].detach().cpu().squeeze().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            objmask = (output[0, m, 4:5].detach().cpu().squeeze().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            depth = (output[0, m, 5:6].detach().cpu().squeeze().numpy())
            depth_u8 = (np.clip(depth * 255.0, 0, 255)).astype(np.uint8)

            Image.fromarray(rgb).resize((resolution, resolution)).save(os.path.join(images_dir, f"{num_im}.png"))
            Image.fromarray(partmask).resize((resolution, resolution)).save(os.path.join(images_dir, f"{num_im}_partmask.png"))
            Image.fromarray(objmask).resize((resolution, resolution)).save(os.path.join(images_dir, f"{num_im}_mask.png"))
            Image.fromarray(depth_u8).resize((resolution, resolution)).save(os.path.join(images_dir, f"{num_im}_depth.png"))
            np.save(os.path.join(images_dir, f"{num_im}_depth.npy"), depth)

    os.makedirs(seq_dir, exist_ok=True)
    with open(os.path.join(seq_dir, "transforms.json"), "w") as f:
        json.dump(transforms, f, indent=2)

    frames = []
    for idx in range(num_target_views):
        img_path = os.path.join(images_dir, f"{idx}.png")
        img = cv2.imread(img_path)
        if img is not None:
            frames.append(img)
    if frames:
        h, w = frames[0].shape[:2]
        video_path = os.path.join(seq_dir, "video.mp4")
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for frame in frames:
            writer.write(frame)
        writer.release()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified LARM multitool with four modes: --random, --multilink, --view, --video (mutually exclusive)."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--random", action="store_true", help="Run RANDOM mode.")
    g.add_argument("--multilink", action="store_true", help="Run MULTILINK mode (views sampled once on the FIRST JSON, reused for all).")
    g.add_argument("--view", action="store_true", help="Run VIEW mode.")
    g.add_argument("--video", action="store_true", help="Run VIDEO mode.")

    p.add_argument('--save_dir', type=str, default="output", help="Output root directory.")
    p.add_argument('--datalist_path', type=str, default="data_sample/data.txt", help="Text file: one consolidated JSON path per line.")
    p.add_argument('--model_ckpt', type=str, default="weight/larm/model_198000.pth", help="Model checkpoint path.")
    p.add_argument('--resolution', type=int, default=512, help="Image resolution (square).")
    p.add_argument('--batch_size', type=int, default=4, help="Batch size for target rendering.")
    p.add_argument('--num_input_views', type=int, default=6, help="Number of input context views (must be 6).")
    p.add_argument('--image_root', type=str, default=None, help="Optional root to resolve relative image paths in metadata (unused by strict loader).")

    p.add_argument('--qpos_in_a', type=str, default=None, help="(ignored for selection) Anchor qpos token for first half of inputs.")
    p.add_argument('--qpos_in_b', type=str, default=None, help="(ignored for selection) Anchor qpos token for second half of inputs.")

    p.add_argument('--num_target_views', type=int, default=32, help="[random, multilink, video] total target views to render.")
    p.add_argument('--qpos_list', type=str, default="0.0,0.25,0.5,0.75,1.0",
                   help="[random, multilink] Comma-separated qpos values to render, e.g. '0.00,0.25,0.50,0.75,1.00'.")

    p.add_argument('--fps', type=int, default=25, help="[video] FPS for the output MP4.")

    return p.parse_args()

def run_random(args: argparse.Namespace, model: LARM, device: torch.device) -> None:
    json_list = read_datalist(args.datalist_path)
    if not json_list:
        print(f"[WARN] No JSONs found in {args.datalist_path}")
        return
    os.makedirs(args.save_dir, exist_ok=True)
    for jp in json_list:
        if not os.path.isfile(jp):
            print(f"[WARN] Skipping missing JSON: {jp}")
            continue
        print(f"[INFO] [random] Processing {jp}")
        _random_run_one(
            model=model,
            device=device,
            json_path=jp,
            save_dir=args.save_dir,
            resolution=args.resolution,
            batch_size=args.batch_size,
            num_input_views=args.num_input_views,
            num_target_views=args.num_target_views,
            qpos_list_str=args.qpos_list,
            qpos_in_a=args.qpos_in_a,
            qpos_in_b=args.qpos_in_b,
            image_root=args.image_root,
        )
    print("Done (random).")

from collections import defaultdict

def run_multilink(args: argparse.Namespace, model: LARM, device: torch.device) -> None:
    json_list = read_datalist(args.datalist_path)
    if not json_list:
        print(f"[WARN] No JSONs found in {args.datalist_path}")
        return

    grouped: Dict[str, List[str]] = defaultdict(list)
    for jp in json_list:
        if not os.path.isfile(jp):
            print(f"[WARN] Missing JSON, skipping: {jp}")
            continue
        obj_id = _extract_obj_id(jp) or os.path.dirname(os.path.dirname(jp))
        grouped[obj_id].append(jp)

    os.makedirs(args.save_dir, exist_ok=True)

    for obj_id, jsons in grouped.items():
        jsons.sort(key=_joint_idx_from_path)
        print(f"\n[INFO] [multilink] Processing object {obj_id} ({len(jsons)} joints)")
        shared_extrinsics: Optional[np.ndarray] = None
        for idx, jp in enumerate(jsons):
            if shared_extrinsics is None:
                print(f"[INFO] [multilink] FIRST JOINT (defines cameras): {jp}")
                shared_extrinsics = _multilink_run_one(
                    model=model,
                    device=device,
                    json_path=jp,
                    save_dir=args.save_dir,
                    resolution=args.resolution,
                    batch_size=args.batch_size,
                    num_input_views=args.num_input_views,
                    num_target_views=args.num_target_views,
                    qpos_list_str=args.qpos_list,
                    qpos_in_a=args.qpos_in_a,
                    qpos_in_b=args.qpos_in_b,
                    image_root=args.image_root,
                    override_extrinsics=None,
                )
                try:
                    np.save(os.path.join(args.save_dir, f"extrinsics_{obj_id}.npy"), shared_extrinsics)
                except Exception as e:
                    print(f"[WARN] Could not save extrinsics for {obj_id}: {e}")
            else:
                print(f"[INFO] [multilink] Reusing shared extrinsics for {jp}")
                _ = _multilink_run_one(
                    model=model,
                    device=device,
                    json_path=jp,
                    save_dir=args.save_dir,
                    resolution=args.resolution,
                    batch_size=args.batch_size,
                    num_input_views=args.num_input_views,
                    num_target_views=args.num_target_views,
                    qpos_list_str=args.qpos_list,
                    qpos_in_a=args.qpos_in_a,
                    qpos_in_b=args.qpos_in_b,
                    image_root=args.image_root,
                    override_extrinsics=shared_extrinsics,
                )
    print("\n✅ Done (multilink grouped by object).")

def run_view(args: argparse.Namespace, model: LARM, device: torch.device) -> None:
    json_list = read_datalist(args.datalist_path)
    if not json_list:
        print(f"[WARN] No JSONs found in {args.datalist_path}")
        return
    for jp in json_list:
        if not os.path.isfile(jp):
            print(f"[WARN] Skipping missing JSON: {jp}")
            continue
        print(f"[INFO] [view] Processing {jp}")
        _view_run_one_json(
            model=model,
            device=device,
            input_json=jp,
            save_dir=args.save_dir,
            resolution=args.resolution,
            batch_size=args.batch_size,
            num_input_views=args.num_input_views,
            qpos_in_a=args.qpos_in_a,
            qpos_in_b=args.qpos_in_b,
            image_root=args.image_root,
        )
    print("Done (view).")

def run_video(args: argparse.Namespace, model: LARM, device: torch.device) -> None:
    json_list = read_datalist(args.datalist_path)
    if not json_list:
        print(f"[WARN] No JSON paths found in {args.datalist_path}")
        return
    os.makedirs(args.save_dir, exist_ok=True)
    for jp in json_list:
        if not os.path.isfile(jp):
            print(f"[WARN] Missing JSON: {jp} — skipping.")
            continue
        print(f"[INFO] [video] Processing {jp}")
        _video_run_one_json(
            model=model,
            device=device,
            json_path=jp,
            save_root=args.save_dir,
            resolution=args.resolution,
            num_target_views=args.num_target_views,
            batch_size=args.batch_size,
            num_input_views=args.num_input_views,
            fps=args.fps,
            image_root=args.image_root,
        )
    print("Done (video).")

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_ckpt, args.resolution, args.batch_size, args.num_input_views, device)
    if args.random:
        run_random(args, model, device)
    elif args.multilink:
        run_multilink(args, model, device)
    elif args.view:
        run_view(args, model, device)
    elif args.video:
        run_video(args, model, device)
    else:
        raise RuntimeError("No mode selected.")

if __name__ == "__main__":
    main()
