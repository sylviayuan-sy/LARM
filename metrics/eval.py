#!/usr/bin/env python3

"""
USAGE EXAMPLES

1) PSNR
python eval.py --image --psnr \
  --eval_list ./data_sample/data.txt \
  --category_json ./data_sample/train_test_split.json \
  --render_root gt_renderings \
  --pred_root_name output_view \
  --categories ALL

2) LPIPS
python eval.py --image --lpips \
  --eval_list ./data_sample/data.txt \
  --category_json ./data_sample/train_test_split.json \
  --render_root gt_renderings \
  --pred_root_name output_view

3) CLIP similarity
python eval.py --image --clip \
  --eval_list ./data_sample/data.txt \
  --category_json ./data_sample/train_test_split.json \
  --render_root gt_renderings \
  --pred_root_name output_view

Mesh mode

4) F-score
python eval.py --mesh --fscore \
  --eval_list ./data_sample/data.txt \
  --category_json ./data_sample/train_test_split.json \
  --categories ALL \
  --gt_mesh_root ../../partnet-mobility-render-norm-0 \
  --pred_mesh_root_main ./output \
  --pred_mesh_root_fallback 0.0 \
  --threshold 0.05 --n_samples 20000 --nn_chunk 4096

5) Chamfer distance (CD)
python eval.py --mesh --cd \
  --eval_list ./data_sample/data.txt \
  --category_json ./data_sample/train_test_split.json \
  --categories ALL \
  --gt_mesh_root ../../partnet-mobility-render-norm-0 \
  --pred_mesh_root_main ./output \
  --pred_mesh_root_fallback 0.0 \
  --n_samples 30000 --nn_chunk 4096

Joint/URDF mode

6) Axis/Point/Limit errors
python eval.py --joint \
  --eval_list ./data_sample/random_metadata/data.txt \
  --category_json ./data_sample/train_test_split.json \
  --load_dir output_random \
  --joint_info_json ./data_sample/joint_info.json \
  --categories ALL

Chunking (applies to any mode)
python eval.py --image --psnr --idx 0 --chunks 8  ...
"""


import argparse
import math
import os
import os.path as osp
import re
import sys
import json
from glob import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm
import trimesh
import torch
from transformers import CLIPProcessor, CLIPModel

_LPIPS_AVAILABLE = False
try:
    import lpips
    _LPIPS_AVAILABLE = True
except Exception:
    _LPIPS_AVAILABLE = False


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_uids_from_list(txt_path: str) -> List[str]:
    uids = []
    pat = re.compile(r'(\d+_joint_\d+)')
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pat.search(line)
            if m:
                uids.append(m.group(1))
                continue
            parts = line.split("/")
            if len(parts) >= 5:
                token = "_".join(parts[4].split("_")[1:4])
                if token:
                    uids.append(token)
    return uids


def split_chunk(items: List[str], idx: int, chunks: int) -> List[str]:
    interval = int(math.ceil(len(items) / float(chunks))) if chunks > 0 else len(items)
    start = idx * interval
    end = min((idx + 1) * interval, len(items))
    return items[start:end]


def ensure_rgba_to_rgb(img: PILImage.Image) -> PILImage.Image:
    if img.mode == "RGBA":
        bg = PILImage.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    elif img.mode != "RGB":
        return img.convert("RGB")
    return img


def compute_psnr(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    pred = pred_rgb.astype(np.float32) / 255.0
    gt = gt_rgb.astype(np.float32) / 255.0
    mse = np.mean((pred - gt) ** 2)
    if mse <= 1e-12:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)

class ImageMetricRunner:
    def __init__(
        self,
        render_root: str,
        pred_root_name: str,
        gt_views_suffix: str = "1.0_1.0_1.0_0",
        pred_suffix_hi: str = "1.0_0",
        pred_suffix_lo: str = "0.0_0",
        qpos_list: List[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        num_views: int = -1,
        metric: str = "psnr",
    ):
        self.render_root = render_root
        self.pred_root_name = pred_root_name
        self.gt_views_suffix = gt_views_suffix
        self.pred_suffix_hi = pred_suffix_hi
        self.pred_suffix_lo = pred_suffix_lo
        self.qpos_list = list(qpos_list)
        self.num_views = int(num_views)
        self.metric = metric.lower()
        self.torch_device = _device()
        self._clip_model = None
        self._clip_processor = None
        self._lpips_model = None
        self._warned_lpips_fallback = False

    def _maybe_init_clip(self):
        if self._clip_model is None or self._clip_processor is None:
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.torch_device)
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _compute_clip_similarity(self, img1: PILImage.Image, img2: PILImage.Image) -> float:
        self._maybe_init_clip()
        inputs = self._clip_processor(images=[img1, img2], return_tensors="pt").to(self.torch_device)
        with torch.no_grad():
            feats = self._clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        sim = float((feats[0] @ feats[1].T).item())
        return sim

    def _maybe_init_lpips(self):
        if not _LPIPS_AVAILABLE:
            return
        if self._lpips_model is None:
            self._lpips_model = lpips.LPIPS(net='vgg').to(self.torch_device).eval()

    def _compute_lpips(self, img1: PILImage.Image, img2: PILImage.Image) -> float:
        if _LPIPS_AVAILABLE:
            self._maybe_init_lpips()
            t1 = torch.from_numpy(np.array(img1)).float().to(self.torch_device) / 255.0
            t2 = torch.from_numpy(np.array(img2)).float().to(self.torch_device) / 255.0
            if t1.ndim == 2:
                t1 = t1.unsqueeze(-1).repeat(1, 1, 3)
            if t2.ndim == 2:
                t2 = t2.unsqueeze(-1).repeat(1, 1, 3)
            t1 = t1.permute(2, 0, 1).unsqueeze(0) * 2 - 1
            t2 = t2.permute(2, 0, 1).unsqueeze(0) * 2 - 1
            with torch.no_grad():
                d = self._lpips_model(t1, t2)
            return float(d.item())
        else:
            if not self._warned_lpips_fallback:
                print("[WARN] `lpips` not found; using 1 - CLIP similarity for LPIPS.")
                self._warned_lpips_fallback = True
            sim = self._compute_clip_similarity(img1, img2)
            return float(1.0 - sim)
    
    def eval_one(self, name: str) -> Optional[float]:
        """
        Evaluate image metrics for a single object joint.
        GT pattern:   <render_root>/views_<name>_1.0_1.0_1.0_0/color_<qpos>_in_<idx>.png
        Pred pattern: <pred_root_name>/eval_<name>/images/<idx>_<qpos>.png
        """
        import re

        all_vals = []

        gt_dir = os.path.join(self.render_root, f"views_{name}_{self.gt_views_suffix}")
        pred_dir = os.path.join(self.pred_root_name, f"eval_{name}", "images")

        if not os.path.isdir(gt_dir):
            print(f"[WARN] GT folder missing: {gt_dir}")
            return None
        if not os.path.isdir(pred_dir):
            print(f"[WARN] Pred folder missing: {pred_dir}")
            return None

        def _parse_gt_index(path: str) -> int:
            m = re.search(r"_in_(\d+)\.png$", os.path.basename(path))
            return int(m.group(1)) if m else -1

        def _parse_pred_index(path: str) -> int:
            # Handles "<idx>_<qpos>.png" (e.g. "0_0.00.png")
            m = re.match(r"(\d+)_", os.path.basename(path))
            return int(m.group(1)) if m else -1

        # --- Loop through each qpos
        for qpos in self.qpos_list:
            qstr = f"{qpos:.2f}"
            gt_imgs = sorted(glob(os.path.join(gt_dir, f"color_{qstr}_in_*.png")))
            pred_imgs = sorted(glob(os.path.join(pred_dir, f"*_{qstr}.png")))

            if not gt_imgs or not pred_imgs:
                # Skip if one side missing
                continue

            # Build {index: path} maps
            gt_by_idx = {_parse_gt_index(p): p for p in gt_imgs}
            pred_by_idx = {_parse_pred_index(p): p for p in pred_imgs}

            shared_indices = sorted(set(gt_by_idx.keys()) & set(pred_by_idx.keys()))
            if not shared_indices:
                print(f"[WARN] No matching indices for {name} @ qpos {qstr}")
                continue

            # Limit to num_views if requested
            n = len(shared_indices) if self.num_views <= 0 else min(self.num_views, len(shared_indices))
            for idx in shared_indices[:n]:
                gt_path = gt_by_idx[idx]
                pred_path = pred_by_idx[idx]

                try:
                    gt = ensure_rgba_to_rgb(PILImage.open(gt_path))
                    pred = ensure_rgba_to_rgb(PILImage.open(pred_path))
                    if gt.size != pred.size:
                        pred = pred.resize(gt.size, PILImage.BILINEAR)
                except Exception as e:
                    print(f"[WARN] Failed to load pair for {name}@{qstr}[{idx}]: {e}")
                    continue

                # Compute metric
                if self.metric == "psnr":
                    val = compute_psnr(np.array(pred), np.array(gt))
                elif self.metric == "lpips":
                    val = self._compute_lpips(pred, gt)
                elif self.metric == "clip":
                    val = self._compute_clip_similarity(pred, gt)
                else:
                    val = float("nan")

                if not np.isnan(val):
                    all_vals.append(val)

        if not all_vals:
            print(f"[WARN] No valid image pairs found for {name}")
            return None

        return float(np.mean(all_vals))




def sample_surface_points(mesh: trimesh.Trimesh, n_samples: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    return np.asarray(pts, dtype=np.float32)


def nn_distances(a: np.ndarray, b: np.ndarray, block: int = 4096) -> np.ndarray:
    Na = a.shape[0]
    out = np.empty((Na,), dtype=np.float32)
    for i in range(0, Na, block):
        a_blk = a[i:i+block]
        diffs = a_blk[:, None, :] - b[None, :, :]
        d2 = np.sum(diffs * diffs, axis=2)
        out[i:i+block] = np.sqrt(np.min(d2, axis=1)).astype(np.float32)
    return out


def chamfer_and_fscore(pts_pred: np.ndarray, pts_gt: np.ndarray, threshold: float, nn_chunk: int) -> Dict[str, float]:
    d_pred_to_gt = nn_distances(pts_pred, pts_gt, block=nn_chunk)
    d_gt_to_pred = nn_distances(pts_gt, pts_pred, block=nn_chunk)
    dist1 = float(np.mean(d_pred_to_gt))
    dist2 = float(np.mean(d_gt_to_pred))
    cd = dist1 + dist2
    precision = float(np.mean(d_pred_to_gt < threshold)) if len(d_pred_to_gt) else 0.0
    recall = float(np.mean(d_gt_to_pred < threshold)) if len(d_gt_to_pred) else 0.0
    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0.0
    return {"cd": cd, "fscore": fscore, "dist1": dist1, "dist2": dist2, "precision": precision, "recall": recall}


class MeshMetricRunner:
    def __init__(
        self,
        gt_mesh_root: str,
        pred_mesh_root_main: str,
        pred_mesh_root_fallback: str,
        qpos_list: List[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        threshold: float = 0.05,
        metric: str = "fscore",
        n_samples: int = 20000,
        nn_chunk: int = 4096,
    ):
        self.gt_mesh_root = gt_mesh_root
        self.pred_mesh_root_main = pred_mesh_root_main
        self.pred_mesh_root_fallback = pred_mesh_root_fallback
        self.qpos_list = list(qpos_list)
        self.threshold = float(threshold)
        self.metric = metric.lower()
        self.n_samples = int(n_samples)
        self.nn_chunk = int(nn_chunk)
        import logging as _logging
        _logging.getLogger('trimesh').setLevel(_logging.ERROR)

    def _first_existing_dir(self, root: str, name: str) -> Optional[str]:
        cands = [
            os.path.join(root, f"eval_{name}"),
            os.path.join(root, name),
        ]
        for d in cands:
            if os.path.isdir(d):
                return d
        return None

    def _load_mesh_any(self, path: str) -> trimesh.Trimesh:
        m = trimesh.load(path, process=False)
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        return m

    def _load_pred_mesh(self, name: str, alpha: float) -> trimesh.Trimesh:
        base_dir = self._first_existing_dir(self.pred_mesh_root_main, name)
        p = os.path.join(base_dir, f"pose_{alpha:.02f}.ply")
        if os.path.exists(p):
            return self._load_mesh_any(p)
        raise FileNotFoundError(f"No pred mesh for {name} @ alpha={alpha:.2f}")

    def _load_gt_mesh(self, name: str, alpha: float) -> trimesh.Trimesh:
        base_dir = os.path.join(self.gt_mesh_root, f"views_{name}_1.0_1.0_1.0_0")
        p_obj = os.path.join(base_dir, f"mesh_{alpha:.02f}.obj")
        if os.path.exists(p_obj):
            return self._load_mesh_any(p_obj)
        raise FileNotFoundError(f"No GT mesh for {name} @ alpha={alpha:.2f}")

    def eval_one(self, name: str) -> float:
        vals = []
        for alpha in self.qpos_list:
            try:
                gt_mesh = self._load_gt_mesh(name, alpha)
                pred_mesh = self._load_pred_mesh(name, alpha)
            except Exception:
                continue
            try:
                gt_pts = sample_surface_points(gt_mesh, self.n_samples)
                pr_pts = sample_surface_points(pred_mesh, self.n_samples)
                stats = chamfer_and_fscore(pr_pts, gt_pts, threshold=self.threshold, nn_chunk=self.nn_chunk)
                if self.metric == "fscore":
                    val = stats["fscore"]
                elif self.metric == "cd":
                    val = stats["cd"]
                else:
                    raise ValueError(f"Unknown mesh metric: {self.metric}")
            except Exception:
                val = float("nan")
            if not np.isnan(val):
                vals.append(val)
            del gt_mesh, pred_mesh
        if not vals:
            return float("nan")
        return float(np.mean(vals))



def _safe_unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)[:3]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n


def _axis_angle_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    au = _safe_unit(a)
    bu = _safe_unit(b)
    d = float(np.clip(abs(np.dot(au, bu)), -1.0, 1.0))
    return float(np.degrees(np.arccos(d)))


def _line_line_distance(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray) -> float:
    d0u = _safe_unit(d0)
    d1u = _safe_unit(d1)
    w0 = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    c = np.cross(d0u, d1u)
    denom = np.linalg.norm(c)
    if denom < 1e-9:
        proj = w0 - np.dot(w0, d0u) * d0u
        return float(np.linalg.norm(proj))
    return float(abs(np.dot(w0, c)) / denom)


def _deduced_transform_matrix() -> np.ndarray:
    R = np.array([[ 0,  1, 0],
                  [ 0,  0, 1],
                  [ 1,  0, 0]], dtype=np.float64)
    return R


class JointMetricRunner:
    def __init__(self, load_dir: str, joint_info_json: str,
                 render_root: str, gt_views_suffix: str,
                 qpos_list: List[float]):
        try:
            from urdfpy import URDF
        except Exception as e:
            raise RuntimeError("urdfpy is required for --joint mode") from e
        self._URDF = URDF
        self.load_dir = load_dir
        with open(joint_info_json, "r") as f:
            self.jinfo = json.load(f)
        self.render_root = render_root
        self.gt_views_suffix = gt_views_suffix  # kept for completeness; parsing below mirrors your script
        self.qpos_list = list(qpos_list)
        self._M = _deduced_transform_matrix()

    # ---------------- GT lookup ----------------
    def _gt_entry(self, name: str) -> Optional[Dict]:
        parts = name.split("_")
        if len(parts) < 3:
            return None
        obj_id = parts[0]
        jkey = f"joint_{parts[-1]}"
        obj = self.jinfo.get(obj_id, {})
        return obj.get(jkey, None)

    # ---------------- Pred URDF reading ----------------
    def _default_pred(self) -> Dict:
        return {
            "type": "unknown",
            "axis": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "point": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "motion": 0.0,
        }

    def _read_urdf_joint(self, name: str) -> Dict:
        # Follow your second script's path convention exactly:
        #   load_dir/eval_<name>/eval_<name>.urdf
        urdf_path = osp.join(self.load_dir, f"eval_{name}", f"eval_{name}.urdf")
        if not osp.exists(urdf_path):
            return self._default_pred()
        try:
            model = self._URDF.load(urdf_path)
        except Exception:
            return self._default_pred()
        if not model.joints:
            return self._default_pred()

        j = model.joints[0]
        axis = np.array(getattr(j, "axis", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)[:3]
        origin = np.array(getattr(j, "origin", np.eye(4)), dtype=np.float64)
        if origin.shape == (4, 4):
            point = origin[:3, 3]
        else:
            point = np.zeros(3, dtype=np.float64)

        motion = 0.0
        lim = getattr(j, "limit", None)
        lower = getattr(lim, "lower", None) if lim is not None else None
        upper = getattr(lim, "upper", None) if lim is not None else None
        if (lower is not None) and (upper is not None):
            motion = float(abs(float(upper) - float(lower)))

        return {"type": j.joint_type, "axis": axis, "point": point, "motion": motion}

    # ---------------- GT meta dir ----------------
    def _gt_dir(self, name: str) -> str:
        # Follow your second script literally:
        #   <render_root>/gt_renderings/views_<name>_1.0_1.0_1.0_0
        return osp.join(self.render_root, f"gt_renderings/views_{name}_1.0_1.0_1.0_0")

    def _parse_suffix_obj_scale(self) -> np.ndarray:
        try:
            parts = [float(x) for x in self.gt_views_suffix.split("_")]
            if len(parts) >= 3:
                return np.array(parts[:3], dtype=np.float64)
        except Exception:
            pass
        return np.array([1.0, 1.0, 1.0], dtype=np.float64)

    def _load_meta_scaling(self, name: str) -> Tuple[np.ndarray, float, np.ndarray]:
        # Same logic you already use: pull obj_scale / scale / center from meta_{qpos0}.json if present
        S_vec = self._parse_suffix_obj_scale()
        uni_scale = 1.0
        center = np.zeros(3, dtype=np.float64)

        meta_dir = self._gt_dir(name)
        q0 = float(self.qpos_list[0]) if self.qpos_list else 0.0
        meta_path = osp.join(meta_dir, f"meta_{q0:.2f}.json")
        if osp.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                if "obj_scale" in meta:
                    S_vec = np.array(meta["obj_scale"], dtype=np.float64).reshape(3)
                if "scale" in meta:
                    uni_scale = float(meta["scale"])
                if "center" in meta:
                    center = np.array(meta["center"], dtype=np.float64).reshape(3)
            except Exception:
                pass
        return S_vec, uni_scale, center

    def eval_one(self, name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        gt = self._gt_entry(name)
        if gt is None:
            return None, None, None
        pr = self._read_urdf_joint(name)
        if pr is None:
            return None, None, None

        gt_axis = np.array(gt.get("joint_axis", [0, 0, 1]), dtype=np.float64)
        gt_point = np.array(gt.get("axis_point", [0, 0, 0]), dtype=np.float64)
        gt_scale = float(abs(gt.get("joint_scale", 0.0)))
        gt_type = str(gt.get("joint_type", "")).lower()

        S_vec, uni_scale, center = self._load_meta_scaling(name)
        S = np.diag(S_vec)

        axis_pred_final = _safe_unit(self._M @ pr["axis"])
        point_pred_final = self._M @ pr["point"]

        point_gt_final = uni_scale * (self._M @ (S @ gt_point)) - center
        axis_gt_for_pos = _safe_unit(S @ gt_axis)

        nref = axis_gt_for_pos
        c = float(np.dot(nref, point_gt_final))
        point_gt_aligned = point_gt_final
        point_pr_aligned = point_pred_final - nref * (np.dot(nref, point_pred_final) - c)

        if gt_type == "prismatic":
            anisotropic = np.linalg.norm(S @ _safe_unit(gt_axis))
            gt_scale_t = gt_scale * anisotropic * uni_scale
        else:
            gt_scale_t = gt_scale

        axis_ang_deg = _axis_angle_error_deg(gt_axis, axis_pred_final)
        if axis_ang_deg > 85:
            print(name)
        if gt_type == "prismatic":
            axis_pos_err = 0.0
        else:
            axis_pos_err = _line_line_distance(point_gt_aligned, axis_gt_for_pos, point_pr_aligned, axis_pred_final)
        limit_err = float(abs(pr["motion"] - gt_scale_t))

        return axis_ang_deg, axis_pos_err, limit_err



def load_category_map(category_json: str) -> Dict[str, List[str]]:
    with open(category_json, "r") as f:
        data = json.load(f)
    out = {}
    for cat, sets in data.items():
        train = sets.get("train", [])
        test = sets.get("test", [])
        out[cat] = list(sorted(set(train + test)))
    return out


# ---------------------------
# PER-JOINT aggregation / reporting
# ---------------------------

def _obj_id_from_joint_key(full_key: str) -> str:
    """
    Given a key like '12345_joint_0' (or similar), return '12345'.
    Fallback: return the first token before '_' if present, else the whole key.
    """
    return full_key.split("_")[0] if "_" in full_key else full_key


def average_per_joint(scores_by_name: Dict[str, float]) -> Dict[str, float]:
    """
    Keep scores at *joint* granularity (no collapsing to object).
    Filters out None/NaN, returns {joint_key: score}.
    """
    out: Dict[str, float] = {}
    for k, v in scores_by_name.items():
        if v is None:
            continue
        try:
            vf = float(v)
        except Exception:
            continue
        if np.isnan(vf):
            continue
        out[k] = vf
    return out


def print_category_means_by_joint(
    avg_by_joint: Dict[str, float],
    category_map: Dict[str, List[str]],
    selected_categories: List[str],
    metric_name: str,
):
    """
    For each category, compute the mean metric over all joints
    whose object id belongs to that category.
    """
    print(f"\n=== Per-Category Means (per-joint: {metric_name}) ===")
    overall_vals: List[float] = []
    found_any = False

    for cat in selected_categories:
        obj_ids = set(category_map.get(cat, []))
        cat_vals = [v for jk, v in avg_by_joint.items()
                    if _obj_id_from_joint_key(jk) in obj_ids]

        if cat_vals:
            cat_mean = float(np.mean(cat_vals))
            overall_vals.extend(cat_vals)
            print(f"{cat}: {cat_mean:.4f}  (joints n={len(cat_vals)})")
            found_any = True
        else:
            print(f"{cat}: No valid joints found.")

    if overall_vals:
        overall_mean = float(np.mean(overall_vals))
        print(f"\nTotal average across selected categories (per-joint): "
              f"{overall_mean:.4f}  (joints n={len(overall_vals)})")
    elif not found_any:
        print("\nNo matches found for any selected categories.")
        if avg_by_joint:
            mean_all = np.mean(list(avg_by_joint.values()))
            print(f"Overall mean across all evaluated joints: {mean_all:.4f}")



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    mode_group = p.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--image", action="store_true")
    mode_group.add_argument("--mesh", action="store_true")
    mode_group.add_argument("--joint", action="store_true")

    metric_group = p.add_mutually_exclusive_group(required=False)
    metric_group.add_argument("--psnr", action="store_true")
    metric_group.add_argument("--lpips", action="store_true")
    metric_group.add_argument("--clip", action="store_true")
    metric_group.add_argument("--fscore", action="store_true")
    metric_group.add_argument("--cd", action="store_true")

    p.add_argument("--datalist_path", type=str, required=True, help="Path to eval data list (was --eval_list)")
    p.add_argument("--category_json", type=str, required=True)
    p.add_argument("--categories", type=str, default="ALL")

    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--chunks", type=int, default=1)
    p.add_argument("--qpos", type=str, default="0.0,0.25,0.5,0.75,1.0")

    p.add_argument("--render_dir", type=str, default="./partnet-render",
                   help="GT render root (was --render_root)")
    p.add_argument("--load_dir", type=str, default="output_view",
                   help="Prediction folder root (was --pred_root_name)")

    # Mesh
    p.add_argument("--gt_mesh_root", type=str, default="./partnet-render")
    p.add_argument("--pred_mesh_root_main", type=str, default="./output_random")
    p.add_argument("--pred_mesh_root_fallback", type=str, default="./output_random")
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--n_samples", type=int, default=20000)
    p.add_argument("--nn_chunk", type=int, default=4096)

    # Joint
    p.add_argument("--joint_info_json", type=str, default="data_sample/joint_info.json")

    # Rendering naming conventions
    p.add_argument("--gt_views_suffix", type=str, default="1.0_1.0_1.0_0")
    p.add_argument("--pred_suffix_hi", type=str, default="1.0_0")
    p.add_argument("--pred_suffix_lo", type=str, default="0.0_0")
    p.add_argument("--num_views", type=int, default=-1)


    return p



def main():
    args = build_argparser().parse_args()
    metric = None
    if args.psnr:
        metric = "psnr"
    elif args.lpips:
        metric = "lpips"
    elif args.clip:
        metric = "clip"
    elif args.fscore:
        metric = "fscore"
    elif args.cd:
        metric = "cd"
    if args.image and metric not in {"psnr", "lpips", "clip"}:
        sys.exit("Error: --image requires --psnr/--lpips/--clip")
    if args.mesh and metric not in {"fscore", "cd"}:
        sys.exit("Error: --mesh requires --fscore/--cd")
    all_uids = load_uids_from_list(args.datalist_path)
    uids = split_chunk(all_uids, args.idx, args.chunks)
    print(f"Loaded {len(all_uids)} items; chunk {args.idx+1}/{args.chunks} → evaluating {len(uids)}.")
    qpos_list = [float(x) for x in args.qpos.split(",") if x.strip() != ""]
    if args.image:
        scores_by_name: Dict[str, float] = {}
        runner = ImageMetricRunner(
            render_root=args.render_dir,
            pred_root_name=args.load_dir,
            gt_views_suffix=args.gt_views_suffix,
            pred_suffix_hi=args.pred_suffix_hi,
            pred_suffix_lo=args.pred_suffix_lo,
            qpos_list=qpos_list,
            num_views=args.num_views,
            metric=metric,
        )
        for name in tqdm(uids, desc=f"Image-{metric}"):
            val = runner.eval_one(name)
            scores_by_name[name] = val
        avg_by_joint = average_per_joint(scores_by_name)
        category_map = load_category_map(args.category_json)
        selected = sorted(category_map.keys()) if args.categories.strip().upper() == "ALL" else [c.strip() for c in args.categories.split(",") if c.strip()]
        print_category_means_by_joint(avg_by_joint, category_map, selected, metric_name=metric)
    elif args.mesh:
        scores_by_name: Dict[str, float] = {}
        runner = MeshMetricRunner(
            gt_mesh_root=args.gt_mesh_root,
            pred_mesh_root_main=args.pred_mesh_root_main,
            pred_mesh_root_fallback=args.pred_mesh_root_fallback,
            qpos_list=qpos_list,
            threshold=args.threshold,
            metric=metric,
            n_samples=args.n_samples,
            nn_chunk=args.nn_chunk,
        )
        for name in tqdm(uids, desc=f"Mesh-{metric}"):
            val = runner.eval_one(name)
            scores_by_name[name] = val
        avg_by_joint = average_per_joint(scores_by_name)
        category_map = load_category_map(args.category_json)
        selected = sorted(category_map.keys()) if args.categories.strip().upper() == "ALL" else [c.strip() for c in args.categories.split(",") if c.strip()]
        print_category_means_by_joint(avg_by_joint, category_map, selected, metric_name=metric)
    elif args.joint:
        axis_deg_scores: Dict[str, float] = {}
        pos_err_scores: Dict[str, float] = {}
        limit_err_scores: Dict[str, float] = {}
        runner = JointMetricRunner(
            load_dir=args.load_dir,
            joint_info_json=args.joint_info_json,
            render_root=args.render_dir,
            gt_views_suffix=args.gt_views_suffix,
            qpos_list=qpos_list,
        )
        for name in tqdm(uids, desc="Joint-URDF"):
            ang, pos, lim = runner.eval_one(name)
            if ang is not None:
                axis_deg_scores[name] = float(ang)
            if pos is not None:
                pos_err_scores[name] = float(pos)
            if lim is not None:
                limit_err_scores[name] = float(lim)
        category_map = load_category_map(args.category_json)
        selected = sorted(category_map.keys()) if args.categories.strip().upper() == "ALL" else [c.strip() for c in args.categories.split(",") if c.strip()]
        avg_axis = average_per_joint(axis_deg_scores)
        print_category_means_by_joint(avg_axis, category_map, selected, metric_name="axis_angle_deg (↓ better)")
        avg_pos = average_per_joint(pos_err_scores)
        print_category_means_by_joint(avg_pos, category_map, selected, metric_name="axis_position_distance (↓ better)")
        avg_lim = average_per_joint(limit_err_scores)
        print_category_means_by_joint(avg_lim, category_map, selected, metric_name="joint_limit_magnitude_error (↓ better)")
        if axis_deg_scores:
            print(f"\nOverall mean axis angle error (per-joint): {np.mean(list(axis_deg_scores.values())):.4f} deg")
        if pos_err_scores:
            print(f"Overall mean axis position distance (per-joint): {np.mean(list(pos_err_scores.values())):.6f}")
        if limit_err_scores:
            print(f"Overall mean joint limit magnitude error (per-joint): {np.mean(list(limit_err_scores.values())):.6f}")
    else:
        sys.exit("Invalid mode selection.")


if __name__ == "__main__":
    main()
