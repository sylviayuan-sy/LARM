#!/usr/bin/env python3
# Consolidated pipeline: (1) unnormalize meshes -> (2) swap URDF mesh refs & copy PLYs -> (3) multiview color projection

import os
import re
import argparse
import shutil
import json
from glob import glob

import numpy as np
import numpy
import trimesh
import tqdm

import torch
from PIL import Image
import matplotlib.pyplot as plotlib

import calibur
from color_projection import multiview_color_projection, pix2faces_renderer
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, TexturesVertex


# =========================
# ===== Utilities =========
# =========================

VALID_SUB_RE = re.compile(r"^eval_\d+_joint_\d+$")
QPOS_RX = re.compile(r"_(?P<qpos>[+-]?\d+(?:\.\d+)?)\.png$", re.IGNORECASE)

def as_mesh(obj):
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if isinstance(obj, trimesh.Scene):
        g = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        return trimesh.util.concatenate(g) if g else None
    if isinstance(obj, (list, tuple)):
        g = [g for g in obj if isinstance(g, trimesh.Trimesh)]
        return trimesh.util.concatenate(g) if g else None
    return None

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _first_existing_dir(candidates):
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return None

def _scan_for_off_dir(root):
    """Find a directory under root that contains <sub>_{base,part}.off files."""
    hits = glob(os.path.join(root, "**", "*_base.off"), recursive=True)
    if not hits:
        return None
    parent_counts = {}
    for h in hits:
        parent = os.path.dirname(h)
        parent_counts[parent] = parent_counts.get(parent, 0) + 1
    return max(parent_counts.items(), key=lambda kv: kv[1])[0]

def infer_paths(load_dir, overrides=None):
    """
    Infer all pipeline paths from load_dir.
    You can pass a dict of overrides to force a path.
    """
    overrides = overrides or {}

    # 1) mesh_dir inference
    mesh_dir_candidates = [
        overrides.get("mesh_dir"),
        os.path.join(load_dir, "sap_out", "generation", "meshes"),
        os.path.join(load_dir, "generation", "meshes"),
        _scan_for_off_dir(load_dir),
    ]
    mesh_dir = _first_existing_dir(mesh_dir_candidates)

    # 2) sap_dir inference (where <sub>_<part>/pointcloud.npz live)
    sap_dir_candidates = [
        overrides.get("sap_dir"),
        os.path.join(load_dir, "sap_in"),
        load_dir,
    ]
    sap_dir = _first_existing_dir(sap_dir_candidates)

    # 3) urdf_src_dir is typically the load root (contains <sub>/<sub>.urdf)
    urdf_src_dir = overrides.get("urdf_src_dir") or load_dir

    # 4) output dirs under load_dir (can be overridden)
    ply_out_dir       = overrides.get("ply_out_dir")       or os.path.join(load_dir, "sap_unnorm")
    urdf_swap_out_dir = overrides.get("urdf_swap_out_dir") or os.path.join(load_dir, "sap_urdf_swap")
    final_out_dir     = overrides.get("final_out_dir")     or os.path.join(load_dir, "sap_urdf_final")

    return {
        "mesh_dir": mesh_dir,
        "sap_dir": sap_dir,
        "urdf_src_dir": urdf_src_dir,
        "ply_out_dir": ply_out_dir,
        "urdf_swap_out_dir": urdf_swap_out_dir,
        "final_out_dir": final_out_dir,
    }


# =========================
# ===== Stage 1: PLYs =====
# =========================

def unnormalize_mesh(mesh: trimesh.Trimesh, bbox):
    mi, ma = bbox  # allow scalars or 3-vectors; numpy will broadcast
    vertices = (mesh.vertices + 0.5) * (ma - mi) + mi
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)

def stage1_unnormalize_to_ply(load_dir: str, sap_dir: str, mesh_dir: str, out_dir: str):
    """
    For each <sub> in load_dir matching eval_*_joint_*, read normalized OFF meshes and bbox from pointcloud.npz,
    unnormalize and write PLYs to out_dir.
    """
    ensure_dir(out_dir)
    subs = [s for s in os.listdir(load_dir) if VALID_SUB_RE.match(s)]

    if mesh_dir is None:
        print("[WARN] Could not infer mesh_dir (no *_base.off found). Stage 1 will skip all.")
        return subs
    if sap_dir is None:
        print("[WARN] Could not infer sap_dir. Stage 1 may skip entries if pointcloud.npz is missing.")

    for sub in tqdm.tqdm(subs, desc="[Stage 1] Unnormalize OFF -> PLY"):
        for part in ['base', 'part']:
            try:
                mesh_path = os.path.join(mesh_dir, f"{sub}_{part}.off")
                npz_path  = os.path.join(sap_dir,  f"{sub}_{part}", "pointcloud.npz")
                out_path  = os.path.join(out_dir,  f"{sub}_{part}.ply")

                if not os.path.isfile(mesh_path):
                    raise FileNotFoundError(f"OFF not found: `{mesh_path}`")
                if not os.path.isfile(npz_path):
                    raise FileNotFoundError(f"NPZ not found: `{npz_path}`")

                mesh = as_mesh(trimesh.load(mesh_path, force='mesh'))
                if mesh is None:
                    raise FileNotFoundError(f"Invalid mesh: `{mesh_path}`")

                nbox = np.load(npz_path)["nbox"]  # accept (2,), (2,3), etc. via broadcasting
                unnormalized = unnormalize_mesh(mesh, nbox)
                unnormalized.export(out_path)
            except Exception as e:
                print(f"[WARN] Skipping {sub}_{part}: {e}")
    return subs


# ==========================================
# ===== Stage 2: URDF mesh path update =====
# ==========================================

def update_urdf_paths(urdf_text: str, sub: str) -> str:
    urdf_new = re.sub(r'(\S*base\S*\.obj)', f"{sub}_base.ply", urdf_text)
    if urdf_new == urdf_text:
        base_name = sub.replace('eval_', '').replace('_1.0', '_mesh_1.0')
        urdf_new = urdf_text.replace(f"{base_name}_pred_base_smooth.obj", f"{sub}_base.ply")

    urdf_final = re.sub(r'(\S*part\S*\.obj)', f"{sub}_part.ply", urdf_new)
    if urdf_final == urdf_new:
        base_name = sub.replace('eval_', '').replace('_1.0', '_mesh_1.0')
        urdf_final = urdf_new.replace(f"{base_name}_pred_part_smooth.obj", f"{sub}_part.ply")

    if urdf_final == urdf_text:
        raise AssertionError(f"Mesh names not found in URDF for {sub}")
    return urdf_final

def stage2_swap_urdf_and_copy_meshes(urdf_src_dir: str, ply_in_dir: str, urdf_out_dir: str, subs_hint=None):
    ensure_dir(urdf_out_dir)

    ply_bases = glob(os.path.join(ply_in_dir, "*_base.ply"))
    subs = sorted(set(os.path.basename(p).replace("_base.ply", "") for p in ply_bases))
    if subs_hint:
        subs = [s for s in subs if s in set(subs_hint)]

    for sub in tqdm.tqdm(subs, desc="[Stage 2] Swap URDF refs & copy PLYs"):
        try:
            urdf_path = os.path.join(urdf_src_dir, sub, f"{sub}.urdf")
            if not os.path.isfile(urdf_path):
                raise FileNotFoundError(f"URDF not found: `{urdf_path}`")

            with open(urdf_path, 'r') as f:
                urdf_text = f.read()

            updated_urdf = update_urdf_paths(urdf_text, sub)

            out_subdir = ensure_dir(os.path.join(urdf_out_dir, sub))
            with open(os.path.join(out_subdir, "mobility.urdf"), 'w') as f:
                f.write(updated_urdf)

            for part in ["base", "part"]:
                src = os.path.join(ply_in_dir, f"{sub}_{part}.ply")
                dst = os.path.join(out_subdir, f"{sub}_{part}.ply")
                if not os.path.isfile(src):
                    raise FileNotFoundError(f"PLY not found: `{src}`")
                shutil.copyfile(src, dst)

        except (FileNotFoundError, AssertionError) as e:
            print(f"[WARN] Skipping {sub}: {e}")
            continue


# =========================================
# ===== Stage 3: Color projection PLY =====
# =========================================

# restrict to frames where qpos ~= this value
QPOS_TARGET = 1.0
QPOS_EPS = 1e-6

weights = [2.0, 0.3, 0.5, 0.5, 0.2, 0.2]

def fps(xyzs):
    angles = [
        [3, 0, 0], [-3, 0, 0], [0, 3, 0],
        [0, -3, 0], [0, 0, -3], [0, 0, 3],
    ]
    ids = []
    for r in angles:
        d = numpy.linalg.norm(xyzs - numpy.array(r), axis=-1)
        ids.append(numpy.argmin(d))
    return ids

def _intrinsics_from_transforms(transforms, sample_image_path):
    if all(k in transforms for k in ("fx", "fy", "cx", "cy")):
        fx, fy = transforms["fx"], transforms["fy"]
        cx, cy = transforms["cx"], transforms["cy"]
        if "W" in transforms and "H" in transforms:
            W, H = transforms["W"], transforms["H"]
        else:
            im = plotlib.imread(sample_image_path)
            H, W = im.shape[0], im.shape[1]
        return fx, fy, cx, cy, (H, W)

    im = plotlib.imread(sample_image_path)
    H, W = im.shape[0], im.shape[1]
    if "camera_angle_x" in transforms:
        fovx = transforms["camera_angle_x"]
        fx = 0.5 * W / np.tan(0.5 * fovx)
    else:
        raise KeyError("No intrinsics and no camera_angle_x found in transforms.json")
    fy = fx
    cx, cy = W / 2.0, H / 2.0
    return fx, fy, cx, cy, (H, W)

def _resolve_frame_image(base_dir, frame_entry):
    images_dir = os.path.join(base_dir, "images")

    if "file_path" in frame_entry and frame_entry["file_path"]:
        fp = frame_entry["file_path"]
        if fp.startswith("./"):
            fp = fp[2:]
        cand = os.path.join(base_dir, fp)
        if os.path.isfile(cand):
            return cand
        if not cand.lower().endswith(".png") and os.path.isfile(cand + ".png"):
            return cand + ".png"

        base = os.path.basename(fp)
        cand2 = os.path.join(images_dir, base)
        if os.path.isfile(cand2):
            return cand2
        if not base.lower().endswith(".png") and os.path.isfile(cand2 + ".png"):
            return cand2 + ".png"

    idx = frame_entry.get("idx", None)
    qpos = frame_entry.get("qpos", None)

    if idx is not None and qpos is not None:
        name = f"{idx}_{qpos:.02f}.png"
        p = os.path.join(images_dir, name)
        if os.path.isfile(p):
            return p

    if idx is not None:
        p = os.path.join(images_dir, f"{idx}.png")
        if os.path.isfile(p):
            return p

    pngs = sorted(glob(os.path.join(images_dir, "*.png")))
    if pngs:
        return pngs[0]

    raise FileNotFoundError(f"Cannot resolve image path under '{images_dir}'")

def _parse_qpos_from_path(path):
    m = QPOS_RX.search(os.path.basename(path))
    if not m:
        return None
    try:
        return float(m.group("qpos"))
    except Exception:
        return None

def _frame_matches_qpos(base_dir, frame_entry, target=QPOS_TARGET, eps=QPOS_EPS):
    # 1) Prefer explicit qpos in frame entry
    fq = frame_entry.get("qpos", None)
    if fq is not None:
        try:
            return abs(float(fq) - target) <= eps
        except Exception:
            return False
    # 2) Otherwise try deducing from file path
    try:
        p = _resolve_frame_image(base_dir, frame_entry)
        q = _parse_qpos_from_path(p)
        return (q is not None) and (abs(q - target) <= eps)
    except Exception:
        return False

def _filter_frames_by_qpos(base_dir, frames, target=QPOS_TARGET, eps=QPOS_EPS):
    return [f for f in frames if _frame_matches_qpos(base_dir, f, target, eps)]

def stage3_colorize_from_multiview(urdf_in_dir: str, load_dir: str, out_dir_root: str):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.set_grad_enabled(False)

    eval_list = [d for d in os.listdir(urdf_in_dir)
                 if os.path.isdir(os.path.join(urdf_in_dir, d)) and VALID_SUB_RE.match(d)]
    for eval_id in tqdm.tqdm(eval_list, desc="[Stage 3] Multiview color projection"):
        try:
            base_dir = os.path.join(load_dir, eval_id)
            out_dir = ensure_dir(os.path.join(out_dir_root, eval_id))

            base_p = as_mesh(trimesh.load(f'{urdf_in_dir}/{eval_id}/{eval_id}_base.ply', force='mesh'))
            part_p = as_mesh(trimesh.load(f'{urdf_in_dir}/{eval_id}/{eval_id}_part.ply', force='mesh'))
            if base_p is None or part_p is None:
                raise FileNotFoundError("Missing or invalid base/part PLY")

            base_p = base_p.subdivide().subdivide()
            part_p = part_p.subdivide().subdivide()

            tjson = os.path.join(base_dir, 'transforms.json')
            with open(tjson) as fi:
                transforms = json.load(fi)

            # --- filter frames to qpos ~= 1.0 everywhere ---
            frames_all = transforms.get('frames', [])
            frames_q1 = _filter_frames_by_qpos(base_dir, frames_all, target=QPOS_TARGET, eps=QPOS_EPS)
            if not frames_q1:
                print(f"[WARN] {eval_id}: no frames with qpos≈{QPOS_TARGET}. Skipping.")
                continue

            # sample image for intrinsics & avg color must be qpos==1
            sample_frame = frames_q1[0]
            sample_img_path = _resolve_frame_image(base_dir, sample_frame)

            fx, fy, cx, cy, (H, W) = _intrinsics_from_transforms(transforms, sample_img_path)

            rgb0 = plotlib.imread(sample_img_path)
            mask0_path = os.path.splitext(sample_img_path)[0] + "_mask.png"
            if os.path.isfile(mask0_path):
                mask0 = plotlib.imread(mask0_path)[..., None]
                mask0 = (mask0 > 0.5).astype(np.float32)
            else:
                mask0 = np.ones_like(rgb0[..., :1], dtype=np.float32)
            denom = max(mask0.sum(), 1.0)
            avg_color = (rgb0[..., :3] * mask0).reshape(-1, 3).sum(0) / denom
            avg_color = np.clip(avg_color, 0.0, 1.0)

            verts = torch.from_numpy(
                np.vstack([base_p.vertices, part_p.vertices]).astype(np.float32)
            ).to(device)
            f_base = torch.from_numpy(base_p.faces.astype(np.int64)).to(device)
            f_part = torch.from_numpy(part_p.faces.astype(np.int64)).to(device) + len(base_p.vertices)
            faces = torch.cat([f_base, f_part], dim=0)

            init_tex = torch.from_numpy(
                np.tile(avg_color[None, :], (verts.shape[0], 1)).astype(np.float32)
            ).to(device)
            mesh = Meshes(
                verts=[verts],
                faces=[faces],
                textures=TexturesVertex(verts_features=[init_tex]),
            )

            cams, imgs, cam_xyzs, debug_rows = [], [], [], []
            # candidate viewpoints only from qpos==1 frames
            for frame in frames_q1:
                cam_matrix_gl = numpy.array(frame['transform_matrix'], dtype=numpy.float32)
                cam_xyzs.append(cam_matrix_gl[:3, 3])

            # pick up to 6 diverse qpos==1 views
            if len(cam_xyzs) == 0:
                print(f"[WARN] {eval_id}: qpos≈{QPOS_TARGET} frames have no camera data. Skipping.")
                continue
            sel_ids = fps(numpy.array(cam_xyzs)) if len(cam_xyzs) >= 6 else list(range(len(cam_xyzs)))

            for i in sel_ids:
                frame = frames_q1[i]
                rgb_path = _resolve_frame_image(base_dir, frame)
                # sanity: enforce qpos in file name too
                q_infer = _parse_qpos_from_path(rgb_path)
                if (q_infer is None) or (abs(q_infer - QPOS_TARGET) > QPOS_EPS):
                    # skip any accidental non-1.0 images
                    continue

                rgb_i = plotlib.imread(rgb_path)
                mask_path = os.path.splitext(rgb_path)[0] + "_mask.png"
                if os.path.isfile(mask_path):
                    mask_i = plotlib.imread(mask_path)[..., None]
                    mask_i = np.clip(mask_i, 0, 1)
                else:
                    mask_i = np.ones_like(rgb_i[..., :1], dtype=np.float32)

                rgba = numpy.concatenate([rgb_i[..., :3], mask_i], axis=-1)
                imgs.append(Image.fromarray(numpy.clip(rgba * 255, 0, 255).astype(numpy.uint8), mode='RGBA'))

                cam_matrix_gl = numpy.array(frame['transform_matrix'], dtype=numpy.float32)
                cam_matrix_p3d = numpy.linalg.inv(calibur.convert_pose(cam_matrix_gl, calibur.CC.GL, ('-X', 'Y', 'Z')))

                cams.append(PerspectiveCameras(
                    focal_length=((fx, fy),),
                    principal_point=((cx, cy),),
                    R=torch.tensor(cam_matrix_p3d[:3, :3].T[None]).float().to(device),
                    T=torch.tensor(cam_matrix_p3d[:3, 3][None]).float().to(device),
                    in_ndc=False,
                    image_size=((H, W),),
                    device=device
                ))

                d = pix2faces_renderer.render_pix2faces_nvdiff(mesh, cams[-1], return_rast=True)[..., 2].squeeze().detach().cpu().numpy()
                cmap = plotlib.get_cmap('viridis')
                combo = numpy.concatenate([rgb_i[..., :3], cmap((d - d.min()) / max(d.max() - d.min(), 1e-6))[..., :3]], axis=1)
                debug_rows.append(combo)

            if len(imgs) == 0:
                print(f"[WARN] {eval_id}: no usable qpos≈{QPOS_TARGET} views after filtering. Skipping.")
                continue

            if debug_rows:
                plotlib.imsave(f'{out_dir}/debug_depth_stack.png',
                               numpy.clip(numpy.concatenate(debug_rows), 0, 1))

            mesh_col = multiview_color_projection(
                mesh, imgs, cams, complete_unseen=True, resolution=512, weights=weights
            )
            tex = mesh_col.textures
            if isinstance(tex, TexturesVertex):
                x0 = tex.verts_features_list()[0].detach().cpu().numpy()
            else:
                raise RuntimeError("Unexpected texture type after projection")

            nb = len(base_p.vertices)
            base_p_c = trimesh.Trimesh(base_p.vertices, base_p.faces, vertex_colors=x0[:nb], process=False)
            part_p_c = trimesh.Trimesh(part_p.vertices, part_p.faces, vertex_colors=x0[nb:], process=False)

            shutil.copyfile(f'{urdf_in_dir}/{eval_id}/mobility.urdf', f'{out_dir}/mobility.urdf')
            base_p_c.export(f'{out_dir}/{eval_id}_base.ply')
            part_p_c.export(f'{out_dir}/{eval_id}_part.ply')
        except Exception as e:
            print(f"[WARN] Skipping {eval_id}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Sequential pipeline: unnormalize OFF -> PLY, update URDF refs, then multiview color projection."
    )
    parser.add_argument("--load_dir", type=str, default="./output",
                        help="Root directory containing eval_*_joint_* folders, URDFs, images, etc.")

    # Optional overrides (rarely needed – leave unset to auto-infer)
    parser.add_argument("--sap_dir", type=str, default=None)
    parser.add_argument("--mesh_dir", type=str, default=None)
    parser.add_argument("--ply_out_dir", type=str, default=None)
    parser.add_argument("--urdf_src_dir", type=str, default=None)
    parser.add_argument("--urdf_swap_out_dir", type=str, default=None)
    parser.add_argument("--final_out_dir", type=str, default=None)

    args = parser.parse_args()

    paths = infer_paths(
        args.load_dir,
        overrides={
            "sap_dir": args.sap_dir,
            "mesh_dir": args.mesh_dir,
            "ply_out_dir": args.ply_out_dir,
            "urdf_src_dir": args.urdf_src_dir,
            "urdf_swap_out_dir": args.urdf_swap_out_dir,
            "final_out_dir": args.final_out_dir,
        }
    )

    print("\n=== Resolved paths ===")
    for k, v in paths.items():
        print(f"{k:>18}: {v}")
    print("======================\n")

    subs = stage1_unnormalize_to_ply(
        load_dir=args.load_dir,
        sap_dir=paths["sap_dir"],
        mesh_dir=paths["mesh_dir"],
        out_dir=paths["ply_out_dir"],
    )

    stage2_swap_urdf_and_copy_meshes(
        urdf_src_dir=paths["urdf_src_dir"],
        ply_in_dir=paths["ply_out_dir"],
        urdf_out_dir=paths["urdf_swap_out_dir"],
        subs_hint=subs
    )

    stage3_colorize_from_multiview(
        urdf_in_dir=paths["urdf_swap_out_dir"],
        load_dir=args.load_dir,
        out_dir_root=paths["final_out_dir"]
    )


if __name__ == "__main__":
    main()
