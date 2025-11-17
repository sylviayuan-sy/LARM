#!/usr/bin/env python3
"""
Process pointclouds from rendered images + masks + depth maps,
with optional datalist.txt specifying which eval_*_joint_* objects to process.
"""

import os
import re
import json
import tqdm
import numpy as np
import trimesh
import calibur
import argparse
import matplotlib.pyplot as plt
import multiprocessing as mp

VALID_SUB_RE = re.compile(r"^eval_\d+_joint_\d+$")

def depth_to_xyz(u, v, depth, fx, fy, cx, cy):
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return x, y, z

def pmap_diff_normal(xyz: np.ndarray):
    P = xyz.astype(np.float32).reshape(512, 512, 3)
    dPdx = np.zeros_like(P)
    dPdy = np.zeros_like(P)
    dPdx[:, 1:-1] = (P[:, 2:] - P[:, :-2]) * 0.5
    dPdx[:, 0] = P[:, 1] - P[:, 0]
    dPdx[:, -1] = P[:, -1] - P[:, -2]
    dPdy[1:-1, :] = (P[2:, :] - P[:-2, :]) * 0.5
    dPdy[0, :] = P[1, :] - P[0, :]
    dPdy[-1, :] = P[-1, :] - P[-2, :]
    normals = np.cross(dPdy, dPdx).reshape(-1, 3)
    return normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)

def normalize_pcd(pcd: np.ndarray):
    mi, ma = pcd.min(), pcd.max()
    return np.array([mi, ma]), (pcd - mi) / (ma - mi + 1e-12) - 0.5

_STEM_RX = re.compile(r'^(?P<idx>\d+)(?:_(?P<qpos>[+-]?\d+(?:\.\d+)?))?$')

def _list_frame_stems(images_dir: str, max_views=None, filter_qpos: float = None, eps: float = 1e-6):
    items = []
    for fn in os.listdir(images_dir):
        if not fn.endswith('.png'):
            continue
        if fn.endswith('_mask.png') or fn.endswith('_partmask.png') or fn.endswith('_sum_partmask.png'):
            continue
        base = fn[:-4]
        m = _STEM_RX.match(base)
        if not m:
            continue
        idx = int(m.group('idx'))
        qpos_s = m.group('qpos')
        if filter_qpos is not None:
            if qpos_s is None:
                continue
            try:
                qpos_val = float(qpos_s)
            except ValueError:
                continue
            if abs(qpos_val - filter_qpos) > eps:
                continue
        items.append((idx, base))
    items.sort(key=lambda x: x[0])
    if max_views is not None:
        items = items[:max_views]
    return items

def _first_existing(path_candidates):
    for p in path_candidates:
        if p and os.path.exists(p):
            return p
    return None

def _load_mask_bool(path: str):
    m = plt.imread(path)
    if m.ndim == 3:
        m = m[..., 0]
    m = m.astype(np.float32)
    return (m > 0.5)[..., None]

def process_single(base_dir, out_dir, num_views=4, qpos_only: float = 1.0, rot_aug: bool = False):
    with open(os.path.join(base_dir, 'transforms.json')) as f:
        transforms = json.load(f)

    fx, fy, cx, cy = transforms['fx'], transforms['fy'], transforms['cx'], transforms['cy']
    images_dir = os.path.join(base_dir, 'images')

    frame_items = _list_frame_stems(
        images_dir,
        max_views=(None if num_views == -1 else num_views),
        filter_qpos=qpos_only
    )
    if not frame_items:
        return

    pcd_base, pcd_part, pcd_mbase = [], [], []
    nor_base, nor_part, nor_mbase = [], [], []

    for frame_idx, stem in frame_items:
        rgb_path = _first_existing([
            os.path.join(images_dir, f"{stem}.png"),
            os.path.join(images_dir, f"{frame_idx}.png"),
        ])
        mask_path = _first_existing([
            os.path.join(images_dir, f"{stem}_mask.png"),
            os.path.join(images_dir, f"{frame_idx}_mask.png"),
        ])
        partmask_path = _first_existing([
            os.path.join(images_dir, f"{stem}_partmask.png"),
            os.path.join(images_dir, f"{frame_idx}_partmask.png"),
        ])
        depth_path = _first_existing([
            os.path.join(images_dir, f"{stem}_depth.npy"),
            os.path.join(images_dir, f"{frame_idx}_depth.npy"),
        ])
        sumpart_path = _first_existing([
            os.path.join(images_dir, f"{stem}_sum_partmask.png"),
        ])

        if rgb_path is None or mask_path is None or partmask_path is None or depth_path is None:
            continue

        rgb = plt.imread(rgb_path)
        mask_bool = _load_mask_bool(mask_path)
        part_mask_bool = _load_mask_bool(partmask_path)

        fg_mask = mask_bool
        part_mask = part_mask_bool & fg_mask
        base_mask = fg_mask & (~part_mask)

        mbase_mask = None
        if sumpart_path is not None:
            sumpart_bool = _load_mask_bool(sumpart_path)
            mbase_mask = fg_mask & (~sumpart_bool)

        rgba = np.concatenate([rgb[..., :3], mask_bool.astype(np.float32)], axis=-1)

        depth = np.load(depth_path).reshape(-1) + 0.5 if rot_aug else np.load(depth_path).reshape(-1) * 5.0
        u, v, _ = calibur.unbind(calibur.get_dx_viewport_rays(512, 512, 0.0)[0], -1)
        x, y, z = depth_to_xyz(u, v, depth, fx, fy, cx, cy)
        xyz = np.stack([x, y, z], axis=-1)
        normals = pmap_diff_normal(xyz)

        base_mask_flat = base_mask.reshape(-1)
        part_mask_flat = part_mask.reshape(-1)
        mbase_mask_flat = mbase_mask.reshape(-1) if mbase_mask is not None else None

        for mask_bin, pcd_list, nor_list in [
            (base_mask_flat, pcd_base, nor_base),
            (part_mask_flat, pcd_part, nor_part),
        ]:
            if mask_bin is not None and mask_bin.any():
                pc = trimesh.PointCloud(
                    xyz[mask_bin],
                    colors=(np.clip(rgba.reshape(-1, 4), 0, 1) * 255).astype(np.uint8)[mask_bin]
                )
                cam_matrix = np.array(transforms['frames'][frame_idx]['transform_matrix'], dtype=np.float32)
                pose_cv = calibur.convert_pose(cam_matrix, calibur.CC.GL, calibur.CC.CV)
                pc_t = pc.copy()
                pc_t.apply_transform(pose_cv)
                normal_trans = calibur.transform_vector(normals[mask_bin], pose_cv)
                pcd_list.append(pc_t)
                nor_list.append(normal_trans)

        if mbase_mask_flat is not None and mbase_mask_flat.any():
            pc = trimesh.PointCloud(
                xyz[mbase_mask_flat],
                colors=(np.clip(rgba.reshape(-1, 4), 0, 1) * 255).astype(np.uint8)[mbase_mask_flat]
            )
            cam_matrix = np.array(transforms['frames'][frame_idx]['transform_matrix'], dtype=np.float32)
            pose_cv = calibur.convert_pose(cam_matrix, calibur.CC.GL, calibur.CC.CV)
            pc_t = pc.copy()
            pc_t.apply_transform(pose_cv)
            normal_trans = calibur.transform_vector(normals[mbase_mask_flat], pose_cv)
            pcd_mbase.append(pc_t)
            nor_mbase.append(normal_trans)

    for name, pcd, normals in [
        ('base', pcd_base, nor_base),
        ('part', pcd_part, nor_part),
        ('multilink_base', pcd_mbase, nor_mbase),
    ]:
        if len(pcd) == 0:
            continue
        out_subdir = os.path.join(out_dir, os.path.basename(base_dir) + f'_{name}')
        os.makedirs(out_subdir, exist_ok=True)

        all_points = np.concatenate([pc.vertices for pc in pcd], axis=0)
        if all_points.size == 0:
            continue
        all_colors = np.concatenate([pc.colors for pc in pcd], axis=0)
        all_normals = np.concatenate(normals, axis=0)

        nbox, normed_pcd = normalize_pcd(all_points)
        np.savez(
            os.path.join(out_subdir, 'pointcloud.npz'),
            points=normed_pcd,
            colors=all_colors,
            normals=all_normals,
            nbox=nbox
        )

def generate_test_list(out_root: str, list_name: str = "test.lst", require_npz: bool = True):
    if not os.path.exists(out_root):
        return
    names = []
    for entry in os.listdir(out_root):
        if not entry.startswith("eval_"):
            continue
        full = os.path.join(out_root, entry)
        if not os.path.isdir(full):
            continue
        if require_npz and not os.path.exists(os.path.join(full, "pointcloud.npz")):
            continue
        names.append(entry)
    names.sort()
    out_path = os.path.join(out_root, list_name)
    with open(out_path, "w") as f:
        f.write("\n".join(names))
        if names:
            f.write("\n")

def _worker(args):
    base_dir, out_dir, num_views, qpos_only, rot_aug = args
    try:
        process_single(base_dir, out_dir, num_views=num_views, qpos_only=qpos_only, rot_aug=rot_aug)
        return True
    except Exception as e:
        return f"{os.path.basename(base_dir)}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Process pointclouds from datalist")
    parser.add_argument('--load_dir', type=str, default="./output_multilink",
                        help="Root containing eval_*_joint_* subfolders.")
    parser.add_argument('--out_dir', type=str, default=None,
                        help="If not set, defaults to <load_dir>/sap_in.")
    parser.add_argument('--datalist_path', type=str, required=True,
                        help="Path to datalist.txt containing JSONs (e.g. ./data_sample_multilink/random_metadata/44817_joint_0.json).")
    parser.add_argument('--workers', type=int, default=min(36, mp.cpu_count()))
    parser.add_argument('--num_views', type=int, default=16)
    parser.add_argument('--qpos_only', type=float, default=1.0)
    parser.add_argument('--rot-aug', action='store_true',
                        help="Use rotation augmentation mode: depth + 0.5 instead of depth * 5.0")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.load_dir, "sap_in")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[info] load_dir = {args.load_dir}")
    print(f"[info] out_dir  = {out_dir}")
    print(f"[info] datalist = {args.datalist_path}")

    with open(args.datalist_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    subs = []
    for line in lines:
        m = re.search(r"(\d+)_joint_(\d+)", line)
        if not m:
            continue
        obj_id, joint_id = m.groups()
        sub = os.path.join(args.load_dir, f"eval_{obj_id}_joint_{joint_id}")
        if os.path.isdir(sub):
            subs.append(sub)
        else:
            print(f"[WARN] Missing folder for {line}: {sub}")

    if not subs:
        print("[WARN] No valid subfolders found matching datalist.")
        return

    num_views = None if args.num_views == -1 else args.num_views
    rot_aug = getattr(args, 'rot_aug', False)
    jobs = [(sub, out_dir, num_views, args.qpos_only, rot_aug) for sub in subs]

    if args.workers <= 1:
        for job in tqdm.tqdm(jobs, total=len(jobs)):
            res = _worker(job)
            if res is not True:
                print(f"[WARN] {res}")
    else:
        with mp.Pool(processes=args.workers) as pool:
            for res in tqdm.tqdm(pool.imap_unordered(_worker, jobs), total=len(jobs)):
                if res is not True:
                    print(f"[WARN] {res}")

    generate_test_list(out_dir, list_name="test.lst", require_npz=True)


if __name__ == '__main__':
    main()
