#!/usr/bin/env python3
import os
import cv2
import numpy as np
from glob import glob
from collections import defaultdict
import argparse

def parse_args():
    ap = argparse.ArgumentParser(description="Build union (sum_partmask) across joints for each object and qpos.")
    ap.add_argument('--load_dir', type=str, required=True,
                    help='Directory containing eval_<obj>_joint_<idx>/images with *_partmask.png')
    ap.add_argument('--qpos_list', type=str, default="0.00,0.25,0.50,0.75,1.00",
                    help='Comma-separated qpos values (e.g. "0.00,0.25,0.50,0.75,1.00")')
    return ap.parse_args()

def main():
    args = parse_args()
    load_dir = args.load_dir
    qpos_list = [f"{float(x):.2f}" for x in args.qpos_list.split(",") if x.strip() != ""]

    # All joint folders: eval_<obj>_joint_<joint>
    all_joint_folders = sorted(glob(os.path.join(load_dir, "eval_*_joint_*")))
    if not all_joint_folders:
        print(f"[WARN] No joint folders under {load_dir}")
        return

    # Group by object id
    objects = {}
    for jf in all_joint_folders:
        base = os.path.basename(jf)
        parts = base.split("_")
        if len(parts) < 4 or parts[0] != "eval" or parts[2] != "joint":
            continue
        obj_id = parts[1]
        joint_id = parts[3]
        objects.setdefault(obj_id, []).append((int(joint_id), jf))

    for obj_id, joint_list in sorted(objects.items(), key=lambda kv: int(kv[0])):
        # Sort joints, pick the lowest joint to host outputs (canonical place)
        joint_list.sort(key=lambda t: t[0])
        out_host_joint_id, out_host_joint_dir = joint_list[0]
        out_images_dir = os.path.join(out_host_joint_dir, "images")
        os.makedirs(out_images_dir, exist_ok=True)

        # Build map: qpos -> idx -> [mask_paths from all joints]
        qpos_idx_to_paths = {q: defaultdict(list) for q in qpos_list}

        for _, joint_dir in joint_list:
            image_dir = os.path.join(joint_dir, "images")
            if not os.path.isdir(image_dir):
                continue
            # find *_partmask.png
            for mask_path in glob(os.path.join(image_dir, "*_partmask.png")):
                stem = os.path.splitext(os.path.basename(mask_path))[0]  # e.g. "12_1.00_partmask"
                parts = stem.split("_")
                if len(parts) < 3:
                    continue
                idx = parts[0]
                qpos_token = parts[1]
                try:
                    qpos_norm = f"{float(qpos_token):.2f}"
                except Exception:
                    continue
                if qpos_norm not in qpos_list:
                    continue
                qpos_idx_to_paths[qpos_norm][idx].append(mask_path)

        # Write union masks
        for qpos in qpos_list:
            idx_map = qpos_idx_to_paths[qpos]
            if not idx_map:
                continue
            for idx, paths in idx_map.items():
                if not paths:
                    continue
                combined = None
                for p in paths:
                    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        continue
                    # binarize then OR
                    mb = (m > 127).astype(np.uint8)
                    if combined is None:
                        combined = mb
                    else:
                        combined |= mb
                if combined is None:
                    continue
                out_name = f"{idx}_{qpos}_sum_partmask.png"
                out_path = os.path.join(out_images_dir, out_name)
                cv2.imwrite(out_path, (combined * 255).astype(np.uint8))
                print(f"[OK] {obj_id}: saved {out_path}")

if __name__ == "__main__":
    main()
