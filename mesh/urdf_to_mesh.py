#!/usr/bin/env python3
import os
import argparse
from glob import glob

import numpy as np
import trimesh
from urdfpy import URDF
from tqdm import tqdm

JOINT_NAME = "joint"


def parse_qpos_list(s: str):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def find_urdfs(load_dir: str, mode: str):
    if mode == "sap":
        pattern = os.path.join(load_dir, "sap_urdf_final", "eval_*_joint_*", "mobility.urdf")
        urdfs = sorted(glob(pattern))
    elif mode == "tsdf":
        candidate_dirs = sorted(glob(os.path.join(load_dir, "eval_*_joint_0")))
        urdfs = []
        for d in candidate_dirs:
            base = os.path.basename(d)
            candidate = os.path.join(d, f"{base}.urdf")
            if os.path.exists(candidate):
                urdfs.append(candidate)
    else:
        raise ValueError("mode must be 'sap' or 'tsdf'")
    return urdfs


def export_joint_poses_as_objs(urdf_path: str, mesh_root: str, output_root: str, qsteps: list[float]):
    robot = URDF.load(urdf_path)
    urdf_parent = os.path.basename(os.path.dirname(urdf_path))
    out_dir = os.path.join(output_root, urdf_parent)
    os.makedirs(out_dir, exist_ok=True)
    obj_name = os.path.basename(out_dir)

    joint = next((j for j in robot.joints if j.name == JOINT_NAME), None)
    if joint is None:
        raise ValueError(f"Joint '{JOINT_NAME}' not found in {urdf_path}")

    lower = joint.limit.lower if joint.limit and joint.limit.lower is not None else -1.0
    upper = joint.limit.upper if joint.limit and joint.limit.upper is not None else 1.0

    for alpha_out in qsteps:
        alpha_in = 1.0 - float(alpha_out)
        q = lower + alpha_in * (upper - lower)

        joint_states = {JOINT_NAME: q}
        fk_transforms = robot.link_fk(cfg=joint_states)

        all_meshes = []
        for link in robot.links:
            if not link.visuals:
                continue
            link_T = fk_transforms[link]
            for visual in link.visuals:
                if visual.geometry.mesh is None:
                    continue
                mesh_rel = visual.geometry.mesh.filename
                mesh_path = os.path.join(mesh_root, obj_name, mesh_rel)
                if not os.path.exists(mesh_path):
                    alt_path = os.path.join(os.path.dirname(urdf_path), mesh_rel)
                    if os.path.exists(alt_path):
                        mesh_path = alt_path
                    else:
                        print(f"[WARN] Mesh not found: {mesh_path}")
                        continue
                mesh = trimesh.load(mesh_path, force="mesh")
                visual_T = visual.origin if visual.origin is not None else np.eye(4)
                full_T = link_T @ visual_T
                mesh.apply_transform(full_T)
                all_meshes.append(mesh)

        if not all_meshes:
            print(f"[WARN] No meshes for alpha_out={alpha_out:.2f} (alpha_in={alpha_in:.2f}) in {urdf_parent}")
            continue

        merged = trimesh.util.concatenate(all_meshes)
        out_path = os.path.join(out_dir, f"pose_{alpha_in:.02f}.ply")
        merged.export(out_path)
        print(f"[OK] Exported: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sap", "tsdf"], required=True)
    parser.add_argument("--load_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--qpos_list", default="0.0,0.25,0.5,0.75,1.0")
    args = parser.parse_args()

    qsteps = parse_qpos_list(args.qpos_list)
    urdf_paths = find_urdfs(args.load_dir, args.mode)
    if not urdf_paths:
        print(f"[ERROR] No URDFs found for mode='{args.mode}' under {args.load_dir}")
        return

    print(f"[INFO] Found {len(urdf_paths)} URDF(s). Save-alphas: {qsteps}")
    for urdf_file in tqdm(urdf_paths):
        try:
            export_joint_poses_as_objs(
                urdf_path=urdf_file,
                mesh_root=args.load_dir,
                output_root=args.output_dir,
                qsteps=qsteps,
            )
        except Exception as e:
            print(f"[ERROR] {urdf_file}: {e}")


if __name__ == "__main__":
    main()
