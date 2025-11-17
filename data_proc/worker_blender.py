#!/usr/bin/env python3
import os
import json
import random
import argparse
import subprocess
import numpy as np
from glob import glob
from urdfpy import URDF
from multiprocessing.pool import ThreadPool
import calibur
import numpy as np
for alias in ("float", "int", "bool"):
    if not hasattr(np, alias):
        setattr(np, alias, eval(alias))

# Rendering configuration (can still be edited here or exposed via args if you want)
NUM_VIEWS = 1
NUM_INPUTS = 32
NUM_OUTPUTS = 0
RESOLUTION = 512
NUM_JOINT_POSES = 5        # how many poses we will render (first N from the pose list)
NUM_TEX = 1
NUM_SCALE = 1
OBJECT_CONFIGS = {}

# Output json name used by both writer and reader
POSE_JSON_NAME = "joint_pose.json"

# Globals set from argparse
POSE_LIST_OVERRIDE = None   # type: list[float] | None
POSE_COUNT = 25             # length of pose list when generating randomly


def prepare_assets():
    if not os.path.exists('./haven'):
        os.system("wget --no-verbose 1pb.skis.ltd/rendering/hdri.zip/hdri.zip")
        os.system("unzip -o hdri.zip")
    if not os.path.exists('./metadata'):
        os.system("wget --no-verbose 1pb.skis.ltd/rendering/meta.zip/meta.zip")
        os.system("unzip -o meta.zip")
    paths = glob("./haven/hdris/*/*.hdr")
    with open("./hdri.txt", "w") as f:
        for p in paths:
            f.write(p + "\n")


def build_pose_sequence() -> list[float]:
    """
    Build the joint pose list:
      - If POSE_LIST_OVERRIDE (from --pose-list) is provided, sort it descending so the largest is first.
      - Otherwise generate random samples in [0,1], always start with 1.0 and end with 0.0.
    """
    if POSE_LIST_OVERRIDE:
        seq = sorted(POSE_LIST_OVERRIDE, reverse=True)
        return seq

    n = max(2, POSE_COUNT)  # ensure at least [1.0, 0.0]
    mid_n = n - 2
    if mid_n <= 0:
        return [1.0, 0.0][:n]

    middle = np.random.rand(mid_n).tolist()
    middle.sort(reverse=True)
    return [1.0] + middle + [0.0]


def read(x: str):
    print(f"Processing {x}")
    meta_path = os.path.join(x, "meta.json")
    if not os.path.isfile(meta_path):
        print(f"{x}: missing meta.json; skipped.")
        return -1

    with open(meta_path, "r") as f:
        meta = json.load(f)
    if meta.get("model_cat") not in [
        "StorageFurniture", "Microwave", "Oven", "Refrigerator",
        "Safe", "TrashCan", "Table"
    ]:
        print(f"{x}: category filtered.")
        return -1

    urdf_path = os.path.join(x, "mobility.urdf")
    if not os.path.isfile(urdf_path):
        print(f"{x}: missing mobility.urdf; skipped.")
        return -1

    urdf = URDF.load(urdf_path)
    joints = [j for j in urdf.joints if j.joint_type in ["prismatic", "revolute"]]
    if not joints:
        print(f"{x}: no prismatic/revolute joints; skipped.")
        return -1

    text_pose_list = []
    all_texture_paths = glob("./texture/*.jpg")
    current_texture_paths = glob(os.path.join(x, "images/*.jpg"))
    random.shuffle(all_texture_paths)

    for t in range(NUM_TEX):
        # Map each current texture to itself (t==0) or to a shuffled external texture
        if t == 0 or len(all_texture_paths) < len(current_texture_paths):
            tex_map = {p: p for p in current_texture_paths}
        else:
            tex_map = {p: q for p, q in zip(current_texture_paths, all_texture_paths)}

        intrinsics, fovy = generate_intrinsics()
        scaled_pose_list = {
            "intrinsics": intrinsics.tolist(),
            "fovy": fovy,
            "texture": tex_map
        }

        for joint in joints:
            # SAFE axis extraction (avoid boolean on numpy arrays)
            axis_attr = getattr(joint, "axis", None)
            axis = [0.0, 0.0, 1.0]
            if axis_attr is not None:
                arr = np.asarray(axis_attr, dtype=float).reshape(-1)
                if arr.size >= 3:
                    axis = arr[:3].tolist()

            joint_data = {
                "move_parts": [],
                "origin": {},
                "axis": axis
            }

            # Collect visuals for the child link of this joint
            child_link = urdf.link_map.get(joint.child)
            if child_link is None:
                continue
            for v in getattr(child_link, "visuals", []):
                try:
                    mesh_path = v.geometry.mesh.filename
                    name = os.path.basename(mesh_path).split(".obj")[0]
                    joint_data["origin"][name] = v.origin[:3, 3].tolist()
                    joint_data["move_parts"].append(mesh_path)
                except Exception:
                    # Skip malformed visual entries
                    continue

            # Joint limits
            lower = joint.limit.lower if joint.limit and joint.limit.lower is not None else 0.0
            upper = joint.limit.upper if joint.limit and joint.limit.upper is not None else (2 * np.pi)

            # Scales
            scales = ["1.0 1.0 1.0"] if t == 0 else generate_random_scales()

            # Pose sequence (either override list, sorted desc; or random with [1.0,...,0.0])
            pose_seq = build_pose_sequence()

            for s in scales:
                pose_list = {
                    "pose_list": pose_seq,
                    "transform": {},
                    "links": []
                }

                # For each pose value, compute FK for visuals and links
                for pose in pose_seq:
                    q = lower + (upper - lower) * float(pose)
                    fk_visuals = urdf.visual_geometry_fk(cfg={joint.name: q})
                    fk_links = urdf.link_fk(cfg={joint.name: q})

                    # Record link names (duplicates across poses are fine)
                    for lnk in fk_links:
                        pose_list["links"].append(lnk.name)

                    for k, T in fk_visuals.items():
                        key_name = os.path.basename(k.mesh.filename).split(".")[0]
                        # Store [R, t] with lists (as original code)
                        pose_list["transform"].setdefault(key_name, []).append(T.tolist())
                joint_data[s] = pose_list

            scaled_pose_list[joint.name] = joint_data

        text_pose_list.append(scaled_pose_list)

    out_json = os.path.join(x, POSE_JSON_NAME)
    with open(out_json, "w") as f:
        json.dump(text_pose_list, f)
    print(f"Saved joint poses to {out_json}")
    return 1


def generate_intrinsics():
    fovy = np.radians(np.clip(np.random.normal(36, 9), 5, 60))
    fx = fy = calibur.fov_to_focal(fovy, RESOLUTION)
    cx, cy = np.random.normal(RESOLUTION / 2, RESOLUTION / 8, 2)
    return calibur.intrinsic_cv(cx, cy, fx, fy), fovy


def generate_random_scales():
    return [
        " ".join(str(n.item()) for n in np.round(np.random.uniform(0.5, 2.0, 3), 2))
        for _ in range(NUM_SCALE)
    ]


def run(command):
    subprocess.run(command.split(" "), check=True)


def render(x: str, output_dir: str, args):
    if read(x) == -1:
        return

    # Read the same json we just wrote
    pose_json_path = os.path.join(x, POSE_JSON_NAME)
    with open(pose_json_path, "r") as f:
        scaled_pose_list = json.load(f)

    with open("./hdri.txt", "r") as f:
        hdri_list = [line.strip() for line in f if line.strip()]
    with open("./texture.txt", "r") as f:
        texture_list = [line.strip() for line in f if line.strip()]

    object_path = os.path.join(x, "textured_objs")

    # -------------------------------------------------------
    # Detect parent object folder for consistent intrinsics/HDRI
    # -------------------------------------------------------
    parent_obj = os.path.basename(os.path.dirname(x))
    if parent_obj not in OBJECT_CONFIGS:
        intrinsics, fovy = generate_intrinsics()
        hdri_path = random.choice(hdri_list) if hdri_list else ""
        hdri_strength = random.uniform(1.5, 2)
        hdri_rot = random.uniform(-np.pi, np.pi)
        OBJECT_CONFIGS[parent_obj] = {
            "intrinsics": intrinsics,
            "fovy": fovy,
            "hdri_path": hdri_path,
            "hdri_strength": hdri_strength,
            "hdri_rotation": hdri_rot
        }
        print(f"[INFO] Created new lighting config for {parent_obj}")
    else:
        print(f"[INFO] Reusing lighting/intrinsics for {parent_obj}")

    config = OBJECT_CONFIGS[parent_obj]
    hdri_path = config["hdri_path"]
    hdri_strength = config["hdri_strength"]
    hdri_rot = config["hdri_rotation"]

    # -------------------------------------------------------
    # Rendering loop
    # -------------------------------------------------------
    for tex, tex_data in enumerate(scaled_pose_list):
        for joint, joint_data in tex_data.items():
            if joint in ["texture", "intrinsics", "fovy"]:
                continue

            for scale, poses in joint_data.items():
                if scale in ["move_parts", "origin", "axis"]:
                    continue

                # Choose texture indices safely
                k = min(200, len(texture_list))
                color_inds = "_".join(map(str, random.sample(range(len(texture_list)), k))) if k > 0 else ""

                base_cmd = (
                    f"blenderproc run render_shape.py "
                    f"--custom-blender-path=/opt/blender-3.3.1-linux-x64 "
                    f"--object_path {object_path} "
                    f"--hdri_path {hdri_path} --hdri_strength {hdri_strength} "
                    f"--hdri_rotation_euler {hdri_rot} "
                    f"--output_dir {output_dir} "
                    f"--num_views {NUM_VIEWS} --num_inputs {NUM_INPUTS} "
                    f"--num_outputs {NUM_OUTPUTS} --resolution {RESOLUTION} "
                    f"--obj_scale {scale} --joint_name {joint} --tex_no {tex} "
                    f"--out_normal --out_depth --color_ind {color_inds}"
                )

                available = len(poses.get("pose_list", []))
                to_render = min(NUM_JOINT_POSES, max(1, available))

                run(base_cmd + " --joint_pose 0")

                commands = [base_cmd + f" --joint_pose {i}" for i in range(1, to_render)]
                if commands:
                    pool_size = min(len(commands), max(1, (os.cpu_count() or 4)))
                    with ThreadPool(pool_size) as pool:
                        pool.map(run, commands)

    print(f"Render complete for {x}")



def parse_pose_list_arg(s: str) -> list[float]:
    # Accept comma or whitespace separated floats
    if not s:
        return []
    parts = []
    for tok in s.replace(",", " ").split():
        try:
            parts.append(float(tok))
        except ValueError:
            pass
    return parts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of partnet-mobility-v0")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for renderings")
    parser.add_argument("--pose-count", type=int, default=25, help="Number of poses when generating randomly")
    parser.add_argument("--pose-list", type=str, default=None,
                        help="Explicit pose list (comma/space-separated). Will be sorted descending (largest first).")
    args = parser.parse_args()

    # Configure globals from args
    # global POSE_COUNT, POSE_LIST_OVERRIDE
    POSE_COUNT = args.pose_count
    POSE_LIST_OVERRIDE = parse_pose_list_arg(args.pose_list) if args.pose_list else None

    prepare_assets()

    for path in glob(os.path.join(args.data_root, "*")):
        if "44817" in path or "45213" in path:
            render(path, args.output_dir, args)
