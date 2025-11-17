import os
import json
import random
import argparse
import subprocess
import numpy as np
from glob import glob
from tqdm import tqdm
from urdfpy import URDF
from multiprocessing.pool import ThreadPool
for alias in ("float", "int", "bool"):
    if not hasattr(np, alias):
        setattr(np, alias, eval(alias))

import calibur

# Rendering configuration
NUM_VIEWS = 1
NUM_INPUTS = 32
NUM_OUTPUTS = 0
RESOLUTION = 512
NUM_JOINT_POSES = 5
NUM_TEX = 1
NUM_SCALE = 1

def prepare_assets():
    if not os.path.exists('./haven'):
        os.system("wget --no-verbose 1pb.skis.ltd/rendering/hdri.zip/hdri.zip")
        os.system("unzip hdri.zip")
    if not os.path.exists('./metadata'):
        os.system("wget --no-verbose 1pb.skis.ltd/rendering/meta.zip/meta.zip")
        os.system("unzip meta.zip")

def read(x: str):
    print(f"Processing {x}")
    meta_path = os.path.join(x, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    urdf_path = os.path.join(x, "mobility.urdf")
    urdf = URDF.load(urdf_path)

    joints = [j for j in urdf.joints if j.joint_type in ["prismatic", "revolute"]]
    if not joints:
        return -1

    all_texture_paths = glob("./texture/*.jpg")
    current_texture_paths = glob(os.path.join(x, "images/*.jpg"))
    random.shuffle(all_texture_paths)

    text_pose_list = []
    for t in range(NUM_TEX):
        tex_map = {}
        if t == 0:
            tex_map = {tp: tp for tp in current_texture_paths}
        else:
            tex_map = {tp: all_texture_paths[i] for i, tp in enumerate(current_texture_paths)}

        fovy = np.radians(np.clip(np.random.normal(36, 9), 5, 60))
        fx = fy = calibur.fov_to_focal(fovy, RESOLUTION)
        cx, cy = np.random.normal(RESOLUTION / 2, RESOLUTION / 8, 2)
        intrinsics = calibur.intrinsic_cv(cx, cy, fx, fy)

        scaled_pose_list = {
            "intrinsics": intrinsics.tolist(),
            "fovy": fovy,
            "texture": tex_map
        }

        for joint in joints:
            joint_data = {
                "move_parts": [],
                "origin": {},
                "axis": joint.axis.tolist()
            }

            for v in urdf.link_map[joint.child].visuals:
                name = os.path.basename(v.geometry.mesh.filename).split(".obj")[0]
                joint_data["origin"][name] = v.origin[:3, 3].tolist()
                joint_data["move_parts"].append(v.geometry.mesh.filename)

            lower = joint.limit.lower if joint.limit else 0
            upper = joint.limit.upper if joint.limit else 2 * np.pi

            scales = ["1.0 1.0 1.0"] if t <= 1 else [
                " ".join(str(n.item()) for n in np.round(np.random.uniform(0.5, 2.0, 3), 2))
                for _ in range(NUM_SCALE)
            ]

            for s in scales:
                pose_list = {"pose_list": [1.0, 0.25, 0.5, 0.75, 0.0], "transform": {}, "links": []}
                scale_values = list(map(float, s.split()))

                for i in pose_list["pose_list"]:
                    pose = lower + (upper - lower) * i
                    fk = urdf.visual_geometry_fk(cfg={joint.name: pose})
                    fk_link = urdf.link_fk(cfg={joint.name: pose})

                    for l in fk_link:
                        pose_list["links"].append(l.name)

                    for k, T in fk.items():
                        name = os.path.basename(k.mesh.filename).split(".")[0]
                        pose_list["transform"].setdefault(name, []).append(T.tolist())

                joint_data[s] = pose_list

            scaled_pose_list[joint.name] = joint_data

        text_pose_list.append(scaled_pose_list)

    with open(urdf_path.replace("mobility.urdf", "joint_pose_norm_debug.json"), "w") as f:
        json.dump(text_pose_list, f)

    print(f"joint_pose {urdf_path} recorded.")
    return 1

def run(command):
    subprocess.run(command.split(" "))

def export(x: str, out_dir: str):
    if read(x) == -1:
        return

    with open(os.path.join(x, "joint_pose_norm_debug.json"), "r") as f:
        scaled_pose_list = json.load(f)
    with open("./hdri.txt", "r") as f:
        hdri_list = [line.strip() for line in f]
    with open("./texture.txt", "r") as f:
        texture_list = [line.strip() for line in f]

    object_path = os.path.join(x, "textured_objs")
    for tex, tex_pose in enumerate(scaled_pose_list):
        for joint, joint_data in tex_pose.items():
            if joint in ["texture", "intrinsics", "fovy"]:
                continue

            for scale, pose in joint_data.items():
                if scale in ["move_parts", "origin", "axis"]:
                    continue

                k = min(200, len(texture_list))
                color_inds = "_".join(map(str, random.sample(range(len(texture_list)), k))) if k > 0 else ""

                hdri_path = random.choice(hdri_list)
                hdri_strength = random.uniform(1.5, 2)
                hdri_rot = random.uniform(-np.pi, np.pi)

                base_cmd = (
                    f"blenderproc run export_shape.py "
                    f"--custom-blender-path=/opt/blender-3.3.1-linux-x64 "
                    f"--object_path {object_path} "
                    f"--hdri_path {hdri_path} --hdri_strength {hdri_strength} "
                    f"--hdri_rotation_euler {hdri_rot} "
                    f"--output_dir {out_dir} "
                    f"--num_views {NUM_VIEWS} --num_inputs {NUM_INPUTS} "
                    f"--num_outputs {NUM_OUTPUTS} --resolution {RESOLUTION} "
                    f"--obj_scale {scale} --joint_name {joint} --tex_no {tex} "
                    f"--out_normal --out_depth --color_ind {color_inds}"
                )

                os.system(base_cmd + " --joint_pose 0")

                commands = [
                    base_cmd + f" --joint_pose {i}"
                    for i in range(1, NUM_JOINT_POSES)
                ]

                with ThreadPool(NUM_JOINT_POSES * NUM_SCALE * len(joint_data.keys())) as pool:
                    pool.map(run, commands)

    print(f"export {x} complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/path/to/partnet-mobility-v0")
    parser.add_argument("--datalist_path", type=str, default="datalist_path.txt")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    prepare_assets()

    with open(args.datalist_path, "r") as f:
        file_names = [os.path.basename(line).split("_")[0] for line in f]

    for path in tqdm(glob(os.path.join(args.data_root, "*"))):
        if os.path.basename(path) in file_names:
            export(path, args.output_dir)
