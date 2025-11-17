#!/usr/bin/env python3
import os
import json
import torch
import trimesh
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

from data_ransac import Dataset
from loss import Loss
from model import JE
from urdfpy import URDF, Link, Joint, JointLimit, Visual, Geometry, Mesh, Collision, Inertial
import calibur
import random
import copy

config = {"lr": 1e-2, "batch_size": 1024, "num_iters": 500}
orient = torch.Tensor(calibur.convert_pose(np.eye(4), ("X", "Y", "Z"), ("X", "Y", "Z"))[:3, :3]).cuda().reshape(1, 3, 3)

def rot_mat_from_axis_theta(axis, theta):
    R = np.eye(3)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    R[0, 0] = cos_theta + axis[0] ** 2 * (1 - cos_theta)
    R[0, 1] = axis[0] * axis[1] * (1 - cos_theta) - axis[2] * sin_theta
    R[0, 2] = axis[0] * axis[2] * (1 - cos_theta) + axis[1] * sin_theta
    R[1, 0] = axis[1] * axis[0] * (1 - cos_theta) + axis[2] * sin_theta
    R[1, 1] = cos_theta + axis[1] ** 2 * (1 - cos_theta)
    R[1, 2] = axis[1] * axis[2] * (1 - cos_theta) - axis[0] * sin_theta
    R[2, 0] = axis[2] * axis[0] * (1 - cos_theta) - axis[1] * sin_theta
    R[2, 1] = axis[2] * axis[1] * (1 - cos_theta) + axis[0] * sin_theta
    R[2, 2] = cos_theta + axis[2] ** 2 * (1 - cos_theta)
    return R

def _find_existing_mesh_in(path, name, kind):
    # also recognize already-created proxy meshes
    cand = [
        os.path.join(path, f"{name}_{kind}.ply"),
        os.path.join(path, f"{name}_{kind}.obj"),
        os.path.join(path, f"empty_{kind}.obj"),
        os.path.join(path, f"empty_{kind}.ply"),
    ]
    pats = [
        f"*mesh_*_pred_{kind}_smooth.obj",
        f"*pred_{kind}_smooth.obj",
        f"*_{kind}_smooth.obj",
        f"*mesh_*_{kind}.obj",
        f"*_{kind}.obj",
        f"*mesh_*_pred_{kind}_smooth.ply",
        f"*pred_{kind}_smooth.ply",
        f"*_{kind}_smooth.ply",
        f"*mesh_*_{kind}.ply",
        f"*_{kind}.ply",
    ]
    for p in cand:
        if os.path.isfile(p):
            return p
    for pat in pats:
        matches = glob(os.path.join(path, pat))
        if matches:
            return sorted(matches, key=lambda p: (len(os.path.basename(p)), p))[0]
    return None

def _ensure_proxy_mesh(path, kind, size=1e-2):
    """
    Create a tiny cube proxy mesh named empty_{kind}.obj in `path` if missing.
    Returns the relative filename (basename).
    """
    os.makedirs(path, exist_ok=True)
    fname = f"empty_{kind}.obj"
    dst = os.path.join(path, fname)
    if not os.path.isfile(dst):
        # tiny centered box; extents in meters-ish; adjust as needed
        mesh = trimesh.creation.box(extents=[size, size, size])
        mesh.export(dst)
        print(f"[WARN] Created proxy mesh: {dst}")
    return fname

def _localize_mesh_or_fail(path, name, kind):
    src = _find_existing_mesh_in(path, name, kind)
    if src is None:
        # instead of failing, create a proxy mesh (tiny cube) as requested
        return _ensure_proxy_mesh(path, kind)

    if os.path.dirname(os.path.abspath(src)) != os.path.abspath(path):
        dst = os.path.join(path, os.path.basename(src))
        if not os.path.isfile(dst):
            os.makedirs(path, exist_ok=True)
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                fo.write(fi.read())
        return os.path.basename(dst)
    return os.path.basename(src)

def train(path, threshold=0.01, percent_samples=0.1, num_iters=10, eps=1e-10, percent_terminate=0.8, rot_aug=False):
    with open(os.path.join(path, "transforms.json"), "r") as f:
        meta = json.load(f)
    name = os.path.basename(path)
    joint_type = meta["joint_type"]

    dataset = Dataset(path, rot_aug=rot_aug)
    best_model = None
    best_num_inliers = 0
    inliers = torch.ones((len(dataset.all_point_pairs),))

    for iter_idx in tqdm(range(num_iters)):
        model = JE(joint_type).cuda()
        random.shuffle(dataset.all_point_pairs)
        dataset.got_point_pairs = dataset.all_point_pairs[:int(percent_samples * len(dataset.all_point_pairs))]

        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        data_iter = iter(dataloader)
        criterion = Loss("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

        for _ in range(config["num_iters"]):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                data = next(data_iter)

            px = data["px"].float().cuda()
            py = data["py"].float().cuda()
            qpos_x = data["qpos_x"].float().cuda()
            qpos_y = data["qpos_y"].float().cuda()

            px[:, :3] = (orient @ px[:, :3].unsqueeze(-1)).squeeze(-1)
            py[:, :3] = (orient @ py[:, :3].unsqueeze(-1)).squeeze(-1)

            optimizer.zero_grad()
            Tpx = model(px, qpos_x, qpos_y)
            loss = criterion(Tpx, py[:, :3])
            loss.backward()
            optimizer.step()

        dataset.got_point_pairs = dataset.all_point_pairs
        dataloader = DataLoader(dataset, batch_size=len(dataset.all_point_pairs), shuffle=False)
        data = next(iter(dataloader))

        with torch.no_grad():
            Tpx = model(data["px"].float().cuda(), data["qpos_x"].float().cuda(), data["qpos_y"].float().cuda()).cpu()
            diff = torch.sqrt(((Tpx - data["py"].float()[:, :3]) ** 2).sum(dim=-1) + eps)
            inliers = diff < threshold

        num_inliers = inliers.sum()
        if num_inliers > best_num_inliers or best_model is None:
            best_num_inliers = num_inliers
            best_model = copy.deepcopy(model)

        if num_inliers / len(dataset.all_point_pairs) >= percent_terminate:
            break

    model = best_model
    axis = model.axis.cpu().detach().numpy()
    origin = np.eye(4)

    # robustly localize or create proxy meshes when missing
    base_rel = _localize_mesh_or_fail(path, name, "base")
    part_rel = _localize_mesh_or_fail(path, name, "part")
    base_abs = os.path.abspath(os.path.join(path, base_rel))
    part_abs = os.path.abspath(os.path.join(path, part_rel))

    if joint_type == "prismatic":
        # ---- PRISMATIC: enforce lower < upper with upper == 0.0; flip axis if needed ----
        m = float(axis[3])  # predicted travel magnitude (can be signed)
        axis_dir = axis[:3] / (np.linalg.norm(axis[:3]) + 1e-12)

        # Flip axis so motion is non-negative along +axis_dir
        if m < 0:
            axis_dir = -axis_dir
            m = -m

        # Limits: upper fixed to 0.0; lower strictly negative
        lower = -abs(m)
        upper = 0.0

        qpos1_origin = origin.copy()

        base_geom = Geometry(mesh=Mesh(filename=base_abs))
        part_geom = Geometry(mesh=Mesh(filename=part_abs))

        base_link = Link(
            name="base",
            inertial=Inertial(0.0, np.eye(3), origin),
            collisions=[Collision(name="base", origin=origin, geometry=base_geom)],
            visuals=[Visual(name="base", origin=origin, geometry=base_geom)]
        )
        part_link = Link(
            name="part",
            inertial=Inertial(0.0, np.eye(3), qpos1_origin),
            collisions=[Collision(name="part", origin=qpos1_origin, geometry=part_geom)],
            visuals=[Visual(name="part", origin=qpos1_origin, geometry=part_geom)]
        )

        joint = Joint(
            "joint",
            "prismatic",
            "base",
            "part",
            origin=origin,
            axis=axis_dir.tolist(),
            limit=JointLimit(0., 0., lower=lower, upper=upper)
        )

    else:
        # ---- REVOLUTE: enforce lower < upper with upper == 0.0; flip axis if needed ----
        ang = float(axis[6])  # predicted total angle (can be signed)
        axis_dir = axis[:3] / (np.linalg.norm(axis[:3]) + 1e-12)

        # Flip axis so angle is non-negative along +axis_dir
        if ang < 0:
            axis_dir = -axis_dir
            ang = -ang

        # Limits: upper fixed to 0.0; lower strictly negative
        lower = -abs(ang)
        upper = 0.0

        origin[:3, 3] = axis[3:6]  # pivot position
        qpos1_origin = np.linalg.inv(origin)

        base_geom = Geometry(mesh=Mesh(filename=base_abs))
        part_geom = Geometry(mesh=Mesh(filename=part_abs))

        base_link = Link(
            name="base",
            inertial=Inertial(0.0, np.eye(3), np.eye(4)),
            collisions=[Collision(name="base", origin=np.eye(4), geometry=base_geom)],
            visuals=[Visual(name="base", origin=np.eye(4), geometry=base_geom)]
        )
        part_link = Link(
            name="part",
            inertial=Inertial(0.0, np.eye(3), qpos1_origin),
            collisions=[Collision(name="part", origin=qpos1_origin, geometry=part_geom)],
            visuals=[Visual(name="part", origin=qpos1_origin, geometry=part_geom)]
        )

        joint = Joint(
            "joint",
            "revolute",
            "base",
            "part",
            origin=origin,
            axis=axis_dir.tolist(),
            limit=JointLimit(0., 0., lower=lower, upper=upper)
        )


    robot = URDF(name=name, links=[base_link, part_link], joints=[joint])
    urdf_path = os.path.join(path, f"{name}.urdf")
    robot.save(urdf_path)

    with open(urdf_path, "r") as f:
        s = f.read()
    s = s.replace(base_abs, base_rel).replace(part_abs, part_rel)
    with open(urdf_path, "w") as f:
        f.write(s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, default="../output_view",
                        help="Path to the base folder containing eval_ folders")
    parser.add_argument("--datalist_path", type=str, default="./data_sample/view_metadata/data.txt",
                        help="Path to the text file listing source items used to derive eval_* folder names")
    parser.add_argument("--rot-aug", action="store_true",
                        help="Use rotation augmentation mode: depth + 0.5 instead of depth * 5.0")
    args = parser.parse_args()

    rot_aug = getattr(args, 'rot_aug', False)
    with open(args.datalist_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        file_names = ["_".join(line.split("/")[-1].split("_")[:3]).split(".")[0] for line in lines]

    for name in tqdm(file_names):
        train(os.path.join(args.load_dir, f"eval_{name}"), rot_aug=rot_aug)
