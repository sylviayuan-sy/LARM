import os
import json
import random
from glob import glob
from itertools import combinations

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset
import cv2
import imageio
import open3d as o3d
from open3d import geometry
from open3d.utility import Vector3dVector
import trimesh

import calibur
from loftr import LoftrRunner


def ransac(px, py, threshold=0.001, num_samples=4, num_iters=1000, eps=1e-10):
    px = np.concatenate([px, np.ones((px.shape[0], 1))], axis=-1).T
    py = np.concatenate([py, np.ones((py.shape[0], 1))], axis=-1).T
    rand_idx = random.sample(np.arange(0, px.shape[1]).tolist(), num_samples)
    transform_T, _, _, _ = np.linalg.lstsq(px[:, rand_idx].T, py[:, rand_idx].T, rcond=None)
    transform = transform_T.T

    largest_set_x, largest_set_y = np.zeros((4, 0)), np.zeros((4, 0))

    for _ in range(num_iters):
        diff = np.sqrt((transform @ px - py) ** 2 + eps).T.sum(axis=-1)
        if px[:, diff < threshold].shape[1] > largest_set_x.shape[1]:
            largest_set_x = px[:, diff < threshold].copy()
            largest_set_y = py[:, diff < threshold].copy()
        if largest_set_x.shape[1] >= 0.95 * px.shape[1]:
            break
        rand_idx = random.sample(np.arange(0, largest_set_x.shape[1] if largest_set_x.shape[1] > 4 else px.shape[1]).tolist(), num_samples)
        transform_T, _, _, _ = np.linalg.lstsq(
            (largest_set_x if largest_set_x.shape[1] > 4 else px)[:, rand_idx].T,
            (largest_set_y if largest_set_y.shape[1] > 4 else py)[:, rand_idx].T,
            rcond=None
        )
        transform = transform_T.T

    return largest_set_x.T[:, :3], largest_set_y.T[:, :3]


def rot_mat_from_axis_theta(axis, theta):
    R = torch.eye(3, device=theta.device).unsqueeze(0).repeat(theta.shape[0], 1, 1)
    axis = axis.unsqueeze(0).repeat(theta.shape[0], 1)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

    R[:, 0, 0] = cos_theta + (axis[:, 0] ** 2) * (1 - cos_theta)
    R[:, 0, 1] = axis[:, 0] * axis[:, 1] * (1 - cos_theta) - axis[:, 2] * sin_theta
    R[:, 0, 2] = axis[:, 0] * axis[:, 2] * (1 - cos_theta) + axis[:, 1] * sin_theta
    R[:, 1, 0] = axis[:, 0] * axis[:, 1] * (1 - cos_theta) + axis[:, 2] * sin_theta
    R[:, 1, 1] = cos_theta + (axis[:, 1] ** 2) * (1 - cos_theta)
    R[:, 1, 2] = axis[:, 1] * axis[:, 2] * (1 - cos_theta) - axis[:, 0] * sin_theta
    R[:, 2, 0] = axis[:, 0] * axis[:, 2] * (1 - cos_theta) - axis[:, 1] * sin_theta
    R[:, 2, 1] = axis[:, 1] * axis[:, 2] * (1 - cos_theta) + axis[:, 0] * sin_theta
    R[:, 2, 2] = cos_theta + (axis[:, 2] ** 2) * (1 - cos_theta)
    return R


def draw_corr(rgbA, rgbB, corrA, corrB, output_name):
    vis = np.concatenate([rgbA, rgbB], axis=1)
    for i in range(len(corrA)):
        uvA = corrA[i]
        uvB = corrB[i].copy()
        uvB[0] += rgbA.shape[1]
        color = tuple(np.random.randint(0, 255, size=(3)).tolist())
        vis = cv2.circle(vis, uvA, radius=2, color=color, thickness=1)
        vis = cv2.circle(vis, uvB, radius=2, color=color, thickness=1)
        vis = cv2.line(vis, uvA, uvB, color=color, thickness=1, lineType=cv2.LINE_AA)
    imageio.imwrite(f'{output_name}.png', vis.astype(np.uint8))


class Dataset(TorchDataset):
    def __init__(self, data_path):
        super().__init__()
        self.got_point_pairs = None

        with open(os.path.join(data_path, "transforms.json"), "r") as f:
            meta = json.load(f)

        self.joint_type = meta["joint_type"]
        self.fx, self.fy, self.cx, self.cy = meta["fx"], meta["fy"], meta["cx"], meta["cy"]

        def _abs_png(stem_path: str) -> str:
            p = stem_path
            if p.startswith("./"):
                p = p[2:]
            if not p.endswith(".png"):
                p = p + ".png"
            return os.path.normpath(os.path.join(data_path, p))

        def _idx_from_pathlike(stem_path: str) -> int:
            base = os.path.basename(stem_path)
            stem = base.split("/")[-1]
            if stem.endswith(".png"):
                stem = stem[:-4]
            try:
                return int(stem.split("_")[0])
            except Exception:
                return 1 << 30

        frames = meta["frames"]
        q_to_frames = {}
        for fr in frames:
            q = float(fr.get("qpos", 0.0))
            q_to_frames.setdefault(q, []).append(fr)

        if "qpos_list" in meta and len(meta["qpos_list"]) > 0:
            qpos_order = [float(q) for q in meta["qpos_list"] if float(q) in q_to_frames]
        else:
            qpos_order = sorted(q_to_frames.keys())

        # Sort frames for each q by residual view index parsed from file_path
        for q in q_to_frames:
            q_to_frames[q].sort(key=lambda fr: _idx_from_pathlike(fr["file_path"]))

        # Views per qpos = min length across groups (guard against ragged sets)
        self.num_per_qpos = min(len(q_to_frames[q]) for q in qpos_order)
        self.qpos_list = qpos_order

        # Build concrete image paths per qpos from file_path stems in meta
        self.qpos_dict = {}
        for q in qpos_order:
            paths = []
            for fr in q_to_frames[q][:self.num_per_qpos]:
                paths.append(_abs_png(fr["file_path"]))
            self.qpos_dict[f"{q:.02f}"] = paths

        # Per-view extrinsics come from the first qpos group (aligned by residual i)
        first_q = qpos_order[0]
        per_view_extrins = [np.array(fr["transform_matrix"]) for fr in q_to_frames[first_q][:self.num_per_qpos]]

        # Build all image pairs (paired by residual i) for all distinct qpos pairs
        from itertools import combinations
        qpos_combinations = [pair for pair in combinations(self.qpos_list, 2) if abs(pair[0] - pair[1]) > 0.0]

        self.all_image_pairs, self.all_qpos_pairs, self.all_extrin = [], [], []
        for q0, q1 in qpos_combinations:
            q0s, q1s = f"{q0:.02f}", f"{q1:.02f}"
            pairs = [(self.qpos_dict[q0s][i], self.qpos_dict[q1s][i]) for i in range(self.num_per_qpos)]
            limit = min(128, len(pairs))
            self.all_image_pairs += pairs[:limit]
            self.all_qpos_pairs += [(q0, q1)] * limit
            self.all_extrin += per_view_extrins[:limit]

        self.loftr = LoftrRunner()
        self.all_point_pairs = []

        pp_json = os.path.join(data_path, "point_pairs.json")
        if os.path.exists(pp_json):
            with open(pp_json, "r") as f:
                self.all_point_pairs = json.load(f)
            for pair in self.all_point_pairs:
                pair["px"] = np.array(pair["px"])
                pair["py"] = np.array(pair["py"])
        else:
            for idx in range(len(self.all_image_pairs)):
                self.get_pairs(idx)
            to_dump = [{"px": p["px"].tolist(), "py": p["py"].tolist(),
                        "qpos_x": p["qpos_x"], "qpos_y": p["qpos_y"]} for p in self.all_point_pairs]
            with open(pp_json, "w") as f:
                json.dump(to_dump, f)
            for pair in self.all_point_pairs:
                pair["px"] = np.array(pair["px"])
                pair["py"] = np.array(pair["py"])

        self.got_point_pairs = self.all_point_pairs
        print(f"{len(self.all_point_pairs)} point pairs found.")

    def __len__(self):
        return len(self.got_point_pairs) if self.got_point_pairs else 0

    def __getitem__(self, idx):
        return self.got_point_pairs[idx]

    def to_xyz(self, depth, u, v, c2w):
        blender2opencv = np.array([[1, 0, 0, 0],
                                   [0, -1, 0, 0],
                                   [0, 0, -1, 0],
                                   [0, 0, 0, 1]])
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        xyz = trimesh.transform_points(np.stack((x, y, depth), axis=-1), c2w @ blender2opencv)
        return xyz

    def get_pairs(self, idx):
        im_path_0, im_path_1 = self.all_image_pairs[idx]
        extrin = self.all_extrin[idx]
        qpos_0, qpos_1 = self.all_qpos_pairs[idx]

        # Load images, masks, and depth (same folder, suffixed names)
        image_0 = np.array(Image.open(im_path_0).convert("RGB"))
        mask_0 = (np.array(Image.open(im_path_0.replace(".png", "_partmask.png")).convert("L")) / 255.).astype(bool)
        depth_0 = np.load(im_path_0.replace(".png", "_depth.npy")) * 5.0

        image_1 = np.array(Image.open(im_path_1).convert("RGB"))
        mask_1 = (np.array(Image.open(im_path_1.replace(".png", "_partmask.png")).convert("L")) / 255.).astype(bool)
        depth_1 = np.load(im_path_1.replace(".png", "_depth.npy")) * 5.0

        h, w = image_0.shape[:2]

        # Correspondences from LoFTR
        corres = np.array(self.loftr.predict(
            np.expand_dims(image_0, axis=0),
            np.expand_dims(image_1, axis=0)
        ))[0]
        corres = corres[corres[:, -1] > 0.7]

        def get_valid_mask(mask, coords):
            valid = np.logical_and(np.logical_and(coords[..., 0] >= 0, coords[..., 0] < w),
                                   np.logical_and(coords[..., 1] >= 0, coords[..., 1] < h))
            valid = np.logical_and(valid, mask[coords[..., 1], coords[..., 0]])
            return valid

        src_coords = corres[:, :2].round().astype(int)
        tgt_coords = corres[:, 2:4].round().astype(int)

        part_mask = np.logical_and(mask_0, mask_1)
        valid_mask = np.logical_and(
            get_valid_mask(part_mask, src_coords),
            get_valid_mask(part_mask, tgt_coords)
        )

        src_coords = src_coords[np.where(valid_mask)]
        tgt_coords = tgt_coords[np.where(valid_mask)]

        # Filter correspondences by distance
        dist = np.sqrt((src_coords - tgt_coords) ** 2 + 1e-10).sum(axis=-1)
        src_coords = src_coords[dist > 16.]
        tgt_coords = tgt_coords[dist > 16.]

        # Get depth for valid correspondences
        depth_0_vals = depth_0[src_coords[:, 1], src_coords[:, 0]]
        depth_1_vals = depth_1[tgt_coords[:, 1], tgt_coords[:, 0]]

        # Project to 3D
        px = self.to_xyz(depth_0_vals, src_coords[:, 0], src_coords[:, 1], extrin)
        py = self.to_xyz(depth_1_vals, tgt_coords[:, 0], tgt_coords[:, 1], extrin)

        # Remove duplicate 3D points
        if px.shape[0] > 1:
            _, indices = np.unique(px, return_index=True, axis=0)
            px = px[indices]
            py = py[indices]

        if py.shape[0] > 1:
            _, indices = np.unique(py, return_index=True, axis=0)
            px = px[indices]
            py = py[indices]

        # Store results
        if px.shape[0] > 4:
            for i in range(px.shape[0]):
                self.all_point_pairs.append({
                    "px": px[i],
                    "py": py[i],
                    "qpos_x": float(qpos_0),
                    "qpos_y": float(qpos_1)
                })
