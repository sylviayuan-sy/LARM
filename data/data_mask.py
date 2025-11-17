import random
import traceback
from glob import glob

import numpy as np
import pandas as pd
import PIL
from PIL import Image
import torch
from easydict import EasyDict as edict

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
from torch.utils.data import Dataset
import cv2
import json
import calibur
import trimesh
import OpenEXR

from sklearn.cluster import KMeans

def read_exr(path):
    exr = OpenEXR.InputFile(path)
    dw = exr.header()["dataWindow"]
    shape = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    
    data = {}
    for channel_name, channel in exr.header()["channels"].items():
        b = exr.channel(channel_name, channel.type)
        array = np.frombuffer(b, dtype=np.float32).reshape(shape)
        data[channel_name] = array
    # breakpoint()
    # return np.concatenate([np.expand_dims(d, axis=-1) for k, d in data.items()], axis=-1)
    return data["B"]

def kmeans_downsample(points, n_points_to_sample):
    kmeans = KMeans(n_points_to_sample).fit(points)
    return ((points - kmeans.cluster_centers_[..., None, :]) ** 2).sum(-1).argmin(-1).tolist()

def normalize_poses(
    ref_c2w,
    ref_fxfycxcy,
    ref_H,
    ref_W,
    src_c2ws,
    normalize_distance_to=0.0,
    keep_original_elevation=False,
):
    """
    ref_w2c: [4, 4]; ndarray
    ref_fxfycxcy: [4]; ndarray or list
    ref_H, ref_W: int
    src_w2cs: list of [4, 4]; ndarray
    return: [N, 4, 4]; ndarray; c2w transform
    """
    
    cam_to_origin = np.linalg.norm(ref_c2w[:3, 3])
    H, W = ref_H, ref_W
    fx, fy, cx, cy = ref_fxfycxcy

    trgt_ref_x_axis = np.array([1.0, 0.0, 0.0])
    trgt_ref_origin = np.array([0.0, -cam_to_origin, 0.0])
    top_middle_ray = [(W / 2 - cx) / fx, (0.0 - cy) / fy, 1.0]
    image_center_ray = [(W / 2 - cx) / fx, (H / 2 - cy) / fy, 1.0]

    target_image_center = np.array(
        [0.0, -cam_to_origin + np.linalg.norm(image_center_ray), 0.0]
    )
    cos_theta = np.dot(top_middle_ray, image_center_ray) / (
        np.linalg.norm(top_middle_ray) * np.linalg.norm(image_center_ray)
    )
    sin_theta = np.sqrt(1 - cos_theta**2)
    target_top_middle = np.array(
        [
            0.0,
            -cam_to_origin + np.linalg.norm(top_middle_ray) * cos_theta,
            np.linalg.norm(top_middle_ray) * sin_theta,
        ]
    )
    trgt_ref_y_axis = target_image_center - target_top_middle
    trgt_ref_y_axis = trgt_ref_y_axis / np.linalg.norm(trgt_ref_y_axis)
    trgt_ref_z_axis = np.cross(trgt_ref_x_axis, trgt_ref_y_axis)
    trgt_ref_z_axis = trgt_ref_z_axis / np.linalg.norm(trgt_ref_z_axis)

    trgt_ref_c2w = np.eye(4)
    trgt_ref_c2w[:3, 0] = trgt_ref_x_axis
    trgt_ref_c2w[:3, 1] = trgt_ref_y_axis
    trgt_ref_c2w[:3, 2] = trgt_ref_z_axis
    trgt_ref_c2w[:3, 3] = trgt_ref_origin

    if keep_original_elevation:
        # azzume z-axis is up
        camera_origin = ref_c2w[:3, 3]
        elevation = np.arctan2(camera_origin[2], np.linalg.norm(camera_origin[:2]))
        # rotate around x-axis by -elevation angles
        rot_x = np.eye(4)
        rot_x[1, 1] = np.cos(-elevation)
        rot_x[1, 2] = -np.sin(-elevation)
        rot_x[2, 1] = np.sin(-elevation)
        rot_x[2, 2] = np.cos(-elevation)
        trgt_ref_c2w = rot_x @ trgt_ref_c2w

    trgt_ref_w2c = np.linalg.inv(trgt_ref_c2w)

    pose_c2ws = [trgt_ref_c2w]
    for src_c2w in src_c2ws:
        rel_pose = np.linalg.inv(src_c2w) @ ref_c2w
        w2c_new = rel_pose @ trgt_ref_w2c
        c2w_new = np.linalg.inv(w2c_new)
        pose_c2ws.append(c2w_new)
    pose_c2ws = np.array(pose_c2ws)

    if normalize_distance_to > 0.0:
        scale = normalize_distance_to / cam_to_origin
        pose_c2ws[:, :3, 3] *= scale

    return pose_c2ws


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    return c2w


def recenter_mean_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def centerize_scale_poses(
    in_c2ws, mean_pose, scale_range=(1.0, 1.1), scene_scale_method="two_cam"
):
    if mean_pose:
        in_c2ws = recenter_mean_poses(in_c2ws)
    else:
        scene_center = (np.max(in_c2ws[:, :3, 3], 0) + np.min(in_c2ws[:, :3, 3], 0)) / 2
        in_c2ws[:, :3, 3] -= scene_center
    scene_scale = np.max(np.abs(in_c2ws[:, :3, 3]))

    if scene_scale_method == "two_cam":
        two_cam_dist = np.linalg.norm(in_c2ws[0, :3, 3] - in_c2ws[1, :3, 3])
        scene_scale = 1.0 / (two_cam_dist + 0.01)
    elif scene_scale_method == "fix_range":
        scene_scale = random.uniform(scale_range[0], scale_range[1]) * scene_scale
    else:
        raise NotImplementedError

    in_c2ws[:, :3, 3] *= scene_scale

    return in_c2ws


def square_crop(rgbs, input_fxfycxcy, size=256, center=True):
    h, w = rgbs.shape[-2:]
    out_h, out_w = size, size
    # if out_w > w or out_h > h:
    #     return rgbs, intrinsics

    if center:
        center_h, center_w = h // 2, w // 2
    else:
        center_w = (
            w // 2
            if out_w >= w
            else np.random.randint(low=out_w // 2 + 1, high=w - out_w // 2 - 1)
        )
        center_h = (
            h // 2
            if out_h >= h
            else np.random.randint(low=out_h // 2 + 1, high=h - out_h // 2 - 1)
        )

    rgbs_out = rgbs[
        :,
        :,
        center_h - out_h // 2 : center_h + out_h // 2,
        center_w - out_w // 2 : center_w + out_w // 2,
    ]
    input_fxfycxcy[:, 2] -= center_w - out_w // 2
    input_fxfycxcy[:, 3] -= center_h - out_h // 2

    return rgbs_out, input_fxfycxcy


def pil_to_np(pil_image):
    if pil_image.mode == "RGBA":
        # if directly convert to np.array, alpha=0 pixels will be black
        r, g, b, a = pil_image.split()
        r, g, b, a = np.asarray(r), np.asarray(g), np.asarray(b), np.asarray(a)
        image = np.stack([r, g, b, a], axis=2)
    else:
        image = np.asarray(pil_image)
    return image

default_config = edict(
    {
        "inference": False,
        "training": {
            "dataset_path": None,
            "num_views": 12,
            "use_rel_pose": True,
            "normalize_distance_to": 2.0,
            "normal_loss_weight": 0.0,
            "data_convert": True,
            "struct_views": False,
            "remove_alpha": True
        },
        "model": {
            "image_tokenizer": {
                "image_size": 256,
            },
        },
    }
)

def convert(meta, part=False):
    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    out = {"frames": []}
    
    frames = meta["sample_0"]
    fx = np.array(frames["intrinsics"])[0][0]
    fy = np.array(frames["intrinsics"])[1][1]
    cx = np.array(frames["intrinsics"])[0][2]
    cy = np.array(frames["intrinsics"])[1][2]
    if part:
        joint_pose = f'{meta["joint_pose"]:.2f}'
    for i in range(32):
        c2w = np.array(frames[f"input_frame_{i}"]["transform_matrix"]) @ blender2opencv
        w2c = np.linalg.inv(c2w)
        if part:
            out['frames'].append({
                "w": 1024,
                "h": 1024,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "w2c": w2c.tolist(),
                "file_path": f"color_{joint_pose}_in_{i:02d}.png",
            })
        else:
            out['frames'].append({
                "w": 1024,
                "h": 1024,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "w2c": w2c.tolist(),
                "file_path": f"color_00_in_{i:02d}.png",
            })
    return out

class Dataset(Dataset):
    def __init__(self, dataset_list, num_views, resolution, config=default_config, part_finetune=False):
        super().__init__()
        self.config = config
        self.config.training.dataset_path = dataset_list
        self.config.training.num_views = num_views
        self.config.model.image_tokenizer.image_size = resolution
        self.is_train = config.inference is False
        self.is_part = part_finetune

        self.all_camera_paths = open(
            self.config.training.dataset_path
        ).readlines()

        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.all_camera_paths = pd.array(
            [s.strip() for s in self.all_camera_paths if len(s) > 0], dtype="string"
        )

    def __len__(self):
        return len(self.all_camera_paths)
    
    def projection_matrix(self, fx, fy, cx, cy, h, w, n=0.1, f=1000.0, device=None):
        return [[           2*fx/w,    0,            -2*cx/w+1,              0], 
                [           0,      2*fy/h,            2*cy/h-1,              0], 
                [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                [           0,    0,           -1,              0]]

    def __getitem__(self, idx):
        try:
            camera_path = self.all_camera_paths[idx].strip()
            if self.config.training.get("data_convert", True):
                if self.is_part:
                    camera_paths = glob(camera_path.replace("opencv_cameras", "meta_*"))
                    cameras = {}
                    for c in camera_paths:
                        qpos = os.path.basename(c).split("_")[-1].replace(".json", "")
                        cameras[qpos] = json.load(open(c))
                    for qpos in cameras:
                        cameras[qpos] = convert(cameras[qpos], part=True)["frames"]
                else:
                    camera_path = camera_path.replace("opencv_cameras", "meta")
                    cameras = json.load(open(camera_path))
                    cameras = convert(cameras)["frames"]
            else:
                cameras = json.load(open(camera_path))["frames"]
            if self.is_part:
                meta = {}
                for c in camera_paths:
                    qpos = os.path.basename(c).split("_")[-1].replace(".json", "")
                    meta[qpos] = json.load(open(c))
                image_dir = os.path.dirname(camera_paths[0])
                num_cameras = len(cameras[qpos])
            else:
                meta = json.load(open(camera_path))
                image_dir = os.path.dirname(camera_path)
                num_cameras = len(cameras)

            if self.config.training.get("struct_views", False):
                print(self.config.training.num_input_views,
                    self.config.training.num_views)
                niv, nv = (
                    self.config.training.num_input_views,
                    self.config.training.num_views,
                )
                ref_idx = 0
                if num_cameras > 6:
                    image_choices = [
                        i for i in range(num_cameras)
                    ]
                else:
                    image_choices = [(ref_idx + i) % num_cameras for i in range(niv)]

                if nv > niv:
                    image_choices += random.sample(
                        list(set(range(num_cameras)) - set(image_choices)), nv - niv
                    )
            else:
                # skip scenes if num_views are not
                if num_cameras < self.config.training.num_views:
                    return self.__getitem__(random.randint(0, len(self) - 1))
                if self.is_part:
                    image_choices_0 = random.sample(
                        range(num_cameras), 3
                    )
                    image_choices_1 = random.sample(
                        range(num_cameras), 3
                    )
                    # print("******", len(camera_paths), num_cameras)
                    image_choices_mid = random.sample(
                        range(num_cameras * (len(camera_paths))), self.config.training.num_views - 6
                    )
                    
                else:
                    # unique indices
                    image_choices = random.sample(
                        range(num_cameras), self.config.training.num_views
                    )
            if self.is_part:
                image_paths_chosen = []
                cameras_chosen = []
                qpos_chosen = []
                for ic in image_choices_0:
                    image_paths_chosen.append(os.path.join(image_dir, cameras["0.00"][ic]["file_path"]))
                    cameras_chosen.append(cameras["0.00"][ic])
                    qpos_chosen.append(0.)
                for ic in image_choices_1:
                    image_paths_chosen.append(os.path.join(image_dir, cameras["1.00"][ic]["file_path"]))
                    cameras_chosen.append(cameras["1.00"][ic])
                    qpos_chosen.append(1.)
                for ic in image_choices_mid:
                    qposes = list(cameras.keys())
                    index_i = ic // num_cameras
                    index_j = ic % num_cameras
                    index_qpos = qposes[index_i]
                    qpos_chosen.append(float(index_qpos))
                    image_paths_chosen.append(os.path.join(image_dir, cameras[index_qpos][index_j]["file_path"]))
                    cameras_chosen.append(cameras[index_qpos][index_j])
            else:
                image_paths_chosen = [
                    os.path.join(image_dir, cameras[ic]["file_path"]) for ic in image_choices
                ]
                cameras_chosen = [cameras[ic] for ic in image_choices]
                

            input_images, input_fxfycxcy, input_c2ws, input_depths, input_normals, input_pos = [], [], [], [], [], []
            if self.is_part:
                input_masks = []
                input_depths = []
            for idx_chosen, (camera, image_path) in enumerate(
                zip(cameras_chosen, image_paths_chosen)
            ):
                image = Image.open(image_path)
                if self.is_part:
                    input_mask = Image.open(image_path.replace("color", "mask"))
                    input_depth = read_exr(image_path.replace("color", "depth").replace(".png", ".exr"))
                    input_depth = np.clip(input_depth, 0.0, 5.0) / 5.0

                resize_ratio = 1.0
                resize_to_y = self.config.model.image_tokenizer.image_size
                resize_ratio_x = resize_ratio_y = resize_ratio

                if image.size[1] != resize_to_y:
                    resize_ratio_y = resize_to_y / image.size[1]
                    resize_to_x = int(image.size[0] * resize_ratio_y)
                    resize_ratio_x = resize_to_x / image.size[0]
                    image = image.resize(
                        (resize_to_x, resize_to_y), resample=PIL.Image.LANCZOS
                    )
                    if self.is_part:
                        input_mask = input_mask.resize(
                            (resize_to_x, resize_to_y), resample=PIL.Image.NEAREST
                        )
                        input_depth = input_depth.resize(
                            (resize_to_x, resize_to_y), resample=PIL.Image.NEAREST
                        )

                # handle png with alpha channel
                if image.mode == "RGBA":
                    background = PIL.Image.new("RGB", image.size, (255, 255, 255))
                    mask = image.split()[3]
                    background.paste(image, mask=mask)
                    image = background
                    if not self.config.training.get("remove_alpha", True):
                        image.putalpha(mask)

                intrinsic_scale = 1.0
                fxfycxcy = np.array(
                    [
                        camera["fx"] / intrinsic_scale,
                        camera["fy"] / intrinsic_scale,
                        camera["cx"] / intrinsic_scale,
                        camera["cy"] / intrinsic_scale,
                    ]
                )
                fxfycxcy *= (
                    resize_ratio_x,
                    resize_ratio_y,
                    resize_ratio_x,
                    resize_ratio_y,
                )
                input_fxfycxcy.append(fxfycxcy)

                c2w = np.linalg.inv(camera["w2c"])
                if np.max(input_depth[input_depth<10000000000.0]) > 5.:
                    print(np.max(input_depth[input_depth<10000000000.0]), np.min(input_depth[input_depth<10000000000.0]), c2w[:3, 3])#, np.median(input_depth[input_depth<10000000000.0]), np.mean(input_depth[input_depth<10000000000.0]))
                    

                if 'up' in meta:
                    neg_grav = -calibur.normalized(meta['up'])
                    cor_grav = [0, 0, -1]
                    rotation = trimesh.transformations.quaternion_matrix(
                        trimesh.transformations.quaternion_about_axis(
                            trimesh.transformations.angle_between_vectors(neg_grav, cor_grav),
                            calibur.normalized(np.cross(neg_grav, cor_grav))
                        )
                    )
                    c2w = rotation @ c2w
                
                input_c2ws.append(c2w)
                input_pos.append(c2w[:3, 3])

                # (3, h, w)
                image = torch.from_numpy(
                    pil_to_np(image).astype(np.float32) / 255.0
                ).permute(2, 0, 1)
                input_images.append(image)
                
                if self.is_part:
                    input_mask = torch.from_numpy(
                        pil_to_np(input_mask).astype(np.float32) / 255.0
                    )[:, :]
                    input_mask = input_mask.reshape(1, 1, input_mask.shape[0], input_mask.shape[1])
                    mask = torch.Tensor(np.array(mask).reshape(1, 1, input_mask.shape[2], input_mask.shape[3])) / 255.
                    input_mask = torch.cat([input_mask, mask], dim=1)
                    input_masks.append(input_mask)
                    
                    input_depth = torch.from_numpy(
                        input_depth.copy()
                    )[:, :]
                    input_depth = input_depth.reshape(1, 1, input_depth.shape[0], input_depth.shape[1])
                    input_depths.append(input_depth)

                if not self.config.inference and (self.config.training.normal_loss_weight > 0.0 \
                                                  or self.config.training.get("normal_map_loss_weight", 0.0) > 0.0):
                    normal_path = image_path.replace("color", "normal").replace('png', 'exr')
                    normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)[..., :3][...,::-1]
                    if normal.shape[1] != resize_to_y:
                        normal = cv2.resize(normal, (resize_to_x, resize_to_y), interpolation=cv2.INTER_NEAREST)

                    # convert normal from camera coordinate to world coordinate
                    nx, ny = normal.shape[0], normal.shape[1]
                    normal = torch.from_numpy(normal.astype(np.float32))
                    normal = normal * 2 - 1

                    gl_c2w = calibur.convert_pose(c2w, calibur.CC.CV, calibur.CC.GL)
                    normal = calibur.transform_vector(normal, gl_c2w)

                    input_normals.append(normal.permute(2, 0, 1))

            input_images = torch.stack(input_images, dim=0)  # [v, 3, h, w]
            if self.is_part:
                input_masks = torch.stack(input_masks, dim=0)
                input_depths = torch.stack(input_depths, dim=0)
            input_fxfycxcy = np.array(input_fxfycxcy)
            input_c2ws = np.array(input_c2ws)
            input_pos = np.array(input_pos)

            if self.config.training.get("struct_views", False) and num_cameras > 6:
                image_choices = kmeans_downsample(input_pos, niv)
                print("k means", image_choices)
                input_images = input_images[image_choices]
                if self.is_part:
                    input_masks = input_masks[image_choices]
                    input_depths = input_depths[image_choices]
                input_fxfycxcy = input_fxfycxcy[image_choices]
                input_c2ws = input_c2ws[image_choices]
        except:
            traceback.print_exc()
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        if self.config.training.get("scale_poses", 0.0) > 0.0:
            input_c2ws[:, :3, 3] *= self.config.training.scale_poses

        poses = input_c2ws @ np.array([
                [1,0,0,0],
                [0,-1,0,0],
                [0,0,-1,0],
                [0,0,0,1],
            ])
        mv = np.linalg.inv(poses)
        input_mvs = mv
        if self.is_part:
            proj = np.array([self.projection_matrix(cameras_chosen[ic]['fx'],
                                                    cameras_chosen[ic]['fy'],
                                                    cameras_chosen[ic]['cx'], 
                                                    cameras_chosen[ic]['cy'],
                                                    cameras_chosen[ic]['h'],
                                                    cameras_chosen[ic]['w']) for ic in range(len(qpos_chosen))], dtype=np.float32)
        else:
            proj = np.array([self.projection_matrix(cameras[ic]['fx'],
                                                    cameras[ic]['fy'],
                                                    cameras[ic]['cx'], 
                                                    cameras[ic]['cy'],
                                                    cameras[ic]['h'],
                                                    cameras[ic]['w']) for ic in image_choices], dtype=np.float32)
        input_mvps = proj @ mv

        # convert to tensor
        input_c2ws = torch.from_numpy(input_c2ws).float()  # [v, 4, 4]
        input_fxfycxcy = torch.from_numpy(input_fxfycxcy).float()
        input_mvs = torch.from_numpy(input_mvs).float()  # [v, 4, 4]
        input_mvps = torch.from_numpy(input_mvps).float()  # [v, 4, 4]

        if not self.is_part:
            image_indices = (
                torch.from_numpy(np.array(image_choices)).long().unsqueeze(-1)
            )  # [v, 1]
            scene_indices = (
                torch.tensor(idx).long().unsqueeze(0).expand_as(image_indices)
            )  # [v, 1]
            indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]

        if self.is_part:
            return {
                "image": input_images,
                "mask": input_masks,
                "depth": input_depths,
                "c2w": input_c2ws,
                "qpos": torch.from_numpy(np.array(qpos_chosen)),
                "fxfycxcy": input_fxfycxcy,
                "mv": input_mvs,
                "mvp": input_mvps,
                "image_path": image_paths_chosen[0],
            }
        else:
            return {
                "image": input_images,
                "c2w": input_c2ws,
                "fxfycxcy": input_fxfycxcy,
                "mv": input_mvs,
                "mvp": input_mvps,
                "index": indices,
                "image_path": image_paths_chosen[0],
            }
