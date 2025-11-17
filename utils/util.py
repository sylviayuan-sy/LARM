import os
import torch
import numpy as np
import torch.nn.functional as F
import einops
import wandb
from PIL import Image

# -----------------------------------------------
# Log dictionary of losses/metrics to Weights & Biases
# -----------------------------------------------
def log_to_wandb(loss_dict: dict, prefix: str = "train", extra_metrics: dict = None):
    log_data = {f"{prefix}/{k}": (v.item() if torch.is_tensor(v) else v) for k, v in loss_dict.items()}
    if extra_metrics:
        log_data.update({f"{prefix}/{k}": (v.item() if torch.is_tensor(v) else v) for k, v in extra_metrics.items()})
    wandb.log(log_data)


# -----------------------------------------------
# Convert depth + pixel coordinates into 3D (XYZ) points
# -----------------------------------------------
def to_xyz(depth, u, v, cx, cy, fx, fy, c2w):
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    xyz = torch.cat((x.unsqueeze(-1),y.unsqueeze(-1),depth.unsqueeze(-1),torch.ones((x.shape[0], 1)).to(depth.device)), dim=-1)
    xyz = (c2w @ xyz.T).T
    return xyz[:, :3]

# -----------------------------------------------
# Compute Plücker ray coordinates from intrinsics + extrinsics
# Citation: https://github.com/echen01/ray-conditioning/blob/8e1d5ae76d4747c771d770d1f042af77af4b9b5d/training/plucker.py#L47
# -----------------------------------------------
def plucker_embedding(intrinsic, extrinsic, image, qpos=None, pos_encoding=None):
    b, v, c, h, w = image.shape
    extrinsic = extrinsic.reshape(-1, 4, 4)

    # Generate pixel coordinate grid
    u, v_coords = torch.meshgrid(
        torch.arange(w, device=extrinsic.device) + 0.5,
        torch.arange(h, device=extrinsic.device) + 0.5,
        indexing="xy"
    )
    u = u.unsqueeze(0).repeat(extrinsic.shape[0], 1, 1)
    v_coords = v_coords.unsqueeze(0).repeat(extrinsic.shape[0], 1, 1)

    fx = intrinsic[:, :, 0].reshape(-1, 1, 1)
    fy = intrinsic[:, :, 1].reshape(-1, 1, 1)
    cx = intrinsic[:, :, 2].reshape(-1, 1, 1)
    cy = intrinsic[:, :, 3].reshape(-1, 1, 1)

    # Compute ray directions in world space
    directions = torch.stack(((u - cx) / fx, (v_coords - cy) / fy, torch.ones_like(u)), dim=-1)
    rays_d = torch.bmm(directions.reshape(-1, h * w, 3), extrinsic[:, :3, :3].permute(0, 2, 1))
    rays_d = F.normalize(rays_d, dim=-1)

    # Ray origins (camera position)
    rays_o = extrinsic[:, :3, 3].unsqueeze(1).expand(-1, h * w, 3)

    # Plücker coordinates: direction + moment
    cross = torch.cross(rays_o, rays_d, dim=-1)
    plucker = torch.cat([rays_d, cross], dim=-1)
    plucker = plucker.reshape(b, v, h, w, 6).permute(0, 1, 4, 2, 3)  # (B, V, 6, H, W)

    return plucker


# -----------------------------------------------
# Visualize input/output view batches as image and save
# -----------------------------------------------
def visualize(in_views, out_views, out_path):
    """
    Args:
        in_views: (B, V, C, H, W)
        out_views: (B, V, C, H, W)
        out_path: str - full path to output .png file
    """
    in_img = einops.rearrange(in_views, "b v c h w -> (b v) c h w")
    out_img = einops.rearrange(out_views, "b v c h w -> (b v) c h w")

    in_img = (in_img.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    out_img = (out_img.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    combined = np.concatenate([in_img, out_img], axis=0)
    grid = np.concatenate(list(combined), axis=1) if combined.shape[0] > 1 else combined[0]

    Image.fromarray(grid).save(out_path)
