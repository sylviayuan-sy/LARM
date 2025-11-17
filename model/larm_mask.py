import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_mask import Decoder
from utils.util import plucker_embedding
from PIL import Image
import einops
import os
import numpy as np
from matplotlib import cm
import open3d as o3d
from open3d import geometry
from open3d.utility import Vector3dVector

# Tokenize image and camera views into patch embeddings
def patchify(intrinsic, extrinsic_in, extrinsic_out, image, patch_size, qpos_in=None, qpos_out=None):
    b, vi, _, _ = extrinsic_in.shape
    b, vo, _, _ = extrinsic_out.shape

    plucker_ray_out = plucker_embedding(intrinsic[:, vi:], extrinsic_out, image[:, vi:], qpos=qpos_out)
    plucker_ray_in = plucker_embedding(intrinsic[:, :vi], extrinsic_in, image[:, :vi], qpos=qpos_in)

    image = image.permute(0, 1, 3, 4, 2)[:, :vi]
    plucker_ray_in = plucker_ray_in.permute(0, 1, 3, 4, 2)
    plucker_ray_out = plucker_ray_out.permute(0, 1, 3, 4, 2)

    hh, ww = image.shape[2:4]
    c = image.shape[-1] + plucker_ray_in.shape[-1]

    # Encode input RGB + ray + (optional) qpos
    rgb_rays_in = torch.cat([image * 2 - 1, plucker_ray_in], dim=-1).repeat_interleave(vo, dim=0)
    input_tokens = einops.rearrange(
        rgb_rays_in, 'bo v (h ph) (w pw) c -> bo (v h w) (ph pw c)',
        bo=b * vo, v=vi, h=hh // patch_size, w=ww // patch_size,
        ph=patch_size, pw=patch_size, c=c
    )
    if qpos_in is not None:
        qpos_token_in = qpos_in.repeat_interleave(vo, dim=0).reshape(b * vo, vi, 1)
        qpos_token_in = qpos_token_in.repeat_interleave((hh // patch_size) * (ww // patch_size), dim=1)
        input_tokens = torch.cat([input_tokens, qpos_token_in], dim=-1).float()

    # Encode target ray + (optional) qpos
    target_tokens = einops.rearrange(
        plucker_ray_out, 'b o (h ph) (w pw) c -> (b o) (h w) (ph pw c)',
        b=b, o=vo, h=hh // patch_size, w=ww // patch_size,
        ph=patch_size, pw=patch_size, c=plucker_ray_out.shape[-1]
    )
    if qpos_out is not None:
        qpos_token_out = qpos_out.reshape(b * vo, 1, 1).repeat_interleave((hh // patch_size) * (ww // patch_size), dim=1)
        target_tokens = torch.cat([target_tokens, qpos_token_out.float()], dim=-1)

    return input_tokens, target_tokens, b, vi, vo

# Convert decoded tokens back into image/mask/depth/xyz maps
def unpatchify(decoded_tokens, patch_size, image, extrinsic_out, extrinsic_in):
    input_tokens, target_tokens, xyz = decoded_tokens
    b, vi, c, hh, ww = image.shape
    vo = extrinsic_out.shape[1]
    vie = extrinsic_in.shape[1]

    # Constants for channel indexing
    c_img, c_mask, c_depth, c_xyz = 3, 2, 1, 3

    result_target_image = einops.rearrange(
        target_tokens[:, :, :patch_size**2 * c_img],
        '(b o) (h w) (ph pw c) -> b o 1 (h ph) (w pw) c',
        b=b, o=vo, h=hh // patch_size, w=ww // patch_size,
        ph=patch_size, pw=patch_size
    )
    result_input_image = einops.rearrange(
        input_tokens[:, :, :patch_size**2 * c_img],
        '(b o) (v h w) (ph pw c) -> b o v (h ph) (w pw) c',
        b=b, o=vo, v=vie, h=hh // patch_size, w=ww // patch_size,
        ph=patch_size, pw=patch_size
    )
    result_image = torch.cat([result_input_image, result_target_image], dim=2).reshape(b, -1, hh, ww, c_img)

    result_target_mask = einops.rearrange(
        target_tokens[:, :, patch_size**2 * c_img : patch_size**2 * (c_img + c_mask)],
        '(b o) (h w) (ph pw c) -> b o 1 (h ph) (w pw) c',
        b=b, o=vo, h=hh // patch_size, w=ww // patch_size,
        ph=patch_size, pw=patch_size
    )
    result_input_mask = einops.rearrange(
        input_tokens[:, :, patch_size**2 * c_img : patch_size**2 * (c_img + c_mask)],
        '(b o) (v h w) (ph pw c) -> b o v (h ph) (w pw) c',
        b=b, o=vo, v=vie, h=hh // patch_size, w=ww // patch_size,
        ph=patch_size, pw=patch_size
    )
    result_mask = torch.cat([result_input_mask, result_target_mask], dim=2).reshape(b, -1, hh, ww, c_mask)

    result_target_depth = einops.rearrange(
        target_tokens[:, :, patch_size**2 * (c_img + c_mask):],
        '(b o) (h w) (ph pw c) -> b o 1 (h ph) (w pw) c',
        b=b, o=vo, h=hh // patch_size, w=ww // patch_size,
        ph=patch_size, pw=patch_size
    )
    result_input_depth = einops.rearrange(
        input_tokens[:, :, patch_size**2 * (c_img + c_mask):],
        '(b o) (v h w) (ph pw c) -> b o v (h ph) (w pw) c',
        b=b, o=vo, v=vie, h=hh // patch_size, w=ww // patch_size,
        ph=patch_size, pw=patch_size
    )
    result_depth = torch.cat([result_input_depth, result_target_depth], dim=2).reshape(b, -1, hh, ww, c_depth)

    return torch.cat([
        result_image.sigmoid(),
        result_mask,
        # result_depth.tanh(),
        result_depth.sigmoid(),
        xyz.reshape(b, -1, hh, ww, c_xyz)
    ], dim=-1)

# Main LARM model
class LARM(nn.Module):
    def __init__(
        self,
        hidden, num_layers, input_dim, target_dim,
        patch_size, linear_dim, resolution,
        num_target_views, num_input_views, batch_size
    ):
        super().__init__()
        self.decoder = Decoder(
            hidden, num_layers, input_dim, target_dim, linear_dim,
            patch_size=patch_size, batch_size=batch_size,
            num_target_views=num_target_views, num_input_views=num_input_views,
            resolution=resolution
        )
        self.patch_size = patch_size
        self.hidden = hidden
        self.resolution = resolution

    def forward(self, intrinsic, extrinsic_in, extrinsic_out, image, qpos_in=None, qpos_out=None):
        input_tokens, target_tokens, b, vi, vo = patchify(
            intrinsic, extrinsic_in, extrinsic_out, image,
            self.patch_size, qpos_in=qpos_in, qpos_out=qpos_out
        )
        decoded = self.decoder(input_tokens, target_tokens)
        return unpatchify(decoded, self.patch_size, image, extrinsic_out, extrinsic_in)
