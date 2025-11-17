import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms
import torchvision
import lpips
import numpy as np
from torch.masked import masked_tensor, as_masked_tensor
import sys
import os
import scipy.io
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from utils.util import to_xyz


class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu8 = nn.ReLU(inplace=True)
        self.pool3 = nn.AvgPool2d(2, 2)

        self.conv9 = nn.Conv2d(256, 512, 3, 1, 1)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu10 = nn.ReLU(inplace=True)
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu12 = nn.ReLU(inplace=True)
        self.pool4 = nn.AvgPool2d(2, 2)

        self.conv13 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu13 = nn.ReLU(inplace=True)
        self.conv14 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu14 = nn.ReLU(inplace=True)
        self.conv15 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu15 = nn.ReLU(inplace=True)
        self.conv16 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu16 = nn.ReLU(inplace=True)
        self.pool5 = nn.AvgPool2d(2, 2)

    def forward(self, x, return_style):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        out3 = x

        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool2(x)
        out8 = x

        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        x = self.relu8(self.conv8(x))
        x = self.pool3(x)
        out13 = x

        x = self.relu9(self.conv9(x))
        x = self.relu10(self.conv10(x))
        x = self.relu11(self.conv11(x))
        x = self.relu12(self.conv12(x))
        x = self.pool4(x)
        out22 = x

        x = self.relu13(self.conv13(x))
        x = self.relu14(self.conv14(x))
        x = self.relu15(self.conv15(x))
        x = self.relu16(self.conv16(x))
        out33 = x

        if return_style > 0:
            return [out3, out8, out13, out22, out33]
        else:
            return out3, out8, out13, out22, out33


class PerceptualLoss(nn.Module):
    def __init__(self, device="cpu", weight_file="imagenet-vgg-verydeep-19.mat"):
        super().__init__()
        self.net = VGG19()
        self.load_vgg_weights(weight_file)
        self.net = self.net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad = False

    def load_vgg_weights(self, weight_file):
        vgg_rawnet = scipy.io.loadmat(weight_file)
        vgg_layers = vgg_rawnet["layers"][0]
        layers = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        names = [f"conv{i+1}" for i in range(16)]
        channels = [64, 64, 128, 128, 256, 256, 256, 256,
                    512, 512, 512, 512, 512, 512, 512, 512]
        for i, name in enumerate(names):
            conv = getattr(self.net, name)
            w = torch.from_numpy(vgg_layers[layers[i]][0][0][2][0][0]).permute(3, 2, 0, 1)
            b = torch.from_numpy(vgg_layers[layers[i]][0][0][2][0][1]).view(channels[i])
            conv.weight = nn.Parameter(w)
            conv.bias = nn.Parameter(b)

    def compute_error(self, a, b):
        return torch.mean(torch.abs(a - b))

    def forward(self, pred_img, real_img):
        # Normalize to VGG style [0,255] and subtract mean
        mean = torch.tensor([123.68, 116.779, 103.939], device=pred_img.device).view(1, 3, 1, 1)
        real_img = real_img * 255.0 - mean
        pred_img = pred_img * 255.0 - mean

        f_r = self.net(real_img, return_style=0)
        f_p = self.net(pred_img, return_style=0)

        # Weighted L1 across layers
        weights = [1.0, 1/2.6, 1/4.8, 1/3.7, 10/1.5]
        errors = [self.compute_error(fr, fp) * w for fr, fp, w in zip(f_r, f_p, weights)]
        loss = (self.compute_error(real_img, pred_img) + sum(errors)) / 255.0
        return loss

class Loss(nn.Module):
    def __init__(self, weight, depth_weight, device):
        super().__init__()
        self.rgb_weight = weight
        self.depth_weight = depth_weight
        self.device = device

        self.mse = nn.MSELoss(reduction="none")
        self.mae = nn.L1Loss(reduction="none")
        self.mse_pm = nn.MSELoss(reduction="none")

        self.perceptual_loss_fn = PerceptualLoss(device=self.device)

    def compute_rgb_loss(self, pred, target, b, v):
        loss = self.mse(target, pred).mean(dim=[1, 2, 3])
        return loss.reshape(b, v).mean(dim=1).mean()

    def compute_perceptual_loss(self, pred, target, b, v):
        return self.perceptual_loss_fn(target, pred)

    def compute_fg_mask_loss(self, pred_mask, gt_mask, b):
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none").mean(dim=[1, 2, 3])
        return loss.reshape(b, -1).mean(dim=1).mean()

    def compute_depth_loss(self, pred_depth, gt_depth, fg_mask):
        masked_depth = pred_depth * fg_mask
        masked_gt = gt_depth * fg_mask
        loss = self.mae(masked_depth, masked_gt).sum() / fg_mask.sum()
        median_mae = torch.median(self.mae(masked_depth, masked_gt)[fg_mask == 1.])
        percent_mae_001 = torch.sum(self.mae(masked_depth, masked_gt)[fg_mask == 1.] < 0.01) / torch.numel(masked_depth[fg_mask == 1.])
        return loss, median_mae, percent_mae_001

    def compute_part_mask_loss(self, images_in, part_mask, fg_mask, intrinsics, depths, extrinsics):
        part_mask_loss, bbox_loss, pm_loss_list, xyz_gt_list = [], [], [], []
        for idx in range(images_in.shape[0]):
            pm_loss, xyz_gt = torch.tensor(0.).to(images_in.device), None
            current_part_mask = part_mask[idx, 0]
            nonzero = current_part_mask.nonzero(as_tuple=False)

            if nonzero.numel() > 0:
                y1, x1 = nonzero.min(dim=0).values
                y2, x2 = nonzero.max(dim=0).values
                logits_crop = images_in[idx, 3, y1:y2+1, x1:x2+1]
                target_crop = current_part_mask[y1:y2+1, x1:x2+1]
                bbox_loss.append(F.binary_cross_entropy_with_logits(logits_crop, target_crop, reduction="mean"))

                mask_bg = F.binary_cross_entropy_with_logits(images_in[idx, 3][current_part_mask == 0.],
                                                             current_part_mask[current_part_mask == 0.],
                                                             reduction="mean")
                mask_part = F.binary_cross_entropy_with_logits(images_in[idx, 3][current_part_mask == 1.],
                                                               current_part_mask[current_part_mask == 1.],
                                                               reduction="mean")
                part_mask_loss.append((mask_bg + mask_part) / 2.)
            else:
                part_mask_loss.append(F.binary_cross_entropy_with_logits(images_in[idx, 3], current_part_mask, reduction="mean"))
                bbox_loss.append(torch.tensor(0.0).to(images_in.device))

            # XYZ supervision via to_xyz if fg_mask has valid points
            fg_idx = (fg_mask[idx] == 1.0).nonzero(as_tuple=False)
            if fg_idx.numel() > 0:
                i, j = fg_idx[:, 1], fg_idx[:, 2]
                gt_d = depths[idx][i, j] * 5.
                xyz_gt = to_xyz(gt_d, j, i, intrinsics[idx, 2], intrinsics[idx, 3],
                                intrinsics[idx, 0], intrinsics[idx, 1], extrinsics[idx])
                xyz = 5. * images_in[idx, 6:, :, :][:, fg_mask[idx][0] == 1.0].T
                pm_loss = self.mse_pm(xyz, xyz_gt).mean()
                xyz_gt_list.append(xyz_gt)
            else:
                xyz_gt_list.append(torch.zeros_like(images_in[idx, 6:, :, :][:, fg_mask[idx][0]==1.].T).to(images_in.device))
                pm_loss = torch.tensor(0.).to(images_in.device)
            pm_loss_list.append(pm_loss)

        part_mask_loss = sum(part_mask_loss) / len(part_mask_loss)
        bbox_loss = sum(bbox_loss) / len(bbox_loss)
        pm_loss = sum(pm_loss_list) / len(pm_loss_list) if pm_loss_list else torch.tensor(0.0).to(images_in.device)
        return part_mask_loss, bbox_loss, pm_loss, xyz_gt_list

    def forward(self, images_in, images_out, masks, depths, intrinsics, extrinsics):
        b, v, c, h, w = images_out.shape
        _, n, _, _, _ = images_in.shape

        images_out = images_out.reshape(-1, c, h, w)
        target = images_in[:, n - v:, :3, :, :].reshape(-1, c, h, w)

        rgb_loss = self.compute_rgb_loss(images_out, target, b, v)
        perceptual_loss = self.compute_perceptual_loss(images_out, target, b, v)

        images_in = images_in.reshape(-1, c + 6, h, w)
        fg_mask = masks.reshape(-1, 2, h, w)[:, 1:2, :, :]
        part_mask = masks.reshape(-1, 2, h, w)[:, 0:1, :, :]

        fg_mask_loss = self.compute_fg_mask_loss(images_in[:, 4:5, :, :], fg_mask, b)

        depth = images_in[:, 5:6, :, :]
        gt_depth = depths.reshape(-1, 1, h, w)
        depth_loss, median_mae, percent_mae_001 = self.compute_depth_loss(depth, gt_depth, fg_mask)

        part_loss, bbox_loss, pm_loss, xyz_gt_list = self.compute_part_mask_loss(
            images_in, part_mask, fg_mask,
            intrinsics.reshape(-1, intrinsics.shape[-1]),
            gt_depth.squeeze(1),
            extrinsics.reshape(-1, 4, 4)
        )

        total_loss = (
            10.0 * rgb_loss +
            perceptual_loss +
            part_loss +
            fg_mask_loss +
            bbox_loss +
            self.depth_weight * (depth_loss + 0.0 * pm_loss)
        )

        return {
            "loss": total_loss,
            "mse_loss": rgb_loss,
            "perceptual_loss": perceptual_loss,
            "part_mask_loss": part_loss,
            "fg_mask_loss": fg_mask_loss,
            "depth_mae_loss": depth_loss,
            "point_map_loss": 0.0 * pm_loss,  # pm_loss is scaled to 0 as in original
            "bbox_part_mask_loss": bbox_loss,
            "median_mae": median_mae,
            "percent_mae_<0.01": percent_mae_001
        }, xyz_gt_list

