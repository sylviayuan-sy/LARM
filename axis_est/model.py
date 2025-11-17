import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from open3d import geometry
import open3d as o3d
from open3d.utility import Vector3dVector

def ortho6d_from_rot_mat(rot):
    return torch.cat((rot[:, 0], rot_matrix[:, 1]))

def axis_angle_from_rot_mat(rot):
    angle = torch.acos(( rot[0, 0] + rot[1, 1] + rot[2, 2] - 1)/2)
    x = (rot[2, 1] - rot[1, 2])/torch.sqrt((rot[2, 1] - rot[1, 2])**2+(rot[0, 2] - rot[2, 0])**2+(rot[1, 0] - rot[0, 1])**2)
    y = (rot[0, 2] - rot[2, 0])/torch.sqrt((rot[2, 1] - rot[1, 2])**2+(rot[0, 2] - rot[2, 0])**2+(rot[1, 0] - rot[0, 1])**2)
    z = (rot[1, 0] - rot[0, 1])/torch.sqrt((rot[2, 1] - rot[1, 2])**2+(rot[0, 2] - rot[2, 0])**2+(rot[1, 0] - rot[0, 1])**2)
    return angle, torch.Tensor([x, y, z])
    
def rot_mat_from_axis_theta(axis, theta):
    # breakpoint()
    R = torch.eye(3, device=theta.device).unsqueeze(0).repeat(theta.shape[0], 1, 1)
    # breakpoint()
    axis = axis.unsqueeze(0).repeat(theta.shape[0],1)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    R[:, 0, 0] = cos_theta + (axis[:, 0]**2) * (1 - cos_theta)
    R[:, 0, 1] = axis[:, 0] * axis[:, 1] * (1 - cos_theta) - axis[:, 2] * sin_theta
    R[:, 0, 2] = axis[:, 0] * axis[:, 2] * (1 - cos_theta) + axis[:, 1] * sin_theta
    R[:, 1, 0] = axis[:, 0] * axis[:, 1] * (1 - cos_theta) + axis[:, 2] * sin_theta
    R[:, 1, 1] = cos_theta + (axis[:, 1]**2) * (1 - cos_theta)
    R[:, 1, 2] = axis[:, 1] * axis[:, 2] * (1 - cos_theta) - axis[:, 0] * sin_theta
    R[:, 2, 0] = axis[:, 0] * axis[:, 2] * (1 - cos_theta) - axis[:, 1] * sin_theta
    R[:, 2, 1] = axis[:, 1] * axis[:, 2] * (1 - cos_theta) + axis[:, 0] * sin_theta
    R[:, 2, 2] = cos_theta + (axis[:, 2]**2) * (1 - cos_theta)
    return R

# https://github.com/NVlabs/DigitalTwinArt/blob/1a48b402e4bf4bb7731296e8e230f0db3d86fe4f/network.py#L197
def rotat_from_6d(ortho6d):
    def normalize_vector(v, return_mag=False):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v / v_mag
        if (return_mag == True):
            return v, v_mag[:, 0]
        else:
            return v

    def cross_product(u, v):
        batch = u.shape[0]
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3
        return out

    x_raw = ortho6d[:, 0:3]  # batch*3  100
    y_raw = ortho6d[:, 3:6]  # batch*3
    x = normalize_vector(x_raw)  # batch*3  100
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix

class JE(nn.Module):
    def __init__(
        self,
        joint_type,
        device = "cuda"
        ):
        super().__init__()
        self.joint_type = joint_type
        if joint_type == "revolute":
            self.axis = torch.nn.Parameter(data=-torch.rand(7, ), requires_grad=True)
        elif joint_type == "prismatic":
            self.axis = torch.nn.Parameter(data=torch.rand(4, ), requires_grad=True)
        
    def forward(self, px, qpos_x, qpos_y):
        px = torch.cat([px[:, :3], torch.ones((px.shape[0], 1), device=px.device)], dim=-1)
        if self.joint_type == "revolute":
            joint_direction = self.axis[:3] / torch.linalg.norm(self.axis[:3])
            joint_origin = self.axis[3:6]
            joint_scale = self.axis[6]
            angle = (qpos_y - qpos_x) * joint_scale
            rot_mat = rot_mat_from_axis_theta(joint_direction, angle)
            translate = torch.eye(4, device=px.device)
            translate[:3, 3] = joint_origin
            rotate = torch.eye(4, device=px.device).unsqueeze(0).repeat(px.shape[0], 1, 1)
            rotate[:, :3, :3] = rot_mat
            Tpx = translate @ rotate @ torch.linalg.inv(translate) @ px.unsqueeze(-1)
        elif self.joint_type == "prismatic":
            joint_direction = self.axis[:3] / torch.linalg.norm(self.axis[:3])
            joint_scale = self.axis[3]
            translate = torch.eye(4, device=px.device).unsqueeze(0).repeat(px.shape[0], 1, 1)
            translate[:, :3, 3] = joint_direction.unsqueeze(0) * (qpos_y - qpos_x).unsqueeze(-1) * joint_scale
            Tpx = translate @ px.unsqueeze(-1)
        return Tpx[:, :3].squeeze(-1)