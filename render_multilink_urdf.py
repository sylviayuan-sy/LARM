#!/usr/bin/env python3
"""
Pure-CPU headless multilink URDF renderer (OSMesa, trimesh, pyrender)
→ robust camera centering + correct look direction
→ interprets qpos as relative fraction of joint limit range.
"""

import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"  # force CPU software renderer

import numpy as np
import imageio
import trimesh
import pyrender
from urdfpy import URDF
from pathlib import Path


# ------------------ Utility ------------------
def compute_scene_bounds(scene):
    """Return (center, radius) of all mesh vertices in scene."""
    all_v = []
    for node in scene.get_nodes():
        if getattr(node, "mesh", None):
            for prim in node.mesh.primitives:
                if prim.positions is not None and len(prim.positions) > 0:
                    all_v.append(prim.positions)
    if not all_v:
        return np.zeros(3), 1.0
    all_v = np.vstack(all_v)
    center = (all_v.max(0) + all_v.min(0)) / 2.0
    radius = np.linalg.norm(all_v.max(0) - all_v.min(0)) / 2.0
    return center, radius


def look_at(camera_pos, target, up=(0, 0, 1)):
    """Create a 4x4 look-at transform matrix."""
    camera_pos = np.array(camera_pos)
    target = np.array(target)
    up = np.array(up)
    forward = (target - camera_pos)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = true_up
    mat[:3, 2] = -forward  # pyrender uses -Z forward
    mat[:3, 3] = camera_pos
    return mat


def add_soft_lights(scene, center, radius):
    """Add 3 gentle lights from different directions."""
    directions = np.array([[1, 1, 1], [-1, -1, 1], [1, -1, 1]])
    for d in directions:
        d = d / np.linalg.norm(d)
        pos = center + d * radius * 2.0
        light_pose = look_at(pos, center)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=light_pose)


# ------------------ Rendering ------------------
def render_pose(scene, output_path, resolution=(512, 512)):
    """Render the scene centered and facing the object."""
    center, radius = compute_scene_bounds(scene)
    cam_dist = max(2.2 * radius, 0.5)

    # position camera diagonally above and in front
    cam_pos = center + np.array([cam_dist, cam_dist, cam_dist * 0.8])
    cam_pose = look_at(cam_pos, center)

    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.5, znear=0.01, zfar=cam_dist * 10), pose=cam_pose)
    add_soft_lights(scene, center, radius)

    r = pyrender.OffscreenRenderer(*resolution)
    color, _ = r.render(scene)
    imageio.imwrite(output_path, color)
    print(f"✅ Saved {output_path}")
    r.delete()


def render_multilink_urdf(urdf_path, save_root, resolution=(512, 512)):
    os.makedirs(save_root, exist_ok=True)
    print(f"Loading URDF: {urdf_path}")
    robot = URDF.load(urdf_path)
    print(f"Loaded robot with {len(robot.joints)} joints and {len(robot.links)} links.")
    base_dir = Path(urdf_path).parent

    for j_idx, joint in enumerate(robot.joints):
        print(f"\nRendering joint {j_idx}: {joint.name}")

        # Extract limits safely
        lower = getattr(joint.limit, "lower", 0.0)
        upper = getattr(joint.limit, "upper", 0.0)
        if lower is None:
            lower = 0.0
        if upper is None:
            upper = 0.0
        if upper == lower:
            # Default to ±1 rad if unspecified
            lower, upper = -1.0, 1.0

        for qfrac in [0.0, 0.5, 1.0]:
            qval = lower + qfrac * (upper - lower)
            cfg = {joint.name: qval}
            link_poses = robot.link_fk(cfg=cfg)

            scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])
            for link in robot.links:
                if not link.visuals:
                    continue
                for vis in link.visuals:
                    mesh_geom = getattr(vis.geometry, "mesh", None)
                    if mesh_geom is None or not mesh_geom.filename:
                        continue
                    mesh_path = Path(mesh_geom.filename)
                    if not mesh_path.is_absolute():
                        mesh_path = base_dir / mesh_path
                    if not mesh_path.exists():
                        print(f"⚠️ Missing mesh: {mesh_path}")
                        continue
                    try:
                        mesh = trimesh.load_mesh(mesh_path, force='mesh')
                        pose = link_poses[link]
                        mesh.apply_transform(pose)
                        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
                    except Exception as e:
                        print(f"⚠️ Failed to load {mesh_path}: {e}")

            out_path = os.path.join(save_root, f"{joint.name}_qpos_{qfrac:.2f}.png")
            render_pose(scene, out_path, resolution)
            del scene


# ------------------ Entry ------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Headless OSMesa multilink renderer (centered + correct camera + scaled qpos)")
    parser.add_argument("--load_dir", type=str, required=True, help="Folder containing mobility.urdf")
    parser.add_argument("--res", type=int, nargs=2, default=[512, 512], help="Image resolution (width height)")
    args = parser.parse_args()

    urdf_path = os.path.join(args.load_dir, "eval_45213_joint_0.urdf")
    save_dir = os.path.join(args.load_dir, "check")
    render_multilink_urdf(urdf_path, save_dir, tuple(args.res))
