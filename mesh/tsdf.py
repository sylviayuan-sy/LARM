#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
import trimesh

# -----------------------------
# Filename parsing & utilities
# -----------------------------

def load_file_names(txt_path: str) -> List[str]:
    """
    Return tokens like '44817_joint_0' from a list file.
    Accepts lines that already contain 'eval_<obj>_joint_<idx>' or nested paths.
    """
    file_names = []
    pat = re.compile(r'(\d+_joint_\d+)')
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pat.search(line)
            if m:
                file_names.append(m.group(1))
                continue
            parts = line.split("/")
            # heuristic extractions
            for pick in (2, 4):
                try:
                    token = "_".join(parts[pick].split("_")[1:4])
                    if token:
                        file_names.append(token)
                        break
                except Exception:
                    pass
            else:
                print(f"[WARN] Could not parse file_name from line: {line}")
    return file_names


def write_debug_grid(out_path: str, img: np.ndarray, fg_mask: np.ndarray, part_mask: np.ndarray, depth: np.ndarray):
    H, W = fg_mask.shape
    strip = Image.new(mode="RGB", size=(W * 4, H))
    strip.paste(Image.fromarray(img.astype(np.uint8)), (0, 0))
    strip.paste(Image.fromarray((fg_mask * 255).astype(np.uint8)), (W, 0))
    strip.paste(Image.fromarray((part_mask * 255).astype(np.uint8)), (W * 2, 0))
    depth_viz = np.clip(depth * part_mask, 0, 5.0)
    denom = (depth_viz.max() + 1e-6)
    depth_viz = (depth_viz / denom * 255).astype(np.uint8)
    strip.paste(Image.fromarray(depth_viz), (W * 3, 0))
    strip.save(out_path)


def load_as_trimesh(mesh_path: str):
    try:
        loaded = trimesh.load(mesh_path, process=False)
    except Exception as e:
        print(f"[WARN] Failed to load mesh with trimesh ({mesh_path}): {e}")
        return None
    if isinstance(loaded, trimesh.Trimesh):
        if loaded.faces is None or len(loaded.faces) == 0:
            print(f"[WARN] Loaded Trimesh has 0 faces: {mesh_path}")
            return None
        return loaded
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            print(f"[WARN] Scene contained no mesh geometries: {mesh_path}")
            return None
        merged = trimesh.util.concatenate(geoms)
        if merged.faces.size == 0:
            print(f"[WARN] Concatenated Scene→Trimesh has 0 faces: {mesh_path}")
            return None
        return merged
    print(f"[WARN] Unsupported trimesh object type: {type(loaded)} for {mesh_path}")
    return None


def cleanup_trimesh(m: trimesh.Trimesh, min_component_faces: int) -> Optional[trimesh.Trimesh]:
    try:
        comps = m.split(only_watertight=False)
        big = [c for c in comps if c.faces.size >= min_component_faces]
        kept = trimesh.util.concatenate(big) if big else m
        kept.remove_unreferenced_vertices()
        kept.remove_degenerate_faces()
        kept.remove_duplicate_faces()
        kept.remove_infinite_values()
        try:
            kept.fill_holes()
        except Exception:
            pass
        if kept.faces.size == 0:
            return None
        return kept
    except Exception as e:
        print(f"[WARN] Trimesh cleanup failed: {e}")
        return m


# filenames like: 12_1.00.png  (qpos with 2 decimals; allows general floats too)
_QPOS_IMG_RE = re.compile(r'^(\d+)_([+-]?(?:\d+(?:\.\d+)?|\.\d+))\.png$')

def list_images_for_qpos(images_dir: str, qpos_only: float, max_frames: int) -> Tuple[str, List[Tuple[int, str, str]]]:
    """
    Return (qpos_str, [(idx, qpos_str, fn), ...]) for frames matching qpos_only,
    limited to first max_frames by ascending idx.
    """
    out = []
    for fn in os.listdir(images_dir):
        if not fn.endswith(".png"):
            continue
        if fn.endswith(("_mask.png", "_partmask.png", "_depth.png", "_sum_partmask.png")):
            continue
        m = _QPOS_IMG_RE.match(fn)
        if not m:
            continue
        idx = int(m.group(1))
        qstr = m.group(2)
        try:
            qval = float(qstr)
        except Exception:
            continue
        if abs(qval - qpos_only) < 1e-9:
            out.append((idx, f"{qval:.2f}", fn))
    out.sort(key=lambda t: t[0])
    if max_frames > 0:
        out = out[:max_frames]
    qpos_str = out[0][1] if out else ""
    return qpos_str, out


# -----------------------------------
# URDF locating and path rewriting
# -----------------------------------
from glob import glob

def _find_urdf_in_eval_dir(eval_dir: str, file_name: str) -> Optional[str]:
    """
    Look for a URDF in the current eval directory.
    Tries common names first, then recursive glob.
    """
    candidates = [
        os.path.join(eval_dir, f"eval_{file_name}.urdf"),
        os.path.join(eval_dir, "mobility.urdf"),
        os.path.join(eval_dir, f"{file_name}.urdf"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    hits = sorted(glob(os.path.join(eval_dir, "**", "*.urdf"), recursive=True))
    return hits[0] if hits else None


def rewrite_urdf_mesh_paths_for_file(file_name: str,
                                     base_clean_ply: Optional[str],
                                     part_clean_ply: Optional[str],
                                     eval_dir: str) -> bool:
    urdf_path = _find_urdf_in_eval_dir(eval_dir, file_name)
    if not urdf_path:
        print(f"[WARN] URDF not found for {file_name} under {eval_dir}")
        return False

    try:
        with open(urdf_path, "r") as f:
            text = f.read()
    except Exception as e:
        print(f"[WARN] Failed to read URDF for {file_name}: {e}")
        return False

    orig = text
    changed = False

    if base_clean_ply and os.path.exists(base_clean_ply) and "empty_base.obj" in text:
        text = text.replace("empty_base.obj", os.path.basename(base_clean_ply))
        changed = True

    if part_clean_ply and os.path.exists(part_clean_ply) and "empty_part.obj" in text:
        text = text.replace("empty_part.obj", os.path.basename(part_clean_ply))
        changed = True

    if not changed or text == orig:
        print(f"[INFO] No URDF mesh-path changes for {file_name} ({urdf_path}).")
        return False

    bak = urdf_path + ".bak"
    try:
        if not os.path.exists(bak):
            with open(bak, "w") as f:
                f.write(orig)
        with open(urdf_path, "w") as f:
            f.write(text)
        print(f"[OK] Updated URDF for {file_name}: {urdf_path}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to write updated URDF for {file_name}: {e}")
        return False


# -----------------------------
# TSDF integration per object
# -----------------------------
import calibur  # GL<->CV conversions

def _integrate_and_save(part_name: str,
                        images: List[Tuple[int, str, str]],
                        frames_meta,
                        images_dir: str,
                        meta,
                        path: str,
                        file_name: str,
                        qstr: str,
                        min_component_faces: int,
                        save_debug: bool,
                        rot_aug: bool = False) -> Optional[str]:
    """
    Integrate only frames listed in `images`:
    - depth is multiplied by 5.0 (or + 0.5 if rot_aug)
    - GL->CV conversion via calibur
    - Open3D extrinsic = inv(c2w_cv)
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.008,
        sdf_trunc=0.004 * 10,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    integrated = 0
    for idx, _, fn in images:
        if idx >= len(frames_meta):
            continue

        fr = frames_meta[idx]
        if "transform_matrix" not in fr:
            continue

        # ------ Pose: c2w (GL) -> c2w (CV) -> extrinsic ------
        c2w_gl = np.array(fr["transform_matrix"], dtype=np.float32)
        c2w_cv = calibur.convert_pose(c2w_gl, calibur.CC.GL, calibur.CC.CV)
        extrinsic = np.linalg.inv(c2w_cv)  # T_world->cam for Open3D

        # ------ IO paths ------
        im_path = os.path.join(images_dir, fn)
        fg_path = im_path.replace(".png", "_mask.png")
        part_path = im_path.replace(".png", "_partmask.png")
        sumpart_path = im_path.replace(".png", "_sum_partmask.png")
        depth_path = im_path.replace(".png", "_depth.npy")

        if not (os.path.exists(fg_path) and os.path.exists(part_path) and os.path.exists(depth_path)):
            continue

        # ------ Masks & image ------
        img = np.array(Image.open(im_path))
        fg_mask = (np.array(Image.open(fg_path)) / 255.0 > 0.5).astype(np.float32)
        part_mask = (np.array(Image.open(part_path)) / 255.0 > 0.5).astype(np.float32) * fg_mask

        if part_name == "part":
            cur_mask = part_mask
        elif part_name == "base":
            cur_mask = np.clip(fg_mask - part_mask, 0.0, 1.0)
        else:  # multilink_base
            if not os.path.exists(sumpart_path):
                continue
            sumpart = (np.array(Image.open(sumpart_path)) / 255.0 > 0.5).astype(np.float32) * fg_mask
            cur_mask = np.clip(fg_mask - sumpart, 0.0, 1.0)

        if np.sum(cur_mask) < 500:
            continue

        color = (img[:, :, :3] * fg_mask[:, :, None]).astype(np.uint8)
        depth_masked = (np.load(depth_path).astype(np.float32) + 0.5 if rot_aug else np.load(depth_path).astype(np.float32) * 5.0) * cur_mask  # *** scale to meters ***

        if save_debug:
            dbg_path = os.path.join(path, f"debug_{part_name}_{idx:03d}.png")
            write_debug_grid(dbg_path, img[:, :, :3], fg_mask, cur_mask, depth_masked)

        depth_masked = np.clip(depth_masked, 0.0, None)

        # ------ Open3D integration ------
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth_masked.astype(np.float32)),
            depth_scale=1.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            img.shape[1], img.shape[0], meta["fx"], meta["fy"], meta["cx"], meta["cy"]
        )
        volume.integrate(rgbd, intrinsic, extrinsic)
        integrated += 1

    if integrated == 0:
        print(f"[WARN] No frames integrated for {file_name} {part_name} (qpos={qstr}).")
        return None

    mesh = volume.extract_triangle_mesh()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    if np.asarray(mesh.triangles).shape[0] == 0:
        print(f"[WARN] Empty mesh extracted for {file_name} {part_name} — skipping")
        return None

    raw_ply = os.path.join(path, f"{file_name}_mesh_{qstr}_pred_{part_name}.ply")
    if not o3d.io.write_triangle_mesh(raw_ply, mesh):
        print(f"[WARN] Open3D failed to write PLY: {raw_ply}")
        return None

    m = load_as_trimesh(raw_ply)
    if m is None:
        print(f"[WARN] Skipping trimesh cleanup for {file_name} {part_name}: could not load as mesh")
        return None
    m = cleanup_trimesh(m, min_component_faces=min_component_faces)
    if m is None:
        print(f"[WARN] Clean resulted in empty mesh for {file_name} {part_name}")
        return None

    clean_ply = os.path.join(path, f"{file_name}_mesh_{qstr}_pred_{part_name}_clean.ply")
    try:
        m.export(clean_ply)
    except Exception as e:
        print(f"[WARN] Export failed ({file_name} {part_name}): {e}")
        return None

    return clean_ply


def process_one(path: str,
                file_name: str,
                max_frames: int,
                min_component_faces: int,
                save_debug: bool,
                qpos_only: float,
                rot_aug: bool = False):
    meta_path = os.path.join(path, "transforms.json")
    if not os.path.exists(meta_path):
        print(f"[WARN] Missing: {meta_path}")
        return False
    images_dir = os.path.join(path, "images")
    if not os.path.isdir(images_dir):
        print(f"[WARN] Missing images dir: {images_dir}")
        return False

    with open(meta_path, "r") as f:
        meta = json.load(f)
    frames_meta = meta.get("frames", [])
    if len(frames_meta) == 0:
        print(f"[WARN] No frames in {meta_path}")
        return False

    # -------- only qpos = qpos_only (default 1.00) and first N frames --------
    qstr, images = list_images_for_qpos(images_dir, qpos_only=qpos_only, max_frames=max_frames)
    if not images:
        print(f"[WARN] No qpos={qpos_only:.2f} images found in {images_dir}")
        return False

    base_clean_ply = _integrate_and_save("base", images, frames_meta, images_dir, meta, path, file_name, qstr, min_component_faces, save_debug, rot_aug)
    part_clean_ply = _integrate_and_save("part", images, frames_meta, images_dir, meta, path, file_name, qstr, min_component_faces, save_debug, rot_aug)

    # Try URDF rewrite (will only change if URDF has empty_* placeholders)
    _ = rewrite_urdf_mesh_paths_for_file(file_name, base_clean_ply, part_clean_ply, eval_dir=path)

    # Optional multilink_base if *_sum_partmask.png exists
    mbase_ply = _integrate_and_save("multilink_base", images, frames_meta, images_dir, meta, path, file_name, qstr, min_component_faces, save_debug, rot_aug)
    return (base_clean_ply is not None) or (part_clean_ply is not None) or (mbase_ply is not None)


def process_entry(load_folder: str,
                  file_name: str,
                  max_frames: int,
                  min_component_faces: int,
                  save_debug: bool,
                  qpos_only: float,
                  rot_aug: bool = False):
    path = os.path.join(load_folder, f"eval_{file_name}")
    if not os.path.isdir(path):
        print(f"[WARN] Missing eval folder: {path}")
        return
    process_one(path, file_name, max_frames, min_component_faces, save_debug, qpos_only, rot_aug)


def main():
    parser = argparse.ArgumentParser(
        description="TSDF reconstruction using ONLY qpos=1.00 (by default) and the first N frames. GL->CV pose conversion and depth*5.0 applied."
    )
    parser.add_argument("--txt_file", type=str, default="./data_sample_multilink/random_metadata/data.txt")
    parser.add_argument("--load_folder", type=str, default="./output_multilink")
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--min_component_faces", type=int, default=2000)
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--qpos_only", type=float, default=1.0, help="Process only frames whose filename qpos==this value (default 1.0).")
    parser.add_argument("--rot-aug", action="store_true",
                        help="Use rotation augmentation mode: depth + 0.5 instead of depth * 5.0")
    args = parser.parse_args()

    file_names = load_file_names(args.txt_file)
    for file_name in tqdm(file_names, desc="Objects"):
        process_entry(
            load_folder=args.load_folder,
            file_name=file_name,
            max_frames=args.max_frames,
            min_component_faces=args.min_component_faces,
            save_debug=args.save_debug,
            qpos_only=args.qpos_only,
            rot_aug=getattr(args, 'rot_aug', False),
        )


if __name__ == "__main__":
    main()
