#!/usr/bin/env python3
import argparse
import json
import os
import re
import glob
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

META_RE = re.compile(r"^meta_([^/\\]+)\.json$")  # capture qpos token exactly as in filename
IMG_RE  = re.compile(r"^color_([^_]+)_in_(\d+)\.png$")  # color_{qpos}_in_{idx}.png
OBJ_ID_RE = re.compile(r"(?<!\d)(\d{3,8})(?!\d)")   # 3-8 digit number
JOINT_KEY_RE = re.compile(r"joint_(\d+)")

# ---------------------------
# Basic meta helpers
# ---------------------------
def find_sample_key(d: Dict[str, Any]) -> Optional[str]:
    if "sample_0" in d and isinstance(d["sample_0"], dict):
        return "sample_0"
    for k, v in d.items():
        if isinstance(k, str) and k.startswith("sample_") and isinstance(v, dict):
            return k
    return None

def pick_intrinsics(root: Dict[str, Any], sample: Dict[str, Any]) -> Optional[List[List[float]]]:
    if isinstance(sample.get("intrinsics"), list):
        return sample["intrinsics"]
    return None

def parse_qpos_token_from_filename(filename: str) -> Optional[str]:
    base = os.path.basename(filename)
    m = META_RE.match(base)
    return m.group(1) if m else None

def token_to_float(token: str) -> Optional[float]:
    try:
        return float(token)
    except Exception:
        return None

def intrinsics_equal(a: Any, b: Any, tol: float = 1e-6) -> bool:
    try:
        if a is None or b is None:
            return a is b
        if len(a) != len(b):
            return False
        for row_a, row_b in zip(a, b):
            if len(row_a) != len(row_b):
                return False
            for x, y in zip(row_a, row_b):
                if abs(float(x) - float(y)) > tol:
                    return False
        return True
    except Exception:
        return False

# ---------------------------
# joint_info.json parsing
# ---------------------------
def _normalize_obj_keys_to_str(d: Dict[Any, Any]) -> Dict[str, Any]:
    return {str(k): v for k, v in d.items()}

def _infer_object_and_joint(in_dir: str, joint_info: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], List[str]]:
    logs: List[str] = []
    joint_info = _normalize_obj_keys_to_str(joint_info)

    path_text = os.path.abspath(in_dir)
    path_ids = [m.group(1) for m in OBJ_ID_RE.finditer(path_text)]
    obj_id = next((oid for oid in path_ids if oid in joint_info), None)

    if obj_id is None:
        if len(joint_info) == 1:
            obj_id = next(iter(joint_info.keys()))
            logs.append(f"[INFO] Could not infer object id from path; using the only id in joint_info: {obj_id}")
        else:
            logs.append(f"[WARN] Could not infer object id from path and joint_info has multiple ids: {list(joint_info.keys())}")

    joint_key = None
    m = JOINT_KEY_RE.search(path_text)
    if m is not None:
        candidate = f"joint_{m.group(1)}"
        if obj_id and isinstance(joint_info.get(obj_id), dict) and candidate in joint_info[obj_id]:
            joint_key = candidate
        else:
            joint_key = candidate

    if obj_id and joint_key is None:
        joints = joint_info.get(obj_id, {})
        if isinstance(joints, dict) and len(joints) > 0:
            if len(joints) == 1:
                joint_key = next(iter(joints.keys()))
                logs.append(f"[INFO] Using the only joint for object {obj_id}: {joint_key}")
            else:
                joint_key = "joint_0" if "joint_0" in joints else sorted(joints.keys())[0]
                logs.append(f"[INFO] Multiple joints for object {obj_id}; defaulting to {joint_key}")

    return obj_id, joint_key, logs

def read_joint_type_from_joint_info(joint_info_path: str, in_dir: str) -> Tuple[str, Optional[str]]:
    if not os.path.isfile(joint_info_path):
        return "revolute", f"[WARN] joint_info.json not found at {joint_info_path}; defaulting joint_type='revolute'."

    try:
        with open(joint_info_path, "r") as f:
            info = json.load(f)
    except Exception as e:
        return "revolute", f"[WARN] Failed to read {joint_info_path} ({e}); defaulting joint_type='revolute'."

    if not isinstance(info, dict) or len(info) == 0:
        return "revolute", f"[WARN] joint_info.json not a non-empty mapping; defaulting joint_type='revolute'."

    obj_id, joint_key, logs = _infer_object_and_joint(in_dir, info)
    for msg in logs:
        print(msg)

    if obj_id is None:
        if len(info) == 1:
            obj_id = next(iter(info))
        else:
            return "revolute", f"[WARN] Unable to infer object id from path and multiple ids present in joint_info; defaulting joint_type='revolute'."

    joints = info.get(str(obj_id), {})
    if not isinstance(joints, dict) or len(joints) == 0:
        return "revolute", f"[WARN] No joints listed for object {obj_id}; defaulting joint_type='revolute'."

    if joint_key is None:
        if len(joints) == 1:
            joint_key = next(iter(joints))
        else:
            joint_key = "joint_0" if "joint_0" in joints else sorted(joints.keys())[0]
            print(f"[INFO] Defaulting to joint key: {joint_key}")

    joint_entry = joints.get(joint_key)
    if not isinstance(joint_entry, dict):
        return "revolute", f"[WARN] Entry for {obj_id}/{joint_key} missing or malformed; defaulting joint_type='revolute'."

    jt = joint_entry.get("joint_type")
    if isinstance(jt, str) and jt.strip():
        return jt.strip(), None
    return "revolute", f"[WARN] 'joint_type' missing under {obj_id}/{joint_key}; defaulting to 'revolute'."

# ---------------------------
# Image-first scan
# ---------------------------
def scan_images_by_qpos(in_dir: str) -> Dict[str, List[Tuple[int, str]]]:
    pattern = os.path.join(in_dir, "color_*_in_*.png")
    paths = glob.glob(pattern)
    groups: Dict[str, List[Tuple[int, str]]] = {}
    for p in paths:
        name = os.path.basename(p)
        m = IMG_RE.match(name)
        if not m:
            continue
        qpos_token = m.group(1)
        idx = int(m.group(2))
        groups.setdefault(qpos_token, []).append((idx, p))
    for q in groups:
        groups[q].sort(key=lambda t: t[0])
    return groups

def load_meta_for_qpos(in_dir: str, qpos_token: str, sample_key: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[List[List[float]]]]:
    meta_path = os.path.join(in_dir, f"meta_{qpos_token}.json")
    if not os.path.isfile(meta_path):
        print(f"[WARN] Missing meta for qpos={qpos_token}: {meta_path}")
        return None, None
    try:
        with open(meta_path, "r") as f:
            root = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {meta_path}: {e}")
        return None, None

    key = sample_key or find_sample_key(root)
    sample = root if key is None else root.get(key, root)
    if not isinstance(sample, dict):
        print(f"[WARN] Malformed sample in {meta_path}; expected dict.")
        return None, None

    intr = pick_intrinsics(root, sample)
    return sample, intr

def build_frames_from_images(
    in_dir: str,
    qpos_token: str,
    images: List[Tuple[int, str]],
    sample: Optional[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    frames: Dict[str, Dict[str, Any]] = {}
    qpos_float = token_to_float(qpos_token)
    if qpos_float is None:
        print(f"[WARN] Non-numeric qpos token encountered: '{qpos_token}'")
        qpos_float = float('nan')

    for new_idx, (orig_idx, img_path) in enumerate(images):
        key = f"input_frame_{new_idx}"
        meta_key = f"input_frame_{orig_idx}"

        tm = None
        if isinstance(sample, dict):
            v = sample.get(meta_key)
            if isinstance(v, dict):
                tm = v.get("transform_matrix")
                if tm is None:
                    print(f"[WARN] {meta_key} missing 'transform_matrix' in meta for qpos={qpos_token}")
            else:
                print(f"[WARN] {meta_key} missing in meta for qpos={qpos_token}")
        else:
            print(f"[WARN] No meta sample available for qpos={qpos_token}; frames will lack transforms.")

        frames[key] = {
            "transform_matrix": tm,
            "image_path": img_path,
            "qpos": qpos_float,
        }
    return frames

# ---------------------------
# KMeans (same as inference script)
# ---------------------------
def _kmeans_pp_init(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = points.shape[0]
    centers = np.empty((k, points.shape[1]), dtype=np.float64)
    idx0 = int(rng.integers(0, n))
    centers[0] = points[idx0]
    d2 = np.full(n, np.inf, dtype=np.float64)
    for c in range(1, k):
        d2 = np.minimum(d2, np.sum((points - centers[c - 1]) ** 2, axis=1))
        probs = d2 / (d2.sum() + 1e-12)
        idx = int(rng.choice(n, p=probs))
        centers[c] = points[idx]
    return centers

def _lloyd(points: np.ndarray, centers: np.ndarray, iters: int = 15) -> np.ndarray:
    for _ in range(iters):
        dists = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        for k in range(centers.shape[0]):
            sel = (labels == k)
            if np.any(sel):
                centers[k] = points[sel].mean(axis=0)
    return centers

def kmeans_downsample(points: List[List[float]], n_points_to_sample: int, iters: int = 15, seed: int = 0) -> List[int]:
    pts = np.asarray(list(points), dtype=np.float64)
    n = pts.shape[0]
    if n == 0 or n_points_to_sample <= 0:
        return []
    k = min(max(1, n_points_to_sample), n)
    rng = np.random.default_rng(seed)
    centers = _kmeans_pp_init(pts, k, rng)
    centers = _lloyd(pts, centers, iters=iters)
    d2 = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    chosen: List[int] = []
    for c in range(k):
        order = np.argsort(d2[:, c])
        pick = next((int(i) for i in order if int(i) not in chosen), int(order[0]))
        chosen.append(pick)
    return chosen

# ---------------------------
# Helpers for candidate selection and splitting
# ---------------------------
def _pos_from_tm(tm: Any) -> Optional[List[float]]:
    try:
        arr = np.asarray(tm, dtype=np.float64)
        if arr.shape == (4, 4):
            t = arr[:3, 3]
            return [float(t[0]), float(t[1]), float(t[2])]
    except Exception:
        pass
    return None

def _numeric_qpos_tokens(frames_by_qpos: Dict[str, Dict[str, Any]]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for tok in frames_by_qpos.keys():
        val = token_to_float(tok)
        if val is not None:
            out.append((tok, val))
    out.sort(key=lambda kv: kv[1])
    return out

def select_extreme_inputs(
    frames_by_qpos: Dict[str, Dict[str, Any]],
    per_extreme: int,
    iters: int,
    seed: int
) -> List[Tuple[str, str]]:
    q_tokens = _numeric_qpos_tokens(frames_by_qpos)
    if not q_tokens:
        return []

    min_tok = q_tokens[0][0]
    max_tok = q_tokens[-1][0]

    def _pick_for_token(tok: str, per_k: int, seed_offset: int) -> List[Tuple[str, str]]:
        frames = frames_by_qpos.get(tok, {})
        candidates: List[str] = []
        points: List[List[float]] = []
        for fkey, rec in frames.items():
            tm = rec.get("transform_matrix")
            img = rec.get("image_path")
            pos = _pos_from_tm(tm)
            if pos is None:
                continue
            if not (isinstance(img, str) and os.path.isfile(img)):
                continue
            candidates.append(fkey)
            points.append(pos)
        if not candidates:
            return []
        idxs = kmeans_downsample(points, n_points_to_sample=min(per_k, len(candidates)), iters=iters, seed=seed + seed_offset)
        return [(tok, candidates[i]) for i in idxs]

    picks_min = _pick_for_token(min_tok, per_extreme, seed_offset=0)
    picks_max = _pick_for_token(max_tok, per_extreme, seed_offset=12345 if max_tok != min_tok else 1)

    all_picks = picks_min + [p for p in picks_max if p not in picks_min]
    return all_picks

def split_inputs_targets(
    frames_by_qpos: Dict[str, Dict[str, Any]],
    picked: List[Tuple[str, str]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Split frames into inputs and targets.

    - Inputs are reindexed sequentially (input_frame_0, 1, 2, ...)
    - Targets are keyed by the *original camera index* parsed from image_path
      (target_frame_{orig_idx}), and we SKIP any orig_idx that were used as inputs.
    """
    picked_set = set(picked)
    inputs_by_qpos: Dict[str, Dict[str, Any]] = {}
    targets_by_qpos: Dict[str, Dict[str, Any]] = {}

    def _orig_cam_idx_from_path(p: Optional[str]) -> Optional[int]:
        if not isinstance(p, str):
            return None
        name = os.path.basename(p)
        m = IMG_RE.match(name)  # color_{qpos}_in_{idx}.png
        if not m:
            return None
        try:
            return int(m.group(2))
        except Exception:
            return None

    for qpos_token, frames in frames_by_qpos.items():
        # Stable order by our internal numeric input_frame index
        def _idx(k: str) -> int:
            try:
                return int(k.split("input_frame_")[1])
            except Exception:
                return 1 << 30
        ordered_keys = sorted(frames.keys(), key=_idx)

        # --- Inputs: sequentially reindexed
        in_dict: Dict[str, Any] = {}
        input_orig_indices: set[int] = set()
        for fkey in ordered_keys:
            if (qpos_token, fkey) in picked_set:
                rec = frames[fkey]
                in_dict[f"input_frame_{len(in_dict)}"] = {
                    "transform_matrix": rec.get("transform_matrix"),
                    "image_path": rec.get("image_path"),
                    "qpos": rec.get("qpos"),
                }
                oi = _orig_cam_idx_from_path(rec.get("image_path"))
                if oi is not None:
                    input_orig_indices.add(oi)
        if in_dict:
            inputs_by_qpos[qpos_token] = in_dict

        # --- Targets: keyed by original camera index & excluding input indices
        tar_dict: Dict[str, Any] = {}
        for fkey in ordered_keys:
            if (qpos_token, fkey) in picked_set:
                continue  # skip input frames entirely
            rec = frames[fkey]
            oi = _orig_cam_idx_from_path(rec.get("image_path"))
            # If we can't parse the original index, skip (or fallback if you prefer)
            if oi is None:
                continue
            # Exclude any original indices that were used as inputs
            if oi in input_orig_indices:
                continue
            tar_dict[f"target_frame_{oi}"] = {
                "transform_matrix": rec.get("transform_matrix"),
                "qpos": rec.get("qpos"),
            }
        if tar_dict:
            targets_by_qpos[qpos_token] = tar_dict

    return inputs_by_qpos, targets_by_qpos




# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build consolidated JSON with KMeans-sampled INPUT views (min/max qpos) and optional TARGET views."
    )
    ap.add_argument("--in-dir", required=True, help="Directory with meta_{qpos}.json and color_{qpos}_in_{idx}.png images.")
    ap.add_argument("--out-json", required=True, help="Path to write the consolidated JSON.")
    ap.add_argument("--joint-info", default="data_sample/joint_info.json",
                    help="Path to joint_info.json (default: data_sample/joint_info.json).")
    ap.add_argument("--mode", choices=["random", "view"], required=True,
                    help="random: write only inputs; view: write inputs and targets (no image paths in targets).")
    # Extreme sampling settings
    ap.add_argument("--inputs-per-extreme", type=int, default=3,
                    help="Number of input views to select from min-qpos AND from max-qpos (default: 3 each).")
    ap.add_argument("--kmeans-iters", type=int, default=15, help="KMeans iterations for input view selection.")
    ap.add_argument("--kmeans-seed", type=int, default=0, help="Random seed for KMeans init.")
    args = ap.parse_args()

    # joint_type
    consolidated: Dict[str, Any] = {"intrinsics": None}
    joint_type, jt_warn = read_joint_type_from_joint_info(args.joint_info, args.in_dir)
    consolidated["joint_type"] = joint_type
    if jt_warn:
        print(jt_warn)

    # Discover images
    img_groups = scan_images_by_qpos(args.in_dir)
    if not img_groups:
        print(f"[WARN] No images matching color_*_in_*.png found in {args.in_dir}")

    first_intrinsics: Optional[List[List[float]]] = None
    all_frames_by_qpos: Dict[str, Dict[str, Any]] = {}

    # Build frames per qpos
    for qpos_token, images in sorted(img_groups.items(), key=lambda kv: kv[0]):
        sample, intr = load_meta_for_qpos(args.in_dir, qpos_token, "sample_0")
        if first_intrinsics is None and intr is not None:
            first_intrinsics = intr
        elif intr is not None and not intrinsics_equal(first_intrinsics, intr):
            print(f"[WARN] Intrinsics differ in meta_{qpos_token}.json. Using the first encountered.")
        frames = build_frames_from_images(args.in_dir, qpos_token, images, sample)
        all_frames_by_qpos[qpos_token] = frames

    # Select inputs from min & max qpos
    picked_pairs = select_extreme_inputs(
        all_frames_by_qpos,
        per_extreme=args.inputs_per_extreme,
        iters=args.kmeans_iters,
        seed=args.kmeans_seed
    )

    # Split
    inputs_by_qpos, targets_by_qpos = split_inputs_targets(all_frames_by_qpos, picked_pairs)

    # Assemble final JSON
    consolidated["intrinsics"] = first_intrinsics
    consolidated["inputs"] = inputs_by_qpos
    if args.mode == "view":
        consolidated["targets"] = targets_by_qpos  # include targets only in view mode

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(consolidated, f, indent=2)

    # Summary
    total_inputs = sum(len(v) for v in inputs_by_qpos.values())
    total_targets = sum(len(v) for v in targets_by_qpos.values()) if args.mode == "view" else 0
    print(f"Wrote {args.out_json}")
    print(f"  - Mode: {args.mode}")
    print(f"  - Intrinsics: {'present' if consolidated['intrinsics'] is not None else 'absent'}")
    print(f"  - QPOS groups discovered: {len(all_frames_by_qpos)}")
    print(f"  - Input views selected (min/max qpos): {total_inputs} (requested up to {2 * args.inputs_per_extreme})")
    if args.mode == "view":
        print(f"  - Target views (no image paths): {total_targets}")
    else:
        print(f"  - Target views: omitted in random mode")
    print(f"  - joint_type: {consolidated.get('joint_type')}")
    print(f"  - joint_info path: {args.joint_info}")
    print(f"  - Source dir: {args.in_dir}")

if __name__ == "__main__":
    main()
