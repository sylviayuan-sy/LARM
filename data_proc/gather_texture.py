import argparse
import json
import os
from glob import glob
from typing import List
from tqdm import tqdm
from PIL import Image


DEFAULT_CATEGORIES = [
    "StorageFurniture",
    "Microwave",
    "Oven",
    "Refrigerator",
    "Safe",
    "TrashCan",
    "Table",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract near-square textures from PartNet-Mobility by category."
    )
    p.add_argument(
        "--pm-path",
        required=True,
        help="Root folder of partnet-mobility-v0 (contains model subfolders).",
    )
    p.add_argument(
        "--texture-path",
        required=True,
        help="Destination folder to save filtered textures.",
    )
    p.add_argument(
        "--output-list",
        default="./texture.txt",
        help="Path to write list of saved texture file paths (one per line).",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Model categories to include (from meta.json:model_cat).",
    )
    p.add_argument(
        "--min-ar",
        type=float,
        default=0.8,
        help="Minimum allowed aspect ratio h/w (inclusive). Default: 0.8",
    )
    p.add_argument(
        "--max-ar",
        type=float,
        default=1.25,
        help="Maximum allowed aspect ratio h/w (inclusive). Default: 1.25",
    )
    p.add_argument(
        "--exts",
        nargs="+",
        default=[".jpg"],
        help="Image extensions to scan under each model's images/ folder.",
    )
    return p.parse_args()


def list_model_dirs(pm_path: str) -> List[str]:
    # Only keep directories directly under pm_path
    candidates = glob(os.path.join(pm_path, "*"))
    return [d for d in candidates if os.path.isdir(d)]


def load_category(model_dir: str) -> str:
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return ""
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return str(meta.get("model_cat", ""))
    except Exception:
        return ""


def find_images(model_dir: str, exts: List[str]) -> List[str]:
    images_dir = os.path.join(model_dir, "images")
    if not os.path.isdir(images_dir):
        return []
    paths: List[str] = []
    for ext in exts:
        # Accept both lowercase and uppercase extensions
        paths.extend(glob(os.path.join(images_dir, f"*{ext}")))
        paths.extend(glob(os.path.join(images_dir, f"*{ext.upper()}")))
    return sorted(paths)


def main() -> None:
    args = parse_args()

    pm_path = os.path.abspath(args.pm_path)
    texture_path = os.path.abspath(args.texture_path)
    output_list = os.path.abspath(args.output_list)
    os.makedirs(texture_path, exist_ok=True)

    # 1) Filter model folders by category
    model_dirs = list_model_dirs(pm_path)
    sample_dirs: List[str] = []

    for d in tqdm(model_dirs, desc="Scanning models"):
        cat = load_category(d)
        if cat and cat in args.categories:
            sample_dirs.append(d)

    # 2) Collect and save textures meeting aspect ratio criteria
    saved_paths: List[str] = []
    count = 0

    for d in tqdm(sample_dirs, desc="Processing selected models"):
        image_paths = find_images(d, args.exts)
        for img_path in image_paths:
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    if w == 0:
                        continue
                    ar = h / float(w)
                    if args.min_ar <= ar <= args.max_ar:
                        out_path = os.path.join(texture_path, f"texture_{count}.jpg")
                        # Save as JPEG; convert if needed
                        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                            im = im.convert("RGB")
                        elif im.mode != "RGB":
                            im = im.convert("RGB")
                        im.save(out_path, format="JPEG")
                        saved_paths.append(out_path)
                        count += 1
            except Exception:
                # Skip unreadable images, keep going
                continue

    # 3) Write the list file
    with open(output_list, "w") as f:
        for p in saved_paths:
            f.write(p + "\n")

    print(f"Done. Saved {len(saved_paths)} textures to: {texture_path}")
    print(f"Wrote list to: {output_list}")


if __name__ == "__main__":
    main()