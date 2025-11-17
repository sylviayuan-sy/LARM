
# Blender-Based Data Processing for PartNet-Mobility

This pipeline generates RGB-D images, masks, and mesh-based ground truth for articulated objects using BlenderProc and PartNet-Mobility. It covers texture gathering, HDRI preparation, rendering, and ground truth export.

---

## Installation

To install the required Blender environment and dependencies, run the following setup script:

```bash
bash install_env_blender.sh
```

---

##  Directory Structure

```
.
├── gather_texture.py
├── get_hdri.py
├── render_all.py
├── export_all.py
├── hdri/                   # HDRI assets (downloaded)
├── metadata/               # Downloaded partnet-mobility metadata
├── partnet-mobility-v0/    # Input dataset (URDFs, textures, images)
└── texture/  # Gathered texture images
```

---

## Step-by-Step Instructions

### 1. Gather Textures

Extract texture images from all PartNet-Mobility objects into a central folder:

```bash
python gather_texture.py --pm-path=/path/to/partnet-mobility-v0 --texture-path=./texture
```

This collects all `.jpg` texture images into `./texture`, which is later used for randomized texture augmentation.

---

### 2. Render RGB-D Images and Masks

Use the renderer script to produce RGB, depth, normal, and mask images for each object:

```bash
python worker_blender.py \
  --data_root /path/to/partnet-mobility-v0 \
  --output_dir /path/to/output/renderings \
  --qpos-count n
```

This will:
- Apply randomized textures and lighting
- Render each object from multiple poses
- Save color, depth, normal maps, and object masks

---

### 3. Export Ground Truth Meshes for Evaluation

Use this to export ground truth mesh sequences at each joint pose, which are required for geometric metric computation (e.g., Chamfer distance):

```bash
python get_gt_mesh.py \
  --data_root /path/to/partnet-mobility-v0 \
  --datalist_path /path/to/data.txt \
  --output_dir /path/to/output/meshes
```

This generates `.ply` meshes aligned with the rendered joint poses for later evaluation.