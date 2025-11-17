# Articulated Object Reconstruction Pipeline

This guide provides instructions to run the full pipeline for reconstructing 3D articulated objects from RGB-D data, consisting of two main alternatives:

1. **TSDF Reconstruction**
  Easier and faster to set up and run, but results in lower quality results. 
2. **SAP (Shape as Points) Pipeline**
  More complicated to set up and slower, but yield better quality meshes. 

---

## SECTION 1: TSDF Reconstruction

This step reconstructs the base and part geometry from LARM's NVS inference results with TSDF.

### To Run

```bash
cd mesh
python tsdf.py --txt_file /path/to/data_list.txt --load_folder /path/to/output_folder
```
Example:
```bash
cd mesh
python tsdf.py --txt_file ../data_sample/random_metadata/data.txt --load_folder ../output_random
```

- `--load_folder`: path to folder containing all inference output results
- `--txt_file`: path to a txt file listing all entries for processing

### Output Structure

The part and base mesh files will be saved to the load folder by default:

```
output_dir/
└── eval_{obj_id}_joint_{joint_idx}/
    ├── transforms.json
    ├── images/
    ├── eval_{obj_id}_joint_{joint_idx}.urdf
    ├── {obj_id}_joint_{joint_idx}_mesh_1.00_pred_base_clean.ply
    └── {obj_id}_joint_{joint_idx}_mesh_1.00_pred_base_clean.ply
    
```

---

## SECTION 2: SAP (Shape as Points) Pipeline

This step performs joint type and joint parameter estimation using the SAP method and links the results with colored meshes.

### Step-by-Step Instructions

#### 1. Clone the SAP Repository

```bash
git clone https://github.com/autonomousvision/shape_as_points.git
cd shape_as_points
```

Follow all installation instructions in their `README.md`, including dataset setup and environment installation.

#### 2. Run `preprocess_sap.py`

This script prepares input data and runs preprocessing steps for SAP.

```bash
python SAP/preprocess_sap.py \
    --load_dir /path/to/larm_results
```

#### 3. Run SAP Main Pipeline

Modify the config file `SAP/sap.yaml` as needed (e.g., input/output paths), then run:

```bash
python generate.py /path/to/sap.yaml
```

#### 4. Run `postprocess_sap.py`

This post-processes SAP predictions to recover mesh transformations, link the recovered joint parameters with the actual reconstructed meshes, and perform multi-view color projection to generate colored base and part meshes.

```bash
python SAP/postprocess_sap.py \
  --load_dir /path/to/larm_results
```

This completes the full pipeline from raw RGB-D to realistic, colored articulated mesh with estimated joints.

### Output Structure

The part and base mesh files will be saved to the load folder by default:

```
output_dir/
└── sap_urdf_final/
    └── eval_{obj_id}_joint_{joint_idx}/
        ├── mobility.urdf
        ├── eval_{obj_id}_joint_{joint_idx}_part.ply
        └── eval_{obj_id}_joint_{joint_idx}_base.ply

```
