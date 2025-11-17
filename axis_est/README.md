# LARM Joint Estimation

This repository contains the joint estimation part of the **LARM** pipeline. It uses LoFTR for feature matching and RANSAC-based methods to estimate joint parameters from LARM outputs.

---

## Setup

### 1. Download LoFTR Model Pretrained Weights

- Clone LoFTR:

  ```bash
  git clone https://github.com/zju3dv/LoFTR.git
  ```
Download pretrained LoFTR weights [(indoor_ds.ckpt)](https://drive.google.com/file/d/1ZRAQ-V4-BNwCMPQRfYmebnFGouAvyCa1/view?usp=drive_link) from the LoFTR releases. Place the weights in `axis_est/weights` directory.

### 2. Running Joint Estimation
- Run the script with:

    ```bash
    cd axis_est
    python estimate_ransac.py --load_dir /path/to/larm/output_dir --datalist_path /path/to/data.txt

    example:
    python estimate_ransac.py --load_dir ../output_random/ --datalist_path ../data_sample/random_metadata/data.txt
    ```
Replace `/path/to/larm/output_dir` with the path to your LARM output folder containing transforms.json and other necessary files.

The script will:
- Use LoFTR for feature matching between image pairs.
- Perform RANSAC-based estimation to find joint parameters.

### Output Structure

The urdf file will be saved to the load folder by default:

```
output_dir/
└── eval_{obj_id}_joint_{joint_idx}/
    ├── transforms.json
    ├── images/
    ├── eval_{obj_id}_joint_{joint_idx}.urdf
    ├── empty_part.obj (placeholder part mesh)
    └── empty_base.obj (placeholder base mesh)
    
```

The output urdf files will contain links to placeholder meshes that you can ignore. 
The urdf files contain:
 - A base link;
 - A part link;

 - A single predicted joint with predicted joint parameters. 

