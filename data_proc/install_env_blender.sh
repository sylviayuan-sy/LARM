#!/usr/bin/env bash

BLENDER_VER=3.3.1
BLENDER_DIR="/opt/blender-${BLENDER_VER}-linux-x64"
BLENDER_TAR="blender-${BLENDER_VER}-linux-x64.tar.xz"
BLENDER_URL="https://mirrors.ocf.berkeley.edu/blender/release/Blender3.3/${BLENDER_TAR}"

apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates curl git wget unzip python3-pip \
  libglib2.0-0 libsm6 libxrender1 libxext6 \
  libxi6 libxkbcommon-x11-0 freeglut3-dev \
  libglfw3-dev libxkbcommon0 libfreeimage-dev rclone
apt-get install -y --reinstall libgl1-mesa-dri || true
ln -sfn /usr/lib/x86_64-linux-gnu/dri /usr/lib/dri || true

install -d /opt
wget -q "${BLENDER_URL}" -O "/opt/${BLENDER_TAR}"
tar -xJf "/opt/${BLENDER_TAR}" -C /opt
rm -f "/opt/${BLENDER_TAR}"

pip install --no-cache-dir blenderproc==2.6.0

pip install --no-cache-dir \
  sapien==3.0.1 scipy trimesh urdfpy boto3 awscli calibur "networkx>=2.6,<3" \
  git+https://github.com/eliphatfs/kubetk.git \
  git+https://github.com/eliphatfs/imgsvc.git \
  git+https://github.com/SarahWeiii/s3_loader.git

blenderproc pip --custom-blender-path="${BLENDER_DIR}" install --no-cache-dir \
  calibur urdfpy "networkx>=2.6,<3"

conda install -c conda-forge -y gcc || true

apt-get clean
rm -rf /var/lib/apt/lists/*

echo "✅ Setup complete."
