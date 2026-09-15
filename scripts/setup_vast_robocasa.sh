#!/usr/bin/env bash
# Run as root inside pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
WORKSPACE="${WORKSPACE:-/workspace}"
MODEL_PATH="${MODEL_PATH:-$WORKSPACE/checkpoints/Xiaomi-Robotics-1-RoboCasa}"
apt-get update
apt-get install -y git libegl1 libgl1 libgles2 libosmesa6 libglfw3 ffmpeg tmux
python -m pip install transformers==4.57.1 torchvision==0.23.0 accelerate einops scipy tyro 'imageio[ffmpeg]' pytest
# The tested Docker image uses Python 3.11 and the PyTorch C++11 ABI.
python -m pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
hf download XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa \
  --revision 82b7c2210c1a39fca4a87e681cffbbed134ccde0 --local-dir "$MODEL_PATH"
conda create -y -n robocasa --override-channels -c conda-forge python=3.10 pip
SIM_PY=/opt/conda/envs/robocasa/bin/python
$SIM_PY -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
# Separate simulation environment is required by RoboCasa's legacy Numba pin.
git clone -b v1.5.1 --depth 1 https://github.com/ARISE-Initiative/robosuite.git "$WORKSPACE/robosuite"
git clone -b v0.2 --depth 1 https://github.com/robocasa/robocasa.git "$WORKSPACE/robocasa"
$SIM_PY -m pip install 'setuptools<81'
$SIM_PY -m pip install -e "$WORKSPACE/robosuite" -e "$WORKSPACE/robocasa" \
  numpy==1.23.3 mujoco==3.2.6 opencv-python==4.8.1.78 mink==0.0.5 scipy==1.13.1
$SIM_PY -m pip install transformers==4.57.1 tyro 'imageio[ffmpeg]' einops
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
$SIM_PY -m robocasa.scripts.setup_macros
$SIM_PY -c 'import builtins; builtins.input = lambda *a: "y"; from robocasa.scripts.download_kitchen_assets import download_kitchen_assets; download_kitchen_assets()'
echo "Ready. Model: $MODEL_PATH; simulation Python: $SIM_PY"
