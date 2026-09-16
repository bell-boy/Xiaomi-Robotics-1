set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
# hf_transfer fetches each checkpoint file over many parallel ranges. Without
# it a 10 GB shard arrives over one connection, which is far below the host's
# link speed. Keep the HF cache so an interrupted download resumes.
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /workspace/checkpoints
apt-get update
apt-get install -y git rsync libegl1 libgl1 libgles2 libopengl0 libosmesa6 libglfw3 ffmpeg tmux
python -m pip install transformers==4.57.1 torchvision==0.23.0 accelerate einops scipy tyro 'imageio[ffmpeg]' pytest nvidia-ml-py 'huggingface_hub[cli]' hf_transfer
python -m pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
hf download XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 --revision 3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4 --local-dir /workspace/checkpoints/Xiaomi-Robotics-1-RoboCasa365 > /workspace/download-365.log 2>&1 &
pid365=$!
hf download XiaomiRobotics/Xiaomi-Robotics-1-5B --revision ee21d524b5c52ac961d941e1bc7d6d92836c3d5e --local-dir /workspace/checkpoints/Xiaomi-Robotics-1-5B > /workspace/download-base.log 2>&1 &
pidbase=$!
python -m venv --system-site-packages /workspace/robocasa365-eval-venv
SIM_PY=/workspace/robocasa365-eval-venv/bin/python
git clone https://github.com/ARISE-Initiative/robosuite.git /workspace/robosuite365
git -C /workspace/robosuite365 checkout 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
git clone https://github.com/robocasa/robocasa.git /workspace/robocasa365
cd /workspace/robocasa365
git checkout 4f8a2980def75a55dff96b990745b83540425f09
$SIM_PY -m pip install -e /workspace/robosuite365 numpy==2.2.5 numba==0.61.2 scipy==1.15.3 mujoco==3.3.1 pygame Pillow opencv-python pyyaml pynput tqdm termcolor imageio h5py lxml hidapi gymnasium==0.29.1
# The simulator uses runtime dependencies only; lerobot/tianshou training stacks are unnecessary.
$SIM_PY -m pip install --no-deps -e /workspace/robocasa365
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
$SIM_PY -m robocasa.scripts.setup_macros
# Parallel, resumable, CRC-checked asset fetch instead of RoboCasa's one-at-a-time loop.
$SIM_PY "$REPO_ROOT/scripts/download_robocasa365_assets.py" --root /workspace/asset-downloads
wait "$pid365"
wait "$pidbase"
$SIM_PY -m pip freeze > /workspace/simulator-requirements.txt
python -m pip freeze > /workspace/deployment-requirements.txt
git -C /workspace/robosuite365 rev-parse HEAD > /workspace/robosuite-revision.txt
touch /workspace/SETUP-DONE
