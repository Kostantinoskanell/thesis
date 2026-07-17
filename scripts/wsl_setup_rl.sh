#!/usr/bin/env bash
# Stage 1 of the WSL2 GPU RL setup: create a venv and install JAX(CUDA)+MuJoCo/MJX,
# then confirm JAX sees the RTX 4060. This is the make-or-break gate for the
# GPU training path (MuJoCo Playground follows in stage 2).
#
# Run:  wsl -d Ubuntu -- bash /mnt/c/Users/hapos/Desktop/thesis/scripts/wsl_setup_rl.sh
set -e
cd ~
# (re)create the venv if it lacks a working pip (a failed run leaves a broken dir)
if [ ! -x nmc-rl/bin/pip ]; then rm -rf nmc-rl; python3 -m venv nmc-rl; fi
source nmc-rl/bin/activate
python -m pip install --upgrade pip -q
echo "installing jax[cuda12] + mujoco + mjx (large, please wait)..."
pip install -q "jax[cuda12]" mujoco mujoco-mjx
python - <<'PY'
import jax, mujoco
print("JAX", jax.__version__)
print("MuJoCo", mujoco.__version__)
print("JAX devices:", jax.devices())
print("GPU_OK" if any(d.platform == "gpu" for d in jax.devices()) else "GPU_MISSING")
PY
echo "STAGE1_DONE"
