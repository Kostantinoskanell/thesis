#!/usr/bin/env bash
# Render the Go1 rollout to a GIF + tracking graph (headless EGL). Installs the
# small render deps into the venv if missing.
set -e
cd ~
source nmc-rl2/bin/activate
python -c "import PIL, matplotlib" 2>/dev/null || pip install -q pillow matplotlib
# EGL needs a GL library; install mesa if the renderer can't find it.
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python /mnt/c/Users/hapos/Desktop/thesis/scripts/rl/render_go1_rollout.py
