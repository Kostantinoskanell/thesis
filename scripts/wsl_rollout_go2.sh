#!/usr/bin/env bash
cd ~
source nmc-rl2/bin/activate
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python /mnt/c/Users/hapos/Desktop/thesis/scripts/rl/rollout_go2.py --vx 1.0 --steps 500
