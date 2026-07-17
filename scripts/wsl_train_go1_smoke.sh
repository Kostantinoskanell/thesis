#!/usr/bin/env bash
# Smoke test the Go1 PPO training loop on the GPU (short run, reduced envs) to
# confirm it runs end-to-end without OOM before a full training.
cd ~
source nmc-rl2/bin/activate
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # allocate on demand (laptop GPU drives display)
python /mnt/c/Users/hapos/Desktop/thesis/scripts/rl/train_go1.py \
  --timesteps 3000000 --num-envs 1024 \
  --out /mnt/c/Users/hapos/Desktop/thesis/assets/go1_policy_smoke.params
