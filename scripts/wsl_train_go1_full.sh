#!/usr/bin/env bash
# Full Go1 PPO training. Run as a HARNESS background task (run_in_background=true)
# so the WSL session stays alive; train_go1.py checkpoints every eval, so a
# partial policy survives even if interrupted.
# Working stack: playground 0.0.5 + mjx 3.4.0 + brax 0.12.1 + jax 0.4.38 (WSL2 GPU).
cd ~
source nmc-rl2/bin/activate
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python /mnt/c/Users/hapos/Desktop/thesis/scripts/rl/train_go1.py \
  --num-envs 1024 \
  --out /mnt/c/Users/hapos/Desktop/thesis/assets/go1_policy.params
