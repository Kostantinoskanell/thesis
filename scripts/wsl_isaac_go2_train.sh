#!/usr/bin/env bash
# Parameterized Isaac Lab Go2 flat-velocity PPO training (WSL2, isaac env).
# Same env-plumbing fixes as the smoke script (see debug-log 2026-07-21). Reads
# NUM_ENVS, MAX_ITER, RUN_TAG from the environment so one script serves both the
# VRAM-ceiling probe and the full baseline run.
#
# Usage:  NUM_ENVS=2048 MAX_ITER=20 RUN_TAG=probe bash wsl_isaac_go2_train.sh
set -u

NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITER="${MAX_ITER:-1500}"
RUN_TAG="${RUN_TAG:-run}"

ISAAC_ENV=/home/hapos/miniconda3/envs/isaac
PYBIN="$ISAAC_ENV/bin/python"
ISAAC_PKG="$ISAAC_ENV/lib/python3.10/site-packages/isaacsim"
LAB=/home/hapos/IsaacLab
LOG="/home/hapos/go2_${RUN_TAG}.log"

: > "$LOG"

LDADD=""
while IFS= read -r d; do
  LDADD="$LDADD$d:"
done < <(find "$ISAAC_PKG" -type d -name bin 2>/dev/null)

export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LDADD}${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES

cd "$LAB"
echo "=== Go2 train: NUM_ENVS=$NUM_ENVS MAX_ITER=$MAX_ITER TAG=$RUN_TAG ===" >> "$LOG"
"$PYBIN" scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless \
  --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITER" \
  >> "$LOG" 2>&1
echo "TRAIN_EXIT=$?" >> "$LOG"
