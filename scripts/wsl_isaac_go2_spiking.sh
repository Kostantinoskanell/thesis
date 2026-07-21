#!/usr/bin/env bash
# Train the PopSAN spiking Go2 policy in Isaac Lab (L-track L3).
# Same env plumbing as wsl_isaac_go2_train.sh; runs the patched train_spiking.py
# which injects nmc.locomotion.rsl_rl_spiking:SpikingActorMLPModel as the actor.
# Reads NUM_ENVS / MAX_ITER / RUN_TAG from the environment.
set -u

NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITER="${MAX_ITER:-15}"
RUN_TAG="${RUN_TAG:-spiking_smoke}"

ISAAC_ENV=/home/hapos/miniconda3/envs/isaac
PYBIN="$ISAAC_ENV/bin/python"
ISAAC_PKG="$ISAAC_ENV/lib/python3.10/site-packages/isaacsim"
LAB=/home/hapos/IsaacLab
LOG="/home/hapos/go2_${RUN_TAG}.log"
: > "$LOG"

LDADD=""
while IFS= read -r d; do LDADD="$LDADD$d:"; done < <(find "$ISAAC_PKG" -type d -name bin 2>/dev/null)
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LDADD}${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES

cd "$LAB"
echo "=== spiking Go2 train: NUM_ENVS=$NUM_ENVS MAX_ITER=$MAX_ITER ===" >> "$LOG"
"$PYBIN" scripts/reinforcement_learning/rsl_rl/train_spiking.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless \
  --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITER" \
  >> "$LOG" 2>&1
echo "TRAIN_EXIT=$?" >> "$LOG"
