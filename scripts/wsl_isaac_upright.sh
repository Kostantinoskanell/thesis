#!/usr/bin/env bash
# Retrain the Go2 MLP *teacher* to walk UPRIGHT (base-height reward), so the distilled
# spiking student can stand tall instead of crouching at 0.18 m. Same env plumbing as
# the other Isaac wrappers. Generates train_upright.py then runs it with anti-crouch on.
#
# Usage: NUM_ENVS=2048 MAX_ITER=1200 TARGET_H=0.30 HEIGHT_W=-10.0 MIN_H=0.18 \
#        bash scripts/wsl_isaac_upright.sh
set -u

NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITER="${MAX_ITER:-1200}"
REPO=/mnt/c/Users/hapos/Desktop/thesis
ISAAC_ENV=/home/hapos/miniconda3/envs/isaac
PYBIN="$ISAAC_ENV/bin/python"
ISAAC_PKG="$ISAAC_ENV/lib/python3.10/site-packages/isaacsim"
LAB=/home/hapos/IsaacLab
LOG="${UPRIGHT_LOG:-/home/hapos/go2_upright.log}"
: > "$LOG"

LDADD=""
while IFS= read -r d; do LDADD="$LDADD$d:"; done < <(find "$ISAAC_PKG" -type d -name bin 2>/dev/null)
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LDADD}${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES

# (1) regenerate the patched trainer
"$PYBIN" "$REPO/scripts/make_isaac_train_upright.py" >> "$LOG" 2>&1

# (2) train upright
export UPRIGHT_ANTICROUCH=1
export UPRIGHT_TARGET_HEIGHT="${TARGET_H:-0.30}"
export UPRIGHT_HEIGHT_WEIGHT="${HEIGHT_W:--10.0}"
export UPRIGHT_MIN_HEIGHT="${MIN_H:-0.18}"
export UPRIGHT_FEET_SLIDE="${FEET_SLIDE:-0}"
export UPRIGHT_FEET_SLIDE_WEIGHT="${FEET_SLIDE_W:--1.0}"

# optional: resume/fine-tune from an existing run instead of training from scratch
# (e.g. RESUME_RUN=2026-07-23_12-11-56 RESUME_CKPT=model_1199.pt to add feet_slide
# on top of the already-upright teacher, far fewer iterations needed)
RESUME_ARGS=()
if [ -n "${RESUME_RUN:-}" ]; then
  RESUME_ARGS=(agent.resume=True "agent.load_run=${RESUME_RUN}" "agent.load_checkpoint=${RESUME_CKPT:-model_.*.pt}")
fi

cd "$LAB"
echo "=== Go2 UPRIGHT train: NUM_ENVS=$NUM_ENVS MAX_ITER=$MAX_ITER target=$UPRIGHT_TARGET_HEIGHT w=$UPRIGHT_HEIGHT_WEIGHT feet_slide=$UPRIGHT_FEET_SLIDE resume=${RESUME_RUN:-none} ===" >> "$LOG"
"$PYBIN" scripts/reinforcement_learning/rsl_rl/train_upright.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless \
  --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITER" \
  "${RESUME_ARGS[@]}" \
  >> "$LOG" 2>&1
echo "UPRIGHT_TRAIN_EXIT=$?" >> "$LOG"
# surface the run dir for the caller
ls -td "$LAB"/logs/rsl_rl/unitree_go2_flat/*/ 2>/dev/null | head -1 >> "$LOG"
