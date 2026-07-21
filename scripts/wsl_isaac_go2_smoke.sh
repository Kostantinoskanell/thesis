#!/usr/bin/env bash
# Isaac Lab Go2 flat-velocity training smoke test (WSL2, isaac conda env).
# Written as a FILE (not inline) to avoid Windows->WSL->bash quoting fragility.
# Sets LD_LIBRARY_PATH from Isaac Sim's own bin dirs so the physx native plugins
# (libhdx.so et al.) resolve when python is launched directly.
set -u

ISAAC_ENV=/home/hapos/miniconda3/envs/isaac
PYBIN="$ISAAC_ENV/bin/python"
ISAAC_PKG="$ISAAC_ENV/lib/python3.10/site-packages/isaacsim"
LAB=/home/hapos/IsaacLab
LOG=/home/hapos/go2_smoke3.log

: > "$LOG"

# Build LD_LIBRARY_PATH from every bin dir shipped in the isaacsim pip package.
LDADD=""
while IFS= read -r d; do
  LDADD="$LDADD$d:"
done < <(find "$ISAAC_PKG" -type d -name bin 2>/dev/null)

{
  echo "=== bin dirs added to LD_LIBRARY_PATH: $(printf '%s' "$LDADD" | tr ':' '\n' | grep -c .) ==="
  echo "=== libhdx.so location: $(find "$ISAAC_PKG" -name libhdx.so 2>/dev/null | head -1) ==="
} >> "$LOG" 2>&1

# /usr/lib/wsl/lib holds the WSL CUDA driver libs (libcuda.so) at a nonstandard
# path PhysX's GPU pipeline can't find on its own -> without this it aborts
# (munmap_chunk / SIGABRT) right at "Starting the simulation".
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LDADD}${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES

cd "$LAB"
echo "=== launching training ===" >> "$LOG"
"$PYBIN" scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless --num_envs 32 --max_iterations 2 \
  >> "$LOG" 2>&1
echo "TRAIN_EXIT=$?" >> "$LOG"
