#!/usr/bin/env bash
# L4 R-STDP terrain-recovery rollout (WSL2, isaac conda env). Same env plumbing
# as the L3 training scripts. Forwards all args straight to l4_rstdp_terrain.py.
set -u

ISAAC_ENV=/home/hapos/miniconda3/envs/isaac
PYBIN="$ISAAC_ENV/bin/python"
ISAAC_PKG="$ISAAC_ENV/lib/python3.10/site-packages/isaacsim"
LAB=/home/hapos/IsaacLab
LOG="${L4_LOG:-/home/hapos/l4_run.log}"
: > "$LOG"

LDADD=""
while IFS= read -r d; do LDADD="$LDADD$d:"; done < <(find "$ISAAC_PKG" -type d -name bin 2>/dev/null)
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LDADD}${LD_LIBRARY_PATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES

cd "$LAB"
"$PYBIN" /mnt/c/Users/hapos/Desktop/thesis/scripts/l4_rstdp_terrain.py "$@" >> "$LOG" 2>&1
echo "L4_EXIT=$?" >> "$LOG"
