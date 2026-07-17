#!/usr/bin/env bash
# List available versions of the RL stack so we can pin a coherent pre-Warp combo.
cd ~
source nmc-rl/bin/activate
for pkg in playground mujoco mujoco-mjx brax jax; do
  echo "=== $pkg ==="
  pip install "$pkg==" 2>&1 | grep -io "from versions:.*" | head -1
done
echo "=== playground 0.2.0 mujoco requirement ==="
pip show playground 2>/dev/null | grep -i requires
