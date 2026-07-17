#!/usr/bin/env bash
# Re-pin to a coherent early-2025 snapshot: jax<0.11 (still has device_put_replicated
# that brax uses) + contemporary brax, keeping the pre-Warp mujoco/playground.
set -e
cd ~
source nmc-rl2/bin/activate
# WORKING coherent stack (hard-won — see debug-log playground-mjx-warp-conflict):
#   pre-Warp playground 0.0.5 + mjx 3.4.0 (has data._impl, no warp) + brax 0.12.1
#   + jax 0.4.38 (has device_put_replicated that brax needs).
echo "re-pinning WORKING stack (jax 0.4.38, mjx 3.4.0) ..."
pip install -q "playground==0.0.5" "mujoco==3.4.0" "mujoco-mjx==3.4.0" \
  "brax==0.12.1" "jax[cuda12]==0.4.38"
python - <<'PY'
import jax, brax, mujoco
print("jax", jax.__version__, "| brax", brax.__version__, "| mujoco", mujoco.__version__)
print("has device_put_replicated:", hasattr(jax, "device_put_replicated"))
print("devices:", jax.devices())
from mujoco_playground import registry
registry.load("Go1JoystickFlatTerrain")
print("LOAD_OK")
PY
echo "REPIN_DONE"
