#!/usr/bin/env bash
# Find the mjx sweet spot: has data._impl (playground 0.0.5 needs it) AND doesn't
# require the Warp backend AND works with jax 0.4.38 (brax needs device_put_replicated).
# Test a REAL env.step, not just load.
set -e
cd ~
source nmc-rl2/bin/activate
echo "pinning mujoco/mjx 3.4.0 (jax held at 0.4.38)..."
pip install -q "mujoco==3.4.0" "mujoco-mjx==3.4.0" "jax[cuda12]==0.4.38"
python - <<'PY'
import jax, jax.numpy as jp
print("jax", jax.__version__, "| has device_put_replicated:", hasattr(jax, "device_put_replicated"))
from mujoco_playground import registry
env = registry.load("Go1JoystickFlatTerrain")
state = env.reset(jax.random.PRNGKey(0))
state = env.step(state, jp.zeros(env.action_size))
print("STEP_OK reward:", float(state.reward))
PY
echo "MJX_STEP_TEST_DONE"
