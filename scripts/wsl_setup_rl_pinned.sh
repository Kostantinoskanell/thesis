#!/usr/bin/env bash
# Fresh venv with a pinned PRE-WARP stack (Playground 0.0.5 is the last release
# before warp-lang became required). Test that the Go1 env loads (classic MJX).
# CPU-first: isolates the version fix from CUDA; GPU jax added after this passes.
set -e
cd ~
rm -rf nmc-rl2
python3 -m venv nmc-rl2
source nmc-rl2/bin/activate
python -m pip install -q --upgrade pip
echo "installing pinned pre-warp stack (playground 0.0.5, mujoco 3.3.5)..."
pip install -q "playground==0.0.5" "mujoco==3.3.5" "mujoco-mjx==3.3.5"
python - <<'PY'
import jax, mujoco
print("jax", jax.__version__, "| mujoco", mujoco.__version__)
from mujoco_playground import registry
names = [n for n in registry.ALL_ENVS if "go1" in n.lower() or "go2" in n.lower()]
print("quadruped envs:", names)
env = registry.load(names[0])
print("LOAD_OK", names[0], "obs", env.observation_size, "act", env.action_size)
PY
echo "PINNED_LOAD_DONE"
