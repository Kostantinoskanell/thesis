#!/usr/bin/env bash
# Stage 2 of the WSL2 GPU RL setup: install MuJoCo Playground (+ brax) and
# smoke-test the Go2 locomotion env (name, obs/action dims). Prereq: stage 1.
#
# Run:  wsl -d Ubuntu -- bash /mnt/c/Users/hapos/Desktop/thesis/scripts/wsl_setup_rl2.sh
set -e
cd ~
source nmc-rl/bin/activate
echo "installing mujoco_playground + brax (large)..."
pip install -q mujoco_playground brax || pip install -q playground brax
python - <<'PY'
from mujoco_playground import registry
try:
    all_envs = list(registry.ALL_ENVS)
except Exception:
    all_envs = list(registry.manipulation.ALL_ENVS) + list(registry.locomotion.ALL_ENVS)
go2 = [n for n in all_envs if 'go2' in n.lower()]
print("Go2 envs:", go2)
name = "Go2JoystickFlatTerrain" if "Go2JoystickFlatTerrain" in go2 else go2[0]
env = registry.load(name)
print("loaded:", name)
print("obs_size:", env.observation_size)
print("action_size:", env.action_size)
PY
echo "STAGE2_DONE"
