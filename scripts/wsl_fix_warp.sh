#!/usr/bin/env bash
# One bounded attempt to fix the Playground/MJX version conflict: install the
# MuJoCo-Warp backend the installed mujoco-mjx 3.10 expects, then retest env load.
cd ~
source nmc-rl/bin/activate
echo "installing mujoco-warp + warp-lang..."
pip install -q mujoco-warp warp-lang 2>&1 | tail -4 || \
  pip install -q "git+https://github.com/google-deepmind/mujoco_warp.git" warp-lang 2>&1 | tail -4
python - <<'PY'
try:
    from mujoco_playground import registry
    env = registry.load("Go1JoystickFlatTerrain")
    print("LOAD_OK  obs:", env.observation_size, " act:", env.action_size)
except Exception as e:
    import traceback; traceback.print_exc()
    print("LOAD_FAIL:", repr(e))
PY
