#!/usr/bin/env bash
# Inspect the Playground 0.0.5 brax PPO config for Go1 so the training script
# matches the exact API (keys, defaults to override for an 8 GB GPU).
cd ~
source nmc-rl2/bin/activate
python - <<'PY'
from mujoco_playground.config import locomotion_params
cfg = locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")
print("type:", type(cfg).__name__)
for k in cfg.keys():
    v = cfg[k]
    if hasattr(v, "keys"):
        print(f"  {k}: (nested) {list(v.keys())}")
    else:
        print(f"  {k} = {v}")
PY
