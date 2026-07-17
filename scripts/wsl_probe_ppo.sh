#!/usr/bin/env bash
# Probe the brax PPO + Playground training API (versions, signatures, Go1 config)
# so the training script matches the installed stack.
cd ~
source nmc-rl/bin/activate
python - <<'PY'
import brax, inspect
print("brax:", brax.__version__)
from brax.training.agents.ppo import train as ppo
sig = list(inspect.signature(ppo.train).parameters)
print("ppo.train params:", sig[:25])
try:
    from mujoco_playground.config import locomotion_params
    cfg = locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")
    print("ppo_config keys:", list(cfg.keys()))
    for k in ("num_timesteps", "num_envs", "batch_size", "num_minibatches",
              "num_evals", "episode_length", "unroll_length"):
        if k in cfg:
            print(f"  {k} = {cfg[k]}")
except Exception as e:
    print("locomotion_params error:", repr(e))
from mujoco_playground import registry
env = registry.load("Go1JoystickFlatTerrain")
print("Go1 obs_size:", env.observation_size, "action_size:", env.action_size)
PY
