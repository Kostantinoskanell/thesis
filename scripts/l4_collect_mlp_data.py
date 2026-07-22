"""Collect (obs, action) pairs from the walking MLP baseline for distillation (L4).

The spiking policy won't DISCOVER walking via PPO (it converges to stationary
optima — belly-flop, then stand-still — regardless of reward shaping; D14). But
the MLP baseline genuinely walks (tracks commanded velocity, err 0.026). So we
fall back to the thesis's own recipe (M3): distill the walking teacher into the
spiking net via BC, giving it a walking initialization it could never find alone.

This script rolls out the MLP across many parallel envs (the env samples diverse
random velocity commands each reset, so the dataset covers the command/state
space) and saves raw (obs, action) pairs. The MLP baseline used
obs_normalization=False, so obs are raw and action = mlp(raw_obs) (no tanh).

Run (isaac env, via wsl_isaac_l4.sh-style launcher):
  python scripts/l4_collect_mlp_data.py --num-envs 256 --steps 500 \
      --mlp-ckpt .../model_1499.pt --out data/l4_distill_data.npz
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/mnt/c/Users/hapos/Desktop/thesis/src")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--mlp-ckpt", required=True)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"  # non-Play: keeps command randomization for a diverse dataset
device = "cuda:0"


def load_mlp(ckpt):
    mlp = nn.Sequential(nn.Linear(48, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(),
                        nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 12)).to(device)
    asd = torch.load(ckpt, map_location=device, weights_only=False)["actor_state_dict"]
    mlp.load_state_dict({k[4:]: v for k, v in asd.items() if k.startswith("mlp.")})
    mlp.eval()
    return mlp


def main():
    env_cfg = parse_env_cfg(TASK, device=device, num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg)
    mlp = load_mlp(args_cli.mlp_ckpt)

    obs_dict, _ = env.reset()
    obs_buf, act_buf = [], []
    for t in range(args_cli.steps):
        raw = obs_dict["policy"]
        with torch.no_grad():
            action = mlp(raw)
        obs_buf.append(raw.cpu().numpy().copy())
        act_buf.append(action.cpu().numpy().copy())
        obs_dict, _, _, _, _ = env.step(action)
    env.close()

    obs = np.concatenate(obs_buf, axis=0)   # (steps*num_envs, 48)
    act = np.concatenate(act_buf, axis=0)   # (steps*num_envs, 12)
    np.savez(args_cli.out, obs=obs, act=act,
             obs_mean=obs.mean(0), obs_std=obs.std(0))
    print(f"[collect] saved {obs.shape[0]} (obs,act) pairs -> {args_cli.out}", flush=True)
    print(f"[collect] action range [{act.min():.3f}, {act.max():.3f}], "
          f"obs range [{obs.min():.2f}, {obs.max():.2f}]", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
