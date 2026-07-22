"""DAgger data collection to make the distilled spiking walker robust to SUSTAINED
commands (L4). The BC-distilled net walks under normal (resampling) commands but
falls ~5s into a held-constant command -- covariate shift: it drifts into states the
teacher's own rollouts never visited, and never learned to recover.

DAgger fix (the M2 nav-layer recipe): let the STUDENT drive the env under HELD
commands (so it visits its own drift/pre-fall states), and label each visited state
with the TEACHER's correct action. Retraining on these teaches recovery in exactly
the regime that was failing. Pure-student rollout (beta=0) -- aggressive, targets the
failure states directly.

Commands are held constant per-env (resampled on episode reset), forward-biased, to
reproduce the sustained-command regime. Both student and teacher read the SAME
command-overridden obs.

Run (isaac env):
  python scripts/l4_dagger_collect.py --num-envs 256 --steps 500 \
     --student data/l4_distilled_spiking.pt --mlp-ckpt .../model_1499.pt \
     --out data/l4_dagger_data.npz
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/mnt/c/Users/hapos/Desktop/thesis/src")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--student", required=True)
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
from nmc.locomotion.popsan_actor import PopSpikingActorNet

TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
device = "cuda:0"


def load_teacher(ckpt):
    mlp = nn.Sequential(nn.Linear(48, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(),
                        nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 12)).to(device)
    asd = torch.load(ckpt, map_location=device, weights_only=False)["actor_state_dict"]
    mlp.load_state_dict({k[4:]: v for k, v in asd.items() if k.startswith("mlp.")})
    mlp.eval()
    return mlp


def load_student(path):
    blob = torch.load(path, map_location=device, weights_only=True)
    mean = blob.pop("_obs_mean").to(device)
    std = blob.pop("_obs_std").to(device)
    net = PopSpikingActorNet(obs_dim=48, act_dim=12, hidden=(128, 128, 128),
                             in_pop=10, out_pop=10, T=8, decoder_tanh=False).to(device)
    net.load_state_dict(blob)
    net.eval()
    return net, mean, std


def sample_cmds(n):
    """Held, forward-biased velocity commands: vx in [0.2,0.6], small vy, small yaw."""
    vx = torch.empty(n, device=device).uniform_(0.2, 0.6)
    vy = torch.empty(n, device=device).uniform_(-0.2, 0.2)
    yaw = torch.empty(n, device=device).uniform_(-0.3, 0.3)
    return torch.stack([vx, vy, yaw], dim=1)


def main():
    env_cfg = parse_env_cfg(TASK, device=device, num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg)
    teacher = load_teacher(args_cli.mlp_ckpt)
    student, mean, std = load_student(args_cli.student)

    obs_dict, _ = env.reset()
    cmds = sample_cmds(args_cli.num_envs)
    obs_buf, act_buf = [], []
    for t in range(args_cli.steps):
        raw = obs_dict["policy"].clone()
        raw[:, 9:12] = cmds                                   # hold the command
        with torch.no_grad():
            student_in = (raw - mean) / (std + 1e-2)
            student_action = student(student_in)               # STUDENT drives
            teacher_label = teacher(raw)                       # TEACHER labels the state
        obs_buf.append(raw.cpu().numpy().copy())
        act_buf.append(teacher_label.cpu().numpy().copy())
        obs_dict, _, terminated, truncated, _ = env.step(student_action)
        done = (terminated | truncated).nonzero(as_tuple=False).flatten()
        if done.numel() > 0:                                   # resample held cmd for reset envs
            cmds[done] = sample_cmds(done.numel())

    env.close()
    obs = np.concatenate(obs_buf, axis=0)
    act = np.concatenate(act_buf, axis=0)
    np.savez(args_cli.out, obs=obs, act=act, obs_mean=obs.mean(0), obs_std=obs.std(0))
    print(f"[dagger] saved {obs.shape[0]} (student-state, teacher-action) pairs -> {args_cli.out}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
