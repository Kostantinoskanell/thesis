"""Train a Go1 velocity-tracking policy in MuJoCo Playground (brax PPO, GPU).

Runs in WSL2 (nmc-rl2 venv, pinned pre-Warp stack). Saves the trained params +
a reward curve. The policy is a small MLP with observation normalization; it is
later exported and its forward pass run in the Windows MuJoCo loop (JAX only here).

Usage (smoke):  python train_go1.py --timesteps 3000000 --num-envs 2048
Usage (full):   python train_go1.py --num-envs 2048
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Go1JoystickFlatTerrain")
    ap.add_argument("--timesteps", type=int, default=None, help="override num_timesteps")
    ap.add_argument("--num-envs", type=int, default=2048, help="reduce for 8GB GPU")
    ap.add_argument("--out", default="/mnt/c/Users/hapos/Desktop/thesis/assets/go1_policy.params")
    args = ap.parse_args()

    env = registry.load(args.env)
    eval_env = registry.load(args.env)
    cfg = locomotion_params.brax_ppo_config(args.env)
    cfg.num_envs = args.num_envs
    if args.timesteps is not None:
        cfg.num_timesteps = args.timesteps

    net = cfg.network_factory
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(net.policy_hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(net.value_hidden_layer_sizes),
        policy_obs_key=net.policy_obs_key,
        value_obs_key=net.value_obs_key,
    )
    train_kwargs = {k: v for k, v in cfg.items() if k != "network_factory"}

    steps, rewards = [], []
    t0 = time.time()

    def progress(step, metrics):
        r = float(metrics.get("eval/episode_reward", float("nan")))
        steps.append(int(step)); rewards.append(r)
        print(f"[{time.time() - t0:6.0f}s] step {step:>11}: eval_reward {r:8.3f}", flush=True)

    print(f"training {args.env}: num_envs={cfg.num_envs} timesteps={cfg.num_timesteps}", flush=True)
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress,
        seed=0,
        **train_kwargs,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_params(str(out), params)
    json.dump({"steps": steps, "reward": rewards, "env": args.env},
              open(str(out) + ".curve.json", "w"))
    print(f"SAVED {out}  (final eval_reward {rewards[-1]:.3f})", flush=True)


if __name__ == "__main__":
    main()
