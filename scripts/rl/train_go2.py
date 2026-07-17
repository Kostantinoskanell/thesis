"""Train a Go2 velocity-tracking policy in MuJoCo Playground (brax PPO, GPU).

Runs in WSL2 (nmc-rl2 venv, pinned pre-Warp stack). Same shape as train_go1.py,
but for our ported "Go2JoystickFlatTerrain" env (mujoco_playground has no
built-in Go2 -- see src/nmc/rl/envs/go2/). Because Playground's own
`locomotion_params.brax_ppo_config` doesn't know this env name, we build the
PPO config by hand here, copied from Go1JoystickFlatTerrain's tuned config
(same morphology family, closest available reference point).

Usage (smoke):  python train_go2.py --timesteps 3000000 --num-envs 1024
Usage (full):   python train_go2.py --num-envs 1024
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import nmc.rl.envs.go2  # noqa: F401  (registers Go2JoystickFlatTerrain)

from ml_collections import config_dict
from mujoco_playground import registry, wrapper
from mujoco_playground._src import locomotion
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model


def go2_ppo_config(env_name: str) -> config_dict.ConfigDict:
    """PPO config for Go2JoystickFlatTerrain, ported from Go1's tuned config
    in mujoco_playground.config.locomotion_params (Go1JoystickFlatTerrain
    branch) since that function doesn't recognize our env name."""
    env_config = locomotion.get_default_config(env_name)
    return config_dict.create(
        num_timesteps=200_000_000,
        num_evals=10,
        reward_scaling=1.0,
        episode_length=env_config.episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=8192,
        batch_size=256,
        max_grad_norm=1.0,
        num_resets_per_eval=1,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Go2JoystickFlatTerrain")
    ap.add_argument("--timesteps", type=int, default=None, help="override num_timesteps")
    ap.add_argument("--num-envs", type=int, default=1024, help="reduce for 8GB GPU")
    ap.add_argument("--out", default="/mnt/c/Users/hapos/Desktop/thesis/assets/go2_policy.params")
    args = ap.parse_args()

    env = registry.load(args.env)
    eval_env = registry.load(args.env)
    cfg = go2_ppo_config(args.env)
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    steps, rewards = [], []
    t0 = time.time()

    def progress(step, metrics):
        r = float(metrics.get("eval/episode_reward", float("nan")))
        steps.append(int(step)); rewards.append(r)
        print(f"[{time.time() - t0:6.0f}s] step {step:>11}: eval_reward {r:8.3f}", flush=True)
        json.dump({"steps": steps, "reward": rewards, "env": args.env},
                  open(str(out) + ".curve.json", "w"))

    def save_ckpt(step, make_policy, params):
        model.save_params(str(out), params)

    print(f"training {args.env}: num_envs={cfg.num_envs} timesteps={cfg.num_timesteps}", flush=True)
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress,
        policy_params_fn=save_ckpt,
        seed=0,
        **train_kwargs,
    )

    model.save_params(str(out), params)
    print(f"SAVED {out}  (final eval_reward {rewards[-1]:.3f})", flush=True)


if __name__ == "__main__":
    main()
