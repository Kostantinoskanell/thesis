"""Roll out the trained Go1 policy and check it tracks a velocity command.

Rebuilds the PPO inference fn from saved params, forces a forward command, and
records commanded-vs-actual forward speed + base height. Saves the qpos
trajectory (.npz) for rendering. Runs in WSL2 (nmc-rl2).

Usage:  python rollout_go1.py --vx 1.0 --steps 500
"""

from __future__ import annotations

import argparse

import numpy as np
import jax
import jax.numpy as jp
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Go1JoystickFlatTerrain")
    ap.add_argument("--params", default="/mnt/c/Users/hapos/Desktop/thesis/assets/go1_policy.params")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--vx", type=float, default=1.0)
    ap.add_argument("--out", default="/mnt/c/Users/hapos/Desktop/thesis/assets/go1_rollout.npz")
    args = ap.parse_args()

    env = registry.load(args.env)
    cfg = locomotion_params.brax_ppo_config(args.env)
    net = cfg.network_factory
    network = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(net.policy_hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(net.value_hidden_layer_sizes),
        policy_obs_key=net.policy_obs_key,
        value_obs_key=net.value_obs_key,
    )
    make_policy = ppo_networks.make_inference_fn(network)
    params = model.load_params(args.params)
    policy = make_policy(params, deterministic=True)

    jit_reset, jit_step, jit_policy = jax.jit(env.reset), jax.jit(env.step), jax.jit(policy)
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)

    def force_cmd(s):
        if "command" in s.info:
            return s.replace(info={**s.info, "command": jp.array([args.vx, 0.0, 0.0])})
        return s

    state = force_cmd(state)
    qpos, act_vx, heights = [], [], []
    for _ in range(args.steps):
        act, _ = jit_policy(state.obs, rng)
        state = jit_step(state, act)
        state = force_cmd(state)
        d = state.data
        q = np.array(d.qpos)
        qpos.append(q)
        vworld = np.array(d.qvel[:3])
        w, x, y, z = q[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        act_vx.append(float(vworld[0] * np.cos(yaw) + vworld[1] * np.sin(yaw)))
        heights.append(float(q[2]))

    qpos = np.array(qpos)
    np.savez(args.out, qpos=qpos, act_vx=np.array(act_vx),
             cmd_vx=np.full(len(act_vx), args.vx), heights=np.array(heights))
    warm = slice(50, None)
    print(f"commanded vx = {args.vx:.2f} m/s")
    print(f"actual   vx = {np.mean(act_vx[50:]):.2f} +/- {np.std(act_vx[50:]):.2f} m/s")
    print(f"height   = {np.mean(heights):.3f} m (min {np.min(heights):.3f})")
    print(f"upright  = {np.min(heights) > 0.15}   |  saved {args.out}")


if __name__ == "__main__":
    main()
