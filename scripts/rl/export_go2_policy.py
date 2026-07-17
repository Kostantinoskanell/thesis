"""Export the trained Go2 policy to a plain .npz for the Windows NumPy runtime.

Runs in WSL2 (nmc-rl2). Two outputs:
  * assets/go2_policy_export.npz  -- obs-normalizer (mean/std of the 48-dim 'state'
    obs) + policy MLP weights (48->512->256->128->24; swish; deterministic action
    = tanh of the first 12 outputs).
  * assets/go2_parity_vectors.npz -- N random obs vectors + the actions brax's own
    inference fn produces for them (deterministic). The Windows side must reproduce
    these bit-near-exactly (float32 tolerance) or the export is wrong -- this is the
    train<->deploy parity gate, guarding against silent obs-ordering/normalization
    bugs that would poison every downstream experiment.

Usage:  python export_go2_policy.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import nmc.rl.envs.go2  # noqa: F401  (registers Go2JoystickFlatTerrain)

import numpy as np
import jax
from mujoco_playground import registry
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model

from train_go2 import go2_ppo_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Go2JoystickFlatTerrain")
    ap.add_argument("--params", default="/mnt/c/Users/hapos/Desktop/thesis/assets/go2_policy.params")
    ap.add_argument("--out", default="/mnt/c/Users/hapos/Desktop/thesis/assets/go2_policy_export.npz")
    ap.add_argument("--vectors-out", default="/mnt/c/Users/hapos/Desktop/thesis/assets/go2_parity_vectors.npz")
    ap.add_argument("--n-vectors", type=int, default=64)
    args = ap.parse_args()

    params = model.load_params(args.params)
    normalizer, policy = params[0], params[1]

    export = {
        "obs_mean": np.asarray(normalizer.mean["state"], dtype=np.float32),
        "obs_std": np.asarray(normalizer.std["state"], dtype=np.float32),
    }
    layers = sorted(policy["params"].keys())  # hidden_0..hidden_3
    for i, name in enumerate(layers):
        export[f"w{i}"] = np.asarray(policy["params"][name]["kernel"], dtype=np.float32)
        export[f"b{i}"] = np.asarray(policy["params"][name]["bias"], dtype=np.float32)
    export["n_layers"] = np.array(len(layers))
    np.savez(args.out, **export)
    shapes = {k: v.shape for k, v in export.items() if hasattr(v, "shape")}
    print(f"exported -> {args.out}")
    print(f"  shapes: {shapes}")

    # --- parity vectors from brax's own inference fn (ground truth) ---
    env = registry.load(args.env)
    cfg = go2_ppo_config(args.env)
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
    policy_fn = jax.jit(make_policy(params, deterministic=True))

    rng = np.random.default_rng(0)
    # Realistic obs magnitudes: draw around the normalizer stats so the test
    # exercises the same numeric range flight obs will have.
    obs_batch = (export["obs_mean"][None, :]
                 + export["obs_std"][None, :]
                 * rng.standard_normal((args.n_vectors, 48))).astype(np.float32)

    acts = []
    key = jax.random.PRNGKey(0)
    for i in range(args.n_vectors):
        obs = {"state": obs_batch[i],
               "privileged_state": np.zeros(123, dtype=np.float32)}
        act, _ = policy_fn(obs, key)
        acts.append(np.asarray(act, dtype=np.float32))
    acts = np.stack(acts)
    np.savez(args.vectors_out, obs=obs_batch, actions=acts)
    print(f"parity vectors -> {args.vectors_out}  obs {obs_batch.shape} actions {acts.shape}")
    print(f"  action range: [{acts.min():.4f}, {acts.max():.4f}]")


if __name__ == "__main__":
    main()
