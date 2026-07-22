"""Behavior-clone the walking MLP into the PopSAN spiking actor (L4 distillation).

The thesis's core recipe (M3, nav layer) applied to locomotion: the spiking policy
can't discover walking via PPO (stationary local optima, D14), so we distill the
walking MLP teacher into it via supervised BC. Trains the bare PopSpikingActorNet
to map standardized obs -> the MLP's action, using surrogate-gradient BPTT (the net's
own rectangular surrogate). Runs in the `nmc` env (pure torch, no Isaac needed).

Saves the distilled net in the format the eval script (l4_rstdp_terrain.load_actor
--load-weights) reads: the net state_dict + _obs_mean/_obs_std for standardization.

Run:  conda run -n nmc python scripts/l4_distill_spiking.py --data data/l4_distill_data.npz \
          --out data/l4_distilled_spiking.pt --epochs 60
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

import numpy as np
import torch
import torch.nn as nn

from nmc.locomotion.popsan_actor import PopSpikingActorNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="comma-separated npz files to aggregate (DAgger)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-weights", default=None, help="warm-start from this distilled .pt (DAgger refine)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", default="128,128,128")
    ap.add_argument("--in-pop", type=int, default=10)
    ap.add_argument("--out-pop", type=int, default=10)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    obs_parts, act_parts = [], []
    for path in args.data.split(","):
        dd = np.load(path)
        obs_parts.append(dd["obs"].astype(np.float32))
        act_parts.append(dd["act"].astype(np.float32))
        print(f"  loaded {obs_parts[-1].shape[0]} pairs from {path}")
    obs = np.concatenate(obs_parts, axis=0)
    act = np.concatenate(act_parts, axis=0)
    mean, std = obs.mean(0).astype(np.float32), obs.std(0).astype(np.float32)
    # MUST match the eval script's normalize(): (obs-mean)/(std+1e-2). Same formula
    # here and at deploy, or the distilled net sees a shifted input distribution.
    obs_n = (obs - mean) / (std + 1e-2)
    print(f"data: {obs.shape[0]} pairs, obs_dim {obs.shape[1]}, act_dim {act.shape[1]}")

    dev = args.device
    X = torch.as_tensor(obs_n, device=dev)
    Y = torch.as_tensor(act, device=dev)
    n = X.shape[0]
    n_val = n // 10
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    hidden = tuple(int(x) for x in args.hidden.split(","))
    net = PopSpikingActorNet(obs_dim=obs.shape[1], act_dim=act.shape[1], hidden=hidden,
                             in_pop=args.in_pop, out_pop=args.out_pop, T=args.T,
                             decoder_tanh=False, actor_lr_scale=1.0).to(dev)
    if args.init_weights:
        blob = torch.load(args.init_weights, map_location=dev, weights_only=True)
        blob.pop("_obs_mean", None); blob.pop("_obs_std", None)
        net.load_state_dict(blob)
        print(f"warm-started from {args.init_weights}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        net.train()
        idx = tr_idx[torch.randperm(tr_idx.shape[0])]
        tot = 0.0
        for i in range(0, idx.shape[0], args.batch):
            b = idx[i:i + args.batch]
            opt.zero_grad()
            pred = net(X[b])
            loss = ((pred - Y[b]) ** 2).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * b.shape[0]
        net.eval()
        with torch.no_grad():
            vpred = net(X[val_idx])
            vloss = float(((vpred - Y[val_idx]) ** 2).mean())
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d}: train_mse {tot/idx.shape[0]:.5f}  val_mse {vloss:.5f}", flush=True)

    blob = dict(net.state_dict())
    blob["_obs_mean"] = torch.as_tensor(mean)
    blob["_obs_std"] = torch.as_tensor(std)
    torch.save(blob, args.out)
    print(f"saved distilled spiking actor -> {args.out}  (final val_mse {vloss:.5f})")


if __name__ == "__main__":
    main()
