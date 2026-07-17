"""M2 (multi-seed): train N frozen MLPs on the aggregated DAgger dataset.

For confidence intervals we need multiple independently-trained controllers.
DAgger data collection (the expensive part) is a fixed asset -- we run it once
(scripts/dagger_go2.py) and here train N models that differ only in init +
SGD order, giving training-seed variance cheaply. (This captures init/optimization
variance, not DAgger-rollout variance -- noted in the archive README.)

Run:  conda run -n nmc python scripts/train_seeds_mlp_go2.py --seeds 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from nmc.controllers.mlp import MLPPolicy


def train_one(X, Y, seed, epochs, batch, lr):
    torch.manual_seed(seed)
    counts = np.bincount(Y, minlength=4)
    w = torch.as_tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
    w = w / w.mean()
    policy = MLPPolicy(obs_dim=X.shape[1], n_actions=4, dropout=0.1)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    for ep in range(epochs):
        policy.train()
        perm = torch.randperm(len(Xt))
        for k in range(0, len(Xt), batch):
            b = perm[k:k + batch]
            logits, _ = policy(Xt[b])
            loss = F.cross_entropy(logits, Yt[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    policy.eval()
    with torch.no_grad():
        acc = float((policy(Xt)[0].argmax(-1) == Yt).float().mean())
    return policy, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--data", default="data/imitation_go2_dagger.npz")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    z = np.load(ROOT / args.data)
    X, Y = z["obs"], z["action"]
    print(f"dataset: {len(X)} steps ({args.data})")

    (ROOT / "assets" / "mlp_seeds").mkdir(parents=True, exist_ok=True)
    for s in range(args.seeds):
        policy, acc = train_one(X, Y, seed=s, epochs=args.epochs, batch=args.batch, lr=args.lr)
        out = ROOT / "assets" / "mlp_seeds" / f"mlp_seed{s}.pt"
        torch.save({"state_dict": policy.state_dict(), "obs_dim": X.shape[1],
                    "n_actions": 4, "hidden": [512, 512]}, out)
        print(f"seed {s}: train_acc {acc:.3f} -> {out.name}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
