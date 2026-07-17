"""M2: behavior-clone the frozen MLP on the dynamic-env A* demos (D3).

Cross-entropy on (obs -> teacher action) from data/imitation_go2.npz, with
inverse-frequency class weights (the teacher's actions are heavily imbalanced
toward forward/brake; unweighted CE risks collapsing to the majority class).
90/10 train/val split at EPISODE granularity is not possible (demos are step-
concatenated), so a random step split is used -- fine for a policy that acts
per-step. Saves weights + a training-curve/per-class-accuracy figure.

Run:  conda run -n nmc python scripts/train_mlp_go2.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Windows conda: torch and matplotlib/MKL each ship an OpenMP runtime; loading
# both aborts the process (OMP Error #15) right at the final figure. Standard
# workaround; harmless for this script's plotting-only matplotlib use.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from nmc.controllers.mlp import MLPPolicy

ACTION_NAMES = ["fwd", "left", "right", "brake"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/imitation_go2.npz")
    ap.add_argument("--out", default="assets/mlp_frozen_go2.pt")
    ap.add_argument("--figdir", default="archive/M2_mlp_baselines_go2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    z = np.load(ROOT / args.data)
    X, Y = z["obs"], z["action"]
    n = len(X)
    idx = rng.permutation(n)
    n_val = n // 10
    val_i, tr_i = idx[:n_val], idx[n_val:]
    Xtr = torch.as_tensor(X[tr_i]); Ytr = torch.as_tensor(Y[tr_i])
    Xva = torch.as_tensor(X[val_i]); Yva = torch.as_tensor(Y[val_i])

    counts = np.bincount(Y, minlength=4)
    w = torch.as_tensor((counts.sum() / np.maximum(counts, 1)), dtype=torch.float32)
    w = w / w.mean()
    print(f"demos: {n} steps | action counts {dict(zip(ACTION_NAMES, counts))}")
    print(f"class weights: {w.numpy().round(2)}")

    # Capacity parity with the LIF-SNN (512x512 + LayerNorm); dropout regularizes BC.
    policy = MLPPolicy(obs_dim=X.shape[1], n_actions=4, dropout=0.1)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    hist = {"epoch": [], "train_loss": [], "val_acc": []}
    for ep in range(args.epochs):
        policy.train()
        perm = torch.randperm(len(Xtr))
        losses = []
        for k in range(0, len(Xtr), args.batch):
            b = perm[k:k + args.batch]
            logits, _ = policy(Xtr[b])
            loss = F.cross_entropy(logits, Ytr[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        policy.eval()
        with torch.no_grad():
            logits, _ = policy(Xva)
            acc = float((logits.argmax(-1) == Yva).float().mean())
        hist["epoch"].append(ep); hist["train_loss"].append(np.mean(losses)); hist["val_acc"].append(acc)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d}: loss {np.mean(losses):.4f}  val_acc {acc:.3f}", flush=True)

    # per-class validation accuracy
    with torch.no_grad():
        pred = policy(Xva)[0].argmax(-1)
    per_class = {}
    for c, name in enumerate(ACTION_NAMES):
        m = Yva == c
        per_class[name] = float((pred[m] == c).float().mean()) if m.any() else float("nan")
    print("per-class val acc:", {k: round(v, 3) for k, v in per_class.items()})

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(),
                "obs_dim": X.shape[1], "n_actions": 4, "hidden": [512, 512]}, out)
    print(f"saved -> {args.out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = ROOT / args.figdir
    figdir.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(hist["epoch"], hist["train_loss"], color="#1950a0")
    a1.set_xlabel("epoch"); a1.set_ylabel("weighted CE loss"); a1.set_title("training loss")
    ax2 = a1.twinx()
    ax2.plot(hist["epoch"], hist["val_acc"], color="#1f9d3a")
    ax2.set_ylabel("val accuracy", color="#1f9d3a")
    a2.bar(per_class.keys(), per_class.values(), color="#1950a0", width=0.6)
    a2.set_ylim(0, 1); a2.set_title("per-class val accuracy")
    fig.suptitle("M2 frozen MLP — imitation on dynamic-env A* demos")
    fig.tight_layout()
    fig.savefig(figdir / "fig_mlp_training.png", bbox_inches="tight", dpi=150)
    print(f"wrote {figdir / 'fig_mlp_training.png'}")


if __name__ == "__main__":
    main()
