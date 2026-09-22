"""D4: retrain the nav-layer SNN with the PopSAN-style learnable Gaussian
population encoder instead of stochastic rate/TTFS coding (snn_popenc.py).

Mirrors scripts/train_snn_go2.py (M3's recipe) exactly -- same DAgger dataset,
same class-weighted cross-entropy on population-vote logits, same 512x512
ALIF hidden stack, same train/val split -- so the ONLY thing that differs
between this checkpoint and M3's is the encoder. That isolates the encoder's
effect on H4 in the H4/M5 re-tests, rather than confounding it with other
architecture changes.

Run:  conda run -n nmc python scripts/train_snn_popenc_go2.py --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

_nt = os.environ.get("NMC_TORCH_THREADS")
if _nt:
    torch.set_num_threads(int(_nt))

from nmc.controllers.snn_popenc import PopEncNavNet, normalize_nav_obs


def logits_from_net(net, obs_norm, T):
    out_sum, _ = net(obs_norm, T)
    return out_sum.view(-1, net.n_pops, net.pop_size).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/imitation_go2_dagger.npz")
    ap.add_argument("--out", default="assets/snn_popenc_seeds/snn_popenc_seed0.pt")
    ap.add_argument("--figdir", default="archive/D4_popenc_snn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tsteps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--in-pop", type=int, default=10)
    ap.add_argument("--neuron", choices=["lif", "alif"], default="alif")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    z = np.load(ROOT / args.data)
    X, Y = z["obs"].astype(np.float32), z["action"].astype(np.int64)
    if args.max_samples and args.max_samples < len(X):
        idx = rng.choice(len(X), args.max_samples, replace=False)
        X, Y = X[idx], Y[idx]
    n = len(X)
    n_val = n // 10
    perm = rng.permutation(n)
    val_i, tr_i = perm[:n_val], perm[n_val:]
    Xtr, Ytr, Xva, Yva = X[tr_i], Y[tr_i], X[val_i], Y[val_i]

    counts = np.bincount(Y, minlength=4)
    w = torch.as_tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
    w = w / w.mean()
    obs_dim = X.shape[1]
    print(f"dataset {n} steps | obs_dim {obs_dim} | in_pop {args.in_pop} | T {args.tsteps} "
          f"| class weights {w.numpy().round(2)}", flush=True)

    net = PopEncNavNet(obs_dim=obs_dim, hidden=(512, 512), n_pops=4, pop_size=16,
                       in_pop=args.in_pop, neuron=args.neuron)
    print(f"neuron model: {args.neuron.upper()} + learnable Gaussian population encoder", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def run_epoch(train: bool):
        net.train(train)
        Xs, Ys = (Xtr, Ytr) if train else (Xva, Yva)
        order = rng.permutation(len(Xs)) if train else np.arange(len(Xs))
        tot_loss, correct, seen = 0.0, 0, 0
        for k in range(0, len(Xs), args.batch):
            b = order[k:k + args.batch]
            obs_norm = normalize_nav_obs(torch.from_numpy(Xs[b]))
            yb = torch.from_numpy(Ys[b])
            if train:
                logits = logits_from_net(net, obs_norm, args.tsteps)
                loss = F.cross_entropy(logits, yb, weight=w)
                opt.zero_grad(); loss.backward(); opt.step()
            else:
                with torch.no_grad():
                    logits = logits_from_net(net, obs_norm, args.tsteps)
                    loss = F.cross_entropy(logits, yb, weight=w)
            tot_loss += float(loss) * len(b)
            correct += int((logits.argmax(-1) == yb).sum()); seen += len(b)
        return tot_loss / seen, correct / seen

    if args.probe:
        t0 = time.time()
        loss, acc = run_epoch(train=True)
        print(f"PROBE: 1 train epoch on {len(Xtr)} samples took {time.time()-t0:.1f}s "
              f"(loss {loss:.3f}, acc {acc:.3f})", flush=True)
        return

    hist = {"epoch": [], "train_loss": [], "val_acc": []}
    t0 = time.time()
    for ep in range(args.epochs):
        tl, _ = run_epoch(train=True)
        _, va = run_epoch(train=False)
        hist["epoch"].append(ep); hist["train_loss"].append(tl); hist["val_acc"].append(va)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"epoch {ep:3d}: loss {tl:.4f}  val_acc {va:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "obs_dim": obs_dim, "hidden": [512, 512],
                "n_pops": 4, "pop_size": 16, "in_pop": args.in_pop, "tsteps": args.tsteps,
                "neuron": args.neuron}, out)
    print(f"saved -> {out}  (final val_acc {hist['val_acc'][-1]:.3f})", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = ROOT / args.figdir
    figdir.mkdir(parents=True, exist_ok=True)
    fig, a1 = plt.subplots(figsize=(6, 4))
    a1.plot(hist["epoch"], hist["train_loss"], color="#1950a0", label="train loss")
    a1.set_xlabel("epoch"); a1.set_ylabel("weighted CE loss", color="#1950a0")
    ax2 = a1.twinx()
    ax2.plot(hist["epoch"], hist["val_acc"], color="#1f9d3a", label="val acc")
    ax2.set_ylabel("val accuracy", color="#1f9d3a"); ax2.set_ylim(0, 1)
    a1.set_title(f"D4 PopEnc-SNN surrogate-gradient pretraining (seed {args.seed})")
    fig.tight_layout()
    fig.savefig(figdir / f"fig_popenc_training_seed{args.seed}.png", bbox_inches="tight", dpi=150)
    print(f"wrote {figdir / f'fig_popenc_training_seed{args.seed}.png'}")


if __name__ == "__main__":
    main()
