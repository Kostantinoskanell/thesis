"""M3: surrogate-gradient pretraining of the LIF-SNN to match the MLP pre-shift.

Trains the LIF-SNN (snnTorch, fast-sigmoid surrogate) by BPTT on the same DAgger
dataset the MLP used (data/imitation_go2_dagger.npz), with the same class-weighted
cross-entropy -- but on **population-vote logits** (one neuron population per
discrete action, proposal Sec. 3.2). This establishes the initial weights that
M4 later "releases" to online R-STDP.

Observations are encoded to spikes with encode_nav_obs (rate lidar + TTFS urgency
+ on/off-split goal/heading/omega). Encoding is done batched + vectorized here
(_encode_batch) because per-sample Python encoding is far too slow for CPU BPTT;
the layout matches encode_nav_obs exactly so train and eval agree.

torch on Windows is CPU-only here, so the encoding window T is kept modest.

Run:  conda run -n nmc python scripts/train_snn_go2.py --seed 0
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

# Cap intra-op threads so several seed trainings can run in parallel without
# thrashing (set by scripts/train_snn_seeds_parallel.sh); harmless if unset.
_nt = os.environ.get("NMC_TORCH_THREADS")
if _nt:
    torch.set_num_threads(int(_nt))

from nmc.controllers.snn import LIFNet
from nmc.encoding.spike_encoding import encode_nav_obs_dim

ACTION_NAMES = ["fwd", "left", "right", "brake"]
N_LIDAR = 32
MAX_RATE_P = 200.0 * 1.0 / 1000.0   # rate_encode default: max_rate_hz * dt_ms / 1000
_ARENA_HALF_M, _V_MAX, _OMEGA_MAX = 5.0, 0.8, 1.5


def _encode_batch(obs_batch: np.ndarray, T: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorized batch encoder matching encode_nav_obs exactly. (B,37)->(T,B,F)."""
    B = obs_batch.shape[0]
    lidar = obs_batch[:, :N_LIDAR]
    goal_dx, goal_dy, heading, v, omega = (obs_batch[:, N_LIDAR + i] for i in range(5))

    nearness = np.clip(1.0 - lidar, 0.0, 1.0)                       # (B, nl)
    p_dist = nearness * MAX_RATE_P
    rate_dist = (rng.random((T, B, N_LIDAR)) < p_dist[None]).astype(np.float32)

    ttfs = np.zeros((T, B, N_LIDAR), dtype=np.float32)
    latency = np.round((1.0 - nearness) * (T - 1)).astype(int)
    bi, fi = np.nonzero(nearness > 0.0)
    ttfs[latency[bi, fi], bi, fi] = 1.0

    def on(x): return np.maximum(x, 0.0)
    def off(x): return np.maximum(-x, 0.0)
    gx = np.clip(goal_dx / _ARENA_HALF_M, -1, 1)
    gy = np.clip(goal_dy / _ARENA_HALF_M, -1, 1)
    h = np.clip(heading / np.pi, -1, 1)
    w = np.clip(omega / _OMEGA_MAX, -1, 1)
    vn = np.clip(v / _V_MAX, 0.0, 1.0)
    scal = np.stack([on(gx), off(gx), on(gy), off(gy), on(h), off(h), vn, on(w), off(w)], 1)
    p_scal = scal * MAX_RATE_P
    rate_scal = (rng.random((T, B, scal.shape[1])) < p_scal[None]).astype(np.float32)

    return np.concatenate([rate_dist, ttfs, rate_scal], axis=2)     # (T,B,F)


def logits_from_net(net, x_seq):
    out_sum, _ = net(x_seq)                                          # (B, n_pops*pop)
    return out_sum.view(-1, net.n_pops, net.pop_size).sum(-1)        # (B, n_pops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/imitation_go2_dagger.npz")
    ap.add_argument("--out", default="assets/snn_seeds/snn_seed0.pt")
    ap.add_argument("--figdir", default="archive/M3_snn_pretrain_go2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tsteps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--neuron", choices=["lif", "alif"], default="alif",
                    help="alif = adaptive-LIF (D2 SOTA upgrade, default); lif = vanilla")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all (probe with e.g. 4000)")
    ap.add_argument("--probe", action="store_true", help="time 1 epoch on a subset and exit")
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
    in_dim = encode_nav_obs_dim(N_LIDAR)
    print(f"dataset {n} steps | in_dim {in_dim} | T {args.tsteps} | class weights {w.numpy().round(2)}",
          flush=True)

    net = LIFNet(in_dim=in_dim, hidden=(512, 512), n_pops=4, pop_size=16,
                 neuron=args.neuron)
    print(f"neuron model: {args.neuron.upper()}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def run_epoch(train: bool):
        net.train(train)
        Xs, Ys = (Xtr, Ytr) if train else (Xva, Yva)
        order = rng.permutation(len(Xs)) if train else np.arange(len(Xs))
        tot_loss, correct, seen = 0.0, 0, 0
        for k in range(0, len(Xs), args.batch):
            b = order[k:k + args.batch]
            raster = _encode_batch(Xs[b], args.tsteps, rng)          # (T,B,F)
            x_seq = torch.from_numpy(raster)
            yb = torch.from_numpy(Ys[b])
            if train:
                logits = logits_from_net(net, x_seq)
                loss = F.cross_entropy(logits, yb, weight=w)
                opt.zero_grad(); loss.backward(); opt.step()
            else:
                with torch.no_grad():
                    logits = logits_from_net(net, x_seq)
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
    torch.save({"state_dict": net.state_dict(), "in_dim": in_dim,
                "hidden": [512, 512], "n_pops": 4, "pop_size": 16,
                "tsteps": args.tsteps, "neuron": args.neuron}, out)
    print(f"saved -> {args.out}  (final val_acc {hist['val_acc'][-1]:.3f})", flush=True)

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
    a1.set_title(f"M3 LIF-SNN surrogate-gradient pretraining (seed {args.seed})")
    fig.tight_layout()
    fig.savefig(figdir / f"fig_snn_training_seed{args.seed}.png", bbox_inches="tight", dpi=150)
    print(f"wrote {figdir / f'fig_snn_training_seed{args.seed}.png'}")


if __name__ == "__main__":
    main()
