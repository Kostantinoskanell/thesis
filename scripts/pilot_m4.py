"""M4 PILOT -- the go/no-go gate (proposal Phase 2).

Question H1: does releasing the pretrained SNN to online R-STDP recover
performance under the mid-episode distribution shift faster/better than the
frozen SNN and the online-MLP -- WITHOUT destroying pre-shift performance?

Two modes:

  --mode guardrail : the proposal's #1 risk check. Run R-STDP on the BASE
      distribution (no shift) for a sweep of learning rates eta. Pre-shift
      success must stay ~ the frozen SNN (41%); if plasticity collapses it,
      eta is too high. Picks the largest safe eta.

  --mode compare  : the recovery experiment. Real Go2NavEnv WITH the mid-episode
      shift, a block of episodes with CONTINUAL adaptation (plastic weights carry
      across episodes, all controllers start from the M3 pretrained weights).
      Compares R-STDP SNN vs frozen SNN vs online-MLP(TD-lambda): success-rate
      trend over the block + mean post-warmup success with Wilson CIs.

Run:  conda run -n nmc python scripts/pilot_m4.py --mode guardrail
      conda run -n nmc python scripts/pilot_m4.py --mode compare --eta 0.02
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

from nmc.controllers.snn import LIFNet, SNNNavController
from nmc.controllers.mlp import MLPPolicy, OnlineMLP
from nmc.plasticity.stdp import STDPConfig
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from nmc.eval.metrics import wilson_ci95

OUT = ROOT / "archive" / "M4_pilot_go2"
SNN_CKPT = ROOT / "assets" / "snn_seeds" / "snn_seed0.pt"
MLP_CKPT = ROOT / "assets" / "mlp_frozen_go2.pt"


def load_snn_net():
    ckpt = torch.load(SNN_CKPT, weights_only=True)
    net = LIFNet(in_dim=ckpt["in_dim"], hidden=tuple(ckpt["hidden"]),
                 n_pops=ckpt["n_pops"], pop_size=ckpt["pop_size"],
                 neuron=ckpt.get("neuron", "lif"))
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    return net, ckpt["tsteps"]


def readout_bounds(net, margin_scale=2.0):
    """STDP weight bounds around the pretrained readout range, so plasticity nudges
    within a sane band instead of the default [0,1] which would clobber the signed
    trained weights. Symmetric band = margin_scale x the max |weight|."""
    w = net.fc[-1].weight.detach().cpu().numpy()
    b = margin_scale * float(np.abs(w).max())
    return -b, b


def make_rstdp(eta, seed, reward_mode="rpe", gate_threshold=0.0):
    net, T = load_snn_net()
    w_min, w_max = readout_bounds(net)
    cfg = STDPConfig(reward_modulated=True, eta=eta, tau_e_ms=200.0,
                     w_min=w_min, w_max=w_max)
    return SNNNavController(net, n_steps=T, plasticity_enabled=True,
                           stdp_cfg=cfg, seed=seed, reward_mode=reward_mode,
                           gate_threshold=gate_threshold)


def make_frozen_snn(seed):
    net, T = load_snn_net()
    return SNNNavController(net, n_steps=T, plasticity_enabled=False, seed=seed)


def make_online_mlp():
    ckpt = torch.load(MLP_CKPT, weights_only=True)
    pol = MLPPolicy(ckpt["obs_dim"], ckpt["n_actions"], hidden=tuple(ckpt["hidden"]))
    pol.load_state_dict(ckpt["state_dict"]); pol.eval()
    return OnlineMLP(pol, frozen=False, lam=0.9)


def run_block(env, controller, seeds, adapt=True):
    """Run one episode per seed; plastic weights carry across (continual). Returns
    per-episode success (1/0) and collision (1/0)."""
    succ, coll = [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        while True:
            a = controller.act(obs)
            nobs, r, term, trunc, info = env.step(a)
            if adapt:
                controller.observe(r, nobs, term or trunc)
            obs = nobs
            if term or trunc:
                break
        succ.append(int(info["reached"])); coll.append(int(info["collision"]))
    return np.array(succ), np.array(coll)


def guardrail(args):
    """R-STDP on the BASE distribution: pre-shift success must survive plasticity."""
    env = Go2NavEnv(Go2NavConfig(shift_time_s=1e9, episode_len_s=60.0))
    seeds = list(range(4000, 4000 + args.episodes))

    fs, _ = run_block(env, make_frozen_snn(seed=7), seeds, adapt=False)
    fr = fs.mean()
    print(f"frozen SNN (no plasticity), base dist: {fs.sum()}/{len(fs)} = {fr:.0%}", flush=True)

    print("\neta sweep (R-STDP on, no shift) -- pre-shift success must stay ~frozen:")
    rows = []
    for eta in args.etas:
        ss, _ = run_block(env, make_rstdp(eta, seed=7, reward_mode=args.reward_mode,
                                          gate_threshold=args.gate), seeds, adapt=True)
        rows.append((eta, ss.mean()))
        verdict = "OK" if ss.mean() >= fr - 0.10 else "DEGRADED"
        print(f"  eta={eta:<7}: {ss.sum()}/{len(ss)} = {ss.mean():.0%}  [{verdict}]", flush=True)
    env.close()

    safe = [e for e, r in rows if r >= fr - 0.10]
    pick = max(safe) if safe else min(e for e, _ in rows)
    print(f"\nfrozen baseline {fr:.0%}; safe etas {safe}; RECOMMENDED eta = {pick}")
    return pick


def compare(args):
    """Recovery experiment: the PERSISTENTLY-DENSE (post-shift) regime, continual
    adaptation across a block. Using the dense distribution from the start avoids
    the mid-episode-timer confound (goals were reachable before a t=30s shift ever
    fired, so the shift rarely bit). Dense = 16 static + 6 dynamic = the same
    obstacle count the mid-episode shift produces."""
    OUT.mkdir(parents=True, exist_ok=True)
    env = Go2NavEnv(Go2NavConfig(n_static_obstacles=16, n_dynamic_obstacles=6,
                                 shift_time_s=1e9, episode_len_s=60.0))
    seeds = list(range(5000, 5000 + args.episodes))

    controllers = {
        "R-STDP SNN": make_rstdp(args.eta, seed=7, reward_mode=args.reward_mode,
                                 gate_threshold=args.gate),
        "frozen SNN": make_frozen_snn(seed=7),
        "online-MLP": make_online_mlp(),
    }
    # Base (unshifted) env for the RETENTION test: after adapting to the shifted
    # regime, does the controller still solve the ORIGINAL task, or did plasticity
    # overwrite it? This is the "stability" half of the stability-plasticity
    # dilemma (the literature's central risk for online STDP) -- measuring only
    # adaptation would tell half the story.
    base_env = Go2NavEnv(Go2NavConfig(shift_time_s=1e9, episode_len_s=60.0))
    base_seeds = list(range(2000, 2000 + args.episodes))

    results = {}
    for name, ctrl in controllers.items():
        adapt = name != "frozen SNN"
        succ, coll = run_block(env, ctrl, seeds, adapt=adapt)
        # retention: re-test on base distribution with adaptation FROZEN (weights
        # stay at their post-shift-adapted state).
        ret, _ = run_block(base_env, ctrl, base_seeds, adapt=False)
        results[name] = {"succ": succ, "coll": coll, "retain": ret}
        k, n = int(succ.sum()), len(succ)
        rate, lo, hi = wilson_ci95(k, n)
        half = n // 2
        print(f"{name:12s}: dense {k}/{n} = {rate:.0%}  Wilson95 [{lo:.0%},{hi:.0%}]  "
              f"coll {coll.mean():.0%}  | 1st->2nd half {succ[:half].mean():.0%}->"
              f"{succ[half:].mean():.0%}  | base-retention {ret.mean():.0%}", flush=True)
    env.close(); base_env.close()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def sliding(x, w=8):
        return np.array([x[max(0, i - w + 1):i + 1].mean() for i in range(len(x))])

    colors = {"R-STDP SNN": "#c0392b", "frozen SNN": "#7f8c8d", "online-MLP": "#1950a0"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, r in results.items():
        a1.plot(sliding(r["succ"]), color=colors[name], lw=2, label=name)
    a1.set_xlabel("episode in shifted-regime block")
    a1.set_ylabel("success rate (8-ep sliding)")
    a1.set_ylim(0, 1); a1.legend(frameon=False, fontsize=9)
    a1.set_title("continual adaptation under the shift (does R-STDP climb?)")

    # Grouped bars: adaptation (shifted regime) vs retention (base task) per
    # controller -- the two halves of the stability-plasticity dilemma.
    names = list(results)
    x = np.arange(len(names)); wbar = 0.38
    shift_rates = [results[n]["succ"].mean() for n in names]
    retain_rates = [results[n]["retain"].mean() for n in names]
    a2.bar(x - wbar / 2, shift_rates, wbar, color="#c0392b", label="shifted regime (plasticity)")
    a2.bar(x + wbar / 2, retain_rates, wbar, color="#2e86c1", label="base task (retention)")
    a2.axhline(0.41, color="#1f9d3a", ls="--", lw=1, label="M3 base baseline (41%)")
    a2.set_xticks(x); a2.set_xticklabels(names, fontsize=8)
    a2.set_ylabel("success rate"); a2.set_ylim(0, 1)
    a2.legend(frameon=False, fontsize=8)
    a2.set_title("stability-plasticity: adapt vs retain")
    fig.suptitle(f"M4 pilot -- adaptation in the dense (post-shift) regime "
                 f"(eta={args.eta}, gate={args.gate}, {args.episodes} eps/controller)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pilot_recovery.png", bbox_inches="tight", dpi=150)
    print(f"\nwrote {OUT / 'fig_pilot_recovery.png'}")

    r = results["R-STDP SNN"]["succ"]; f = results["frozen SNN"]["succ"]
    half = len(r) // 2
    go = r[half:].mean() > f[half:].mean() + 0.05
    print("\nH1 pilot:", "GO (R-STDP > frozen 2nd-half)" if go
          else "NO-GO / inconclusive (valid finding -- see proposal)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["guardrail", "compare"], required=True)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--etas", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--reward-mode", choices=["td", "rpe", "raw"], default="td",
                    help="td = TD-error via linear critic (SOTA, default); "
                         "rpe = reward-prediction error; raw = env reward")
    ap.add_argument("--gate", type=float, default=0.0,
                    help="RPE gate threshold: suppress the update unless |RPE|>gate "
                         "(learn only when surprised). 0 = ungated.")
    args = ap.parse_args()
    if args.mode == "guardrail":
        guardrail(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
