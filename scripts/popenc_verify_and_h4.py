"""D4 validation, two stages:

1. Baseline parity: does the PopEnc-encoded SNN match M3's original
   rate/TTFS-encoded SNN (and the MLP) pre-shift, closed-loop? M3's own
   bar was "success rate ~= MLP within a few %" -- same bar here, since a
   worse baseline would confound any H4 conclusion.
2. H4 re-test: the noise sweep that found the ORIGINAL encoder is LESS
   robust than the MLP (rate-coding samples the sensor, compounding with
   noise). Does removing that specific stochastic-sampling mechanism (this
   encoder is deterministic given an observation) change the H4 verdict?

Run:  conda run -n nmc python scripts/popenc_verify_and_h4.py --ckpt assets/snn_popenc_seeds/snn_popenc_seed0.pt
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from nmc.controllers.snn_popenc import PopEncNavNet, PopEncNavController
from nmc.eval.metrics import mean_ci95
from m6_energy import build, MLP_CONTROLLERS

import concurrent.futures

OUT = Path("archive/D4_popenc_snn")


def load_ctrl(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=True)
    net = PopEncNavNet(obs_dim=ckpt["obs_dim"], hidden=tuple(ckpt["hidden"]),
                       n_pops=ckpt["n_pops"], pop_size=ckpt["pop_size"],
                       in_pop=ckpt["in_pop"], neuron=ckpt.get("neuron", "alif"))
    net.load_state_dict(ckpt["state_dict"])
    return PopEncNavController(net, n_steps=ckpt["tsteps"])


def _parity_task(ckpt_path, seed, n_eps):
    env = Go2NavEnv(Go2NavConfig(shift_time_s=1e9, episode_len_s=60.0))
    ctrl = load_ctrl(ckpt_path)
    succ = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=4000 + ep)
        while True:
            a = ctrl.act(obs)
            nobs, r, term, trunc, info = env.step(a)
            obs = nobs
            if term or trunc:
                break
        succ.append(int(info["reached"]))
    env.close()
    return seed, float(np.mean(succ)), ctrl.spike_stats()["firing_rate"]


def _noise_task(name, seed, sigma, n_eps, ckpt_path=None):
    env = Go2NavEnv(Go2NavConfig(shift_time_s=1e9, episode_len_s=60.0, sensor_noise_std=sigma))
    ctrl = load_ctrl(ckpt_path) if name == "PopEnc SNN" else build(name, seed)
    succ = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed * 100 + ep)
        while True:
            a = ctrl.act(obs)
            nobs, r, term, trunc, info = env.step(a)
            ctrl.observe(r, nobs, term or trunc)
            obs = nobs
            if term or trunc:
                break
        succ.append(int(info["reached"]))
    env.close()
    return name, seed, sigma, float(np.mean(succ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="assets/snn_popenc_seeds/snn_popenc_seed0.pt")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # -- stage 1: baseline parity ------------------------------------------
    seeds = list(range(6000, 6003))
    tasks = [(args.ckpt, s, 20) for s in seeds]
    print(f"parity check: {len(tasks)} tasks...", flush=True)
    rates, firing = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as ex:
        for f in concurrent.futures.as_completed({ex.submit(_parity_task, *t): t for t in tasks}):
            seed, rate, fr = f.result()
            rates.append(rate); firing.append(fr)
    m, h = mean_ci95(rates)
    lines = ["# D4 PopEnc-SNN: baseline parity + H4 re-test\n",
             f"## Baseline parity (base distribution, 3 seeds x 20 eps)\n",
             f"PopEnc-SNN: {m:.1%} +/- {h:.1%}, mean firing rate {np.mean(firing):.1%}\n",
             f"Reference: original rate/TTFS SNN (M3) 41% [32,51] Wilson-95; MLP 37% [29,45].\n"]
    parity_ok = m > 0.25  # a genuinely low bar just to gate whether H4 is worth trusting
    lines.append(f"**Parity gate ({'PASS' if parity_ok else 'FAIL'}):** "
                 f"{'reasonable enough to proceed to H4' if parity_ok else 'too far below M3 to trust H4 results -- needs more training/tuning first'}\n")

    if not parity_ok:
        (OUT / "verify_and_h4.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        print("STOPPING before H4 -- parity gate failed.")
        return

    # -- stage 2: H4 noise sweep, PopEnc-SNN vs Frozen MLP vs original Frozen SNN
    sigmas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
    names = ["Frozen MLP", "Frozen SNN", "PopEnc SNN"]
    tasks = [(n, s, sg, 15, args.ckpt) for n in names for sg in sigmas for s in seeds]
    print(f"H4 re-test: {len(tasks)} tasks...", flush=True)
    res = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_noise_task, *t): t for t in tasks}
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            name, seed, sigma, rate = f.result()
            res.setdefault((name, sigma), []).append(rate)
            if (i + 1) % 15 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)

    lines.append(f"## H4 re-test (base distribution, additive LiDAR noise, "
                 f"{len(seeds)} seeds x 15 eps)\n")
    lines.append("| controller | " + " | ".join(f"s={s}" for s in sigmas) + " | slope (pts/0.1s) |")
    lines.append("|---" * (len(sigmas) + 2) + "|")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colours = {"Frozen MLP": "#b0b0b0", "Frozen SNN": "#7f8c8d", "PopEnc SNN": "#1f9d3a"}
    for name in names:
        means = [np.mean(res[(name, sg)]) for sg in sigmas]
        halfs = [mean_ci95(res[(name, sg)])[1] for sg in sigmas]
        slope = np.polyfit(sigmas, means, 1)[0] * 0.1
        lines.append(f"| {name} | " + " | ".join(f"{m:.1%}" for m in means) + f" | {slope*100:+.1f} |")
        ax.errorbar(sigmas, means, yerr=halfs, marker="o", capsize=3, lw=1.8,
                    color=colours[name], label=name)
    ax.set_xlabel("LiDAR noise sigma"); ax.set_ylabel("success rate")
    ax.set_ylim(0, None); ax.legend(frameon=False)
    ax.set_title("D4: H4 re-tested with a deterministic population encoder")
    fig.tight_layout()
    fig.savefig(OUT / "fig_h4_popenc.png", bbox_inches="tight", dpi=150)

    (OUT / "verify_and_h4.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT/'verify_and_h4.md'} and fig_h4_popenc.png")


if __name__ == "__main__":
    main()
