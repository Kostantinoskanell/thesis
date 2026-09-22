"""M6 / H4 — graceful-degradation curves under sensor noise.

Separate perturbation axis from the dropout *shift*: here the LiDAR is intact
but noisy (additive Gaussian, `Go2NavConfig.sensor_noise_std`), swept on the
BASE distribution so noise-robustness is isolated from shift-recovery. H4 asks
whether the spiking controllers degrade more gracefully than the MLPs -- the
usual neuromorphic claim (temporal filtering + population coding average noise
out) which this thesis should test rather than assert.

Reports per-seed CIs and two summary scalars per controller: the degradation
slope (success points lost per 0.1 sigma, least-squares) and N50 (the noise
level at which success falls to half its clean value, linearly interpolated).

Run:  conda run -n nmc python scripts/m6_noise_sweep.py
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from nmc.eval.metrics import mean_ci95
from m6_energy import build, MLP_CONTROLLERS, SNN_CONTROLLERS

import concurrent.futures

OUT = ROOT / "archive" / "M6_energy"
NOISE_LEVELS = [0.0, 0.025, 0.05, 0.10, 0.20]


def evaluate(name, seed, sigma, num_episodes):
    cfg = Go2NavConfig(shift_time_s=1e9, episode_len_s=60.0, sensor_noise_std=sigma)
    env = Go2NavEnv(cfg)
    ctrl = build(name, seed)
    succ = []
    for ep in range(num_episodes):
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
    return name, sigma, seed, float(np.mean(succ))


def n50(sigmas, rates):
    """Noise level where success first falls to half its clean value."""
    if rates[0] <= 0:
        return float("nan")
    half = rates[0] / 2.0
    for i in range(1, len(rates)):
        if rates[i] <= half:
            x0, x1, y0, y1 = sigmas[i - 1], sigmas[i], rates[i - 1], rates[i]
            if y0 == y1:
                return x1
            return float(x0 + (y0 - half) * (x1 - x0) / (y0 - y1))
    return float("inf")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=float, nargs="+", default=NOISE_LEVELS)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--tag", default="", help="suffix for the output files")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    controllers = MLP_CONTROLLERS + SNN_CONTROLLERS
    seeds = list(range(6000, 6000 + args.seeds))
    n_eps = args.episodes
    globals()["NOISE_LEVELS"] = list(args.levels)

    tasks = [(n, s, sg, n_eps) for n in controllers for sg in NOISE_LEVELS for s in seeds]
    print(f"H4 noise sweep: {len(tasks)} tasks x {n_eps} eps...", flush=True)

    res = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(evaluate, *t): t for t in tasks}
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            name, sigma, seed, rate = f.result()
            res.setdefault((name, sigma), []).append(rate)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)

    lines = ["# M6 / H4 — graceful degradation under sensor noise\n",
             f"Base distribution (no dropout shift), additive Gaussian LiDAR noise, "
             f"{len(seeds)} seeds x {n_eps} eps per point. CI = 95% (t, across seeds).\n",
             "| controller | " + " | ".join(f"σ={s}" for s in NOISE_LEVELS)
             + " | slope (pts per 0.1σ) | N50 |",
             "|---" * (len(NOISE_LEVELS) + 3) + "|"]

    curves = {}
    for name in controllers:
        means, halfs = [], []
        for sg in NOISE_LEVELS:
            m, h = mean_ci95(res[(name, sg)])
            means.append(m); halfs.append(h)
        curves[name] = (np.array(means), np.array(halfs))
        slope = np.polyfit(NOISE_LEVELS, means, 1)[0] * 0.1 * 100   # pts per 0.1 sigma
        cells = " | ".join(f"{m:.1%}±{h:.1%}" for m, h in zip(means, halfs))
        n50v = n50(NOISE_LEVELS, means)
        n50s = "never" if np.isinf(n50v) else ("—" if np.isnan(n50v) else f"{n50v:.3f}")
        lines.append(f"| {name} | {cells} | {slope:.1f} | {n50s} |")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    colours = {"Frozen MLP": "#b0b0b0", "Online MLP": "#1950a0", "Frozen SNN": "#7f8c8d",
               "Pure-STDP SNN": "#e67e22", "R-STDP SNN": "#c0392b", "TM-NORM SNN": "#8e44ad"}
    for name in controllers:
        m, h = curves[name]
        ax.errorbar(NOISE_LEVELS, m, yerr=h, marker="o", capsize=3, lw=1.8,
                    color=colours[name], label=name)
    ax.set_xlabel("LiDAR noise σ"); ax.set_ylabel("success rate")
    ax.set_ylim(0, None); ax.legend(frameon=False, fontsize=8)
    ax.set_title("H4: graceful degradation under sensor noise (95% CI over seeds)")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_h4_noise{args.tag}.png", bbox_inches="tight", dpi=150)

    (OUT / f"noise_sweep{args.tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote noise_sweep{args.tag}.md and fig_h4_noise{args.tag}.png in {OUT}")


if __name__ == "__main__":
    main()
