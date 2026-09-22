"""Stress-test M4c's terrain result (R-STDP closes the gap to the MLP on
sand/ice) at M5's statistical standard.

M4c's own numbers were computed at n=15, one model-seed (7), one env-seed
block per controller -- explicitly "exploratory, not CI-rigorous" per its own
writeup. That is the SAME thin foundation the sensor-dropout "win" had before
M5 broke it (10 seeds properly ran showed it did not replicate). This is the
overdue check on the thesis's other, currently-unverified positive result.

Reproduces render_terrain_videos.py's exact protocol per seed: R-STDP gets a
15-episode silent warm-up (continual adaptation, not measured) before a
30-episode measured block -- matching M4c's own generous-to-R-STDP design --
just repeated across 10 independent seeds instead of 1, with Holm-corrected
significance tests like M5 used.

Run:  conda run -n nmc python scripts/m4c_rigorous.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np
import scipy.stats
from statsmodels.stats.multitest import multipletests

from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from m6_energy import build, MLP_CONTROLLERS

import concurrent.futures

OUT = Path("archive/M4c_rigorous")
WARMUP = 15
MEASURE = 30
CONDITIONS = {"sand": 1.6, "ice_mu028": 0.28}   # mu=1.6 is M4c's default sand; 0.28 is its VALIDATED best ice
CONTROLLERS = ["Frozen MLP", "Frozen SNN", "R-STDP SNN"]


def env_cfg(terrain, mu):
    return Go2NavConfig(shift_type="terrain", terrain_mode="ice" if "ice" in terrain else "sand",
                        terrain_friction={("ice" if "ice" in terrain else "sand"): mu},
                        shift_time_s=0.5, episode_len_s=45.0)


def evaluate(name, terrain, seed, n_eps):
    mu = CONDITIONS[terrain]
    env = Go2NavEnv(env_cfg(terrain, mu))
    ctrl = build(name, seed)
    adapts = name not in MLP_CONTROLLERS and "Frozen" not in name
    if name == "R-STDP SNN":
        warm_seeds = [seed * 1000 + 500 + i for i in range(WARMUP)]
        for ws in warm_seeds:
            obs, _ = env.reset(seed=ws)
            while True:
                a = ctrl.act(obs)
                nobs, r, term, trunc, info = env.step(a)
                ctrl.observe(r, nobs, term or trunc)
                obs = nobs
                if term or trunc:
                    break
    succ = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed * 1000 + 1000 + ep)
        while True:
            a = ctrl.act(obs)
            nobs, r, term, trunc, info = env.step(a)
            if name == "R-STDP SNN":
                ctrl.observe(r, nobs, term or trunc)
            obs = nobs
            if term or trunc:
                break
        succ.append(int(info["reached"]))
    env.close()
    return name, terrain, seed, float(np.mean(succ))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(range(6000, 6010))
    tasks = [(n, t, s, MEASURE) for t in CONDITIONS for n in CONTROLLERS for s in seeds]
    print(f"{len(tasks)} tasks (warmup={WARMUP} for R-STDP, measure={MEASURE})...", flush=True)

    res = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(evaluate, *t): t for t in tasks}
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            name, terrain, seed, rate = f.result()
            res.setdefault((name, terrain), []).append(rate)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)

    lines = ["# M4c stress-test: terrain result at M5 rigor (10 seeds x 30 eps, Holm-corrected)\n",
             f"Reproduces render_terrain_videos.py's protocol (R-STDP: {WARMUP}-ep silent warm-up "
             f"then {MEASURE}-ep measured block) across 10 seeds instead of M4c's single-ish sample.\n"]

    for terrain in CONDITIONS:
        lines.append(f"\n## {terrain} (mu={CONDITIONS[terrain]})\n")
        lines.append("| controller | success (mean +/- SD) | M4c's original (n=15) |")
        lines.append("|---|---|---|")
        m4c_orig = {"sand": {"Frozen MLP": "47%", "Frozen SNN": "27%", "R-STDP SNN": "47%"},
                    "ice_mu028": {"Frozen MLP": "33%", "Frozen SNN": "27%", "R-STDP SNN": "40%"}}
        arrs = {n: np.array(res[(n, terrain)]) for n in CONTROLLERS}
        for n in CONTROLLERS:
            a = arrs[n]
            lines.append(f"| {n} | {a.mean():.1%} +/- {a.std():.1%} | {m4c_orig[terrain][n]} |")

        rs = arrs["R-STDP SNN"]
        lines.append("\n**R-STDP vs the others (Welch t-test, Holm-corrected):**\n")
        lines.append("| comparison | delta | p (raw) | p (Holm) | significant? |")
        lines.append("|---|---|---|---|---|")
        pvals, deltas, labels = [], [], []
        for other in ["Frozen SNN", "Frozen MLP"]:
            o = arrs[other]
            t, p = scipy.stats.ttest_ind(rs, o, equal_var=False)
            pvals.append(p); deltas.append((rs.mean() - o.mean()) * 100); labels.append(other)
        _, p_corr, _, _ = multipletests(pvals, method="holm")
        for lab, d, p in zip(labels, deltas, p_corr):
            lines.append(f"| R-STDP vs {lab} | {d:+.1f} pts | - | {p:.3f} | {'YES' if p < 0.05 else 'no'} |")

    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT/'README.md'}")


if __name__ == "__main__":
    main()
