"""Shift taxonomy: run the three implemented-but-never-tested shift types
(sensor_bias, sensor_range, goal_drift) through the exact same two-stage
protocol M5 used for sensor dropout: (1) severity screen against the frozen
SNN to find a level with real headroom (not floored, not trivial -- the "M4
lesson", repeated three more times because skipping it once already produced
a floored, uninformative severity for sensor dropout), then (2) the full
6-controller, 10-seed, Holm-corrected comparison at that severity.

This turns "R-STDP is shift-dependent" from a two-point anecdote (sensor
dropout: no: terrain: yes) into an actual taxonomy across five shift classes.

Run:
  conda run -n nmc python scripts/shift_taxonomy.py --part severity --shift sensor_bias
  conda run -n nmc python scripts/shift_taxonomy.py --part rigor --shift sensor_bias --severity 1.5
  conda run -n nmc python scripts/shift_taxonomy.py --part all   # screens + auto-picks + runs rigor for all 3
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np
import scipy.stats
from statsmodels.stats.multitest import multipletests

from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from nmc.eval.metrics import mean_ci95
from m6_energy import build, MLP_CONTROLLERS, SNN_CONTROLLERS

import concurrent.futures

OUT = Path("archive/shift_taxonomy")
CONTROLLERS = MLP_CONTROLLERS + SNN_CONTROLLERS

# severity grids per shift type, and the CONFIG FIELD each severity value maps to
SEVERITY_GRID = {
    "sensor_bias": {"field": "lidar_bias_m", "values": [0.5, 1.0, 1.5, 2.0, 3.0]},
    "sensor_range": {"field": "sensor_range_m", "values": [1.5, 2.0, 3.0, 4.0, 6.0]},
    "goal_drift": {"field": "goal_drift_rad", "values": [0.15, 0.3, 0.6, 0.9, 1.2]},
}


def shift_cfg(shift, severity):
    field = SEVERITY_GRID[shift]["field"]
    return Go2NavConfig(shift_type=shift, shift_time_s=0.5, episode_len_s=60.0,
                        **{field: severity})


def _eval_task(name, seed, shift, severity, n_eps):
    env = Go2NavEnv(shift_cfg(shift, severity))
    ctrl = build(name, seed)
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
    return name, seed, float(np.mean(succ))


def _eval_task_rec(name, seed, shift, severity, n_eps):
    """Same as _eval_task but also returns a per-episode success array for
    the rolling-window recovery-time metric, matching m5_full_comparison.py."""
    env = Go2NavEnv(shift_cfg(shift, severity))
    ctrl = build(name, seed)
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
    succ_arr = np.array(succ)
    window = 5
    smoothed = np.convolve(succ_arr, np.ones(window) / window, mode="valid")
    rec_idx = np.where(smoothed >= 0.30)[0]
    rec_time = int(rec_idx[0]) if len(rec_idx) > 0 else n_eps
    return name, seed, float(succ_arr.mean()), rec_time


def run_severity(shift, seeds=range(6000, 6005), n_eps=15):
    grid = SEVERITY_GRID[shift]["values"]
    tasks = [(n, s, shift, sv, n_eps) for n in CONTROLLERS for sv in grid for s in seeds]
    print(f"[{shift}] severity screen: {len(tasks)} tasks x {n_eps} eps...", flush=True)
    res = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_eval_task, *t): t for t in tasks}
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            name, seed, rate = f.result()
            key = (name, futs[f][3])
            res.setdefault(key, []).append(rate)
            if (i + 1) % 20 == 0:
                print(f"  [{shift}] {i+1}/{len(tasks)}", flush=True)

    lines = [f"# Shift taxonomy: severity screen for `{shift}`\n",
             f"5 seeds x {n_eps} eps per point, 95% CI (t) across seeds.\n",
             "| controller | " + " | ".join(f"sev={sv}" for sv in grid) + " |",
             "|---" * (len(grid) + 1) + "|"]
    frozen_means = {}
    for name in CONTROLLERS:
        cells = []
        for sv in grid:
            m, h = mean_ci95(res[(name, sv)])
            cells.append(f"{m:.1%}+/-{h:.1%}")
            if name == "Frozen SNN":
                frozen_means[sv] = m
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    # pick severity: largest drop from base while frozen SNN success stays > 5%
    base_env_rate = frozen_means[grid[0]]  # mildest severity as proxy for "near-base"
    candidates = [(sv, base_env_rate - frozen_means[sv]) for sv in grid if frozen_means[sv] > 0.05]
    picked = max(candidates, key=lambda x: x[1])[0] if candidates else grid[len(grid) // 2]
    lines.append(f"\n**Auto-picked severity for the rigor stage: {picked}** "
                 f"(largest drop from mildest tested that keeps frozen SNN > 5%)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"severity_{shift}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT}/severity_{shift}.md")
    return picked


def run_rigor(shift, severity, seeds=range(6000, 6010), n_eps=30):
    tasks = [(n, s, shift, severity, n_eps) for n in CONTROLLERS for s in seeds]
    print(f"[{shift}] rigor @ severity={severity}: {len(tasks)} tasks x {n_eps} eps...", flush=True)
    results = {n: {"success": [], "recovery": []} for n in CONTROLLERS}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_eval_task_rec, *t): t for t in tasks}
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            name, seed, rate, rec = f.result()
            results[name]["success"].append(rate)
            results[name]["recovery"].append(rec)
            if (i + 1) % 10 == 0:
                print(f"  [{shift}] {i+1}/{len(tasks)}", flush=True)

    arrs = {n: {"success": np.array(v["success"]), "recovery": np.array(v["recovery"])}
           for n, v in results.items()}
    rstdp = arrs["R-STDP SNN"]
    names = [n for n in CONTROLLERS if n != "R-STDP SNN"]
    p_succ = [scipy.stats.ttest_ind(rstdp["success"], arrs[n]["success"], equal_var=False)[1] for n in names]
    p_rec = [scipy.stats.ttest_ind(rstdp["recovery"], arrs[n]["recovery"], equal_var=False)[1] for n in names]
    _, p_succ_c, _, _ = multipletests(p_succ, method="holm")
    _, p_rec_c, _, _ = multipletests(p_rec, method="holm")

    lines = [f"# Shift taxonomy: rigor stage for `{shift}` @ severity={severity}\n",
             f"10 seeds x {n_eps} eps, Holm-corrected significance vs R-STDP SNN.\n",
             "| controller | success (mean+/-SD) | recovery (eps) | p vs R-STDP (success) | p vs R-STDP (recovery) |",
             "|---|---|---|---|---|"]
    for n in CONTROLLERS:
        a = arrs[n]
        if n == "R-STDP SNN":
            ps, pr = "-", "-"
        else:
            idx = names.index(n)
            ps = f"{p_succ_c[idx]:.3f}" + (" *" if p_succ_c[idx] < 0.05 else "")
            pr = f"{p_rec_c[idx]:.3f}" + (" *" if p_rec_c[idx] < 0.05 else "")
        lines.append(f"| {n} | {a['success'].mean():.1%}+/-{a['success'].std():.1%} | "
                     f"{a['recovery'].mean():.1f} | {ps} | {pr} |")

    beats_frozen = (rstdp["success"].mean() > arrs["Frozen SNN"]["success"].mean() and
                   p_succ_c[names.index("Frozen SNN")] < 0.05)
    lines.append(f"\n**R-STDP significantly beats frozen SNN on `{shift}`? "
                 f"{'YES' if beats_frozen else 'NO'}**")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rigor_{shift}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT}/rigor_{shift}.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["severity", "rigor", "all"])
    ap.add_argument("--shift", choices=list(SEVERITY_GRID), default=None)
    ap.add_argument("--severity", type=float, default=None)
    args = ap.parse_args()

    if args.part == "all":
        for shift in SEVERITY_GRID:
            picked = run_severity(shift)
            run_rigor(shift, picked)
        # final cross-shift taxonomy summary
        summary = ["# Shift taxonomy: cross-shift summary\n",
                   "| shift type | R-STDP beats frozen SNN? |", "|---|---|"]
        for shift in SEVERITY_GRID:
            f = OUT / f"rigor_{shift}.md"
            verdict = "?"
            if f.exists():
                txt = f.read_text(encoding="utf-8")
                verdict = "YES" if "beats frozen SNN on" in txt and "YES**" in txt else "NO"
            summary.append(f"| {shift} | {verdict} |")
        summary.append("\n(compare with: sensor dropout = NO (M5), terrain sand/ice = see M4c_rigorous)")
        (OUT / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
        print("\n".join(summary))
        return

    if args.part == "severity":
        run_severity(args.shift)
    else:
        if args.severity is None:
            raise SystemExit("--severity required for --part rigor")
        run_rigor(args.shift, args.severity)


if __name__ == "__main__":
    main()
