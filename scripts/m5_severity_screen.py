"""M5 follow-up: does 30% sensor dropout leave the FROZEN SNN real headroom, or
is it already near-floored (collision-dominated) -- which would explain why no
amount of R-STDP tuning showed a recovery effect? M4's own headroom screening
(shift_headroom.py) picked 30% using a SINGLE model-seed/env-seed run; this
redoes it across multiple seeds at several severities, mirroring the M4c
terrain lesson (ice mu=0.20 turned out too harsh, mu=0.28 left real headroom).
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from m5_full_comparison import make_frozen_snn, OUT

import concurrent.futures


def evaluate(dropout_frac, seed, num_episodes, shifted):
    if shifted:
        cfg = Go2NavConfig(shift_type="sensor", shift_time_s=0.5,
                           sensor_dropout_frac=dropout_frac, sensor_dropout_start=8,
                           episode_len_s=60.0)
    else:
        cfg = Go2NavConfig(shift_time_s=1e9, episode_len_s=60.0)  # never fires
    env = Go2NavEnv(cfg)
    ctrl = make_frozen_snn(seed)
    outcomes = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed * 100 + ep)
        while True:
            a = ctrl.act(obs)
            nobs, r, term, trunc, info = env.step(a)
            ctrl.observe(r, nobs, term or trunc)
            obs = nobs
            if term or trunc:
                break
        outcomes.append("reached" if info["reached"] else
                        "fell" if info["fell"] else
                        "collision" if info["collision"] else "timeout")
    env.close()
    return dropout_frac, shifted, seed, outcomes


def main():
    severities = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    seeds = list(range(6000, 6005))   # 5 seeds
    num_episodes = 15

    tasks = [(s, sd, num_episodes, True) for s in severities for sd in seeds]
    tasks += [(0.0, sd, num_episodes, False) for sd in seeds]   # unshifted baseline, once
    print(f"Screening {len(severities)} severities x {len(seeds)} seeds "
          f"(+ baseline) = {len(tasks)} tasks x {num_episodes} eps...", flush=True)

    results = {}  # dropout_frac (or "base") -> list of outcome-lists
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(evaluate, *t): t for t in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            dropout_frac, shifted, seed, outcomes = future.result()
            key = "base (unshifted)" if not shifted else dropout_frac
            results.setdefault(key, []).append(outcomes)
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{len(tasks)} tasks...", flush=True)

    lines = ["# M5 sensor-dropout severity screen (frozen SNN, 5 seeds x 15 eps)\n",
             "| severity | mean success | sd | collision% | fell% | timeout% | n eps |",
             "|---|---|---|---|---|---|---|"]
    order = ["base (unshifted)"] + severities
    for key in order:
        blocks = results.get(key)
        if not blocks:
            continue
        per_seed_succ = [sum(o == "reached" for o in oc) / len(oc) for oc in blocks]
        flat = [o for oc in blocks for o in oc]
        n = len(flat)
        succ = np.array(per_seed_succ)
        coll = flat.count("collision") / n
        fell = flat.count("fell") / n
        timeout = flat.count("timeout") / n
        lines.append(f"| {key} | {succ.mean():.1%} | {succ.std():.1%} | {coll:.1%} | "
                     f"{fell:.1%} | {timeout:.1%} | {n} |")

    out_path = OUT / "severity_screen.md"
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
