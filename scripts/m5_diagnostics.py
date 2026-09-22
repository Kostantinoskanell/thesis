"""M5 diagnostics: per-seed breakdown + termination-reason decomposition.

Reuses the exact same controllers/shift/seeds as m5_full_comparison.py so the
numbers line up with that run, but additionally records WHY each episode ended
(reached / collision / fell / timeout) -- to check whether the low success
rates are a navigation-decision problem or the underlying locomotion policy
falling over mid-episode (a confound the aggregate success rate can't see).
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
from m5_full_comparison import (make_frozen_mlp, make_online_mlp, make_frozen_snn,
                                 make_pure_stdp, make_rstdp, make_tmnorm, OUT)

import concurrent.futures


def evaluate_seed(name, seed, shift_cfg, num_episodes):
    env = Go2NavEnv(shift_cfg)
    if name == "Frozen MLP": ctrl = make_frozen_mlp()
    elif name == "Online MLP": ctrl = make_online_mlp()
    elif name == "Frozen SNN": ctrl = make_frozen_snn(seed)
    elif name == "Pure-STDP SNN": ctrl = make_pure_stdp(seed)
    elif name == "R-STDP SNN": ctrl = make_rstdp(seed)
    elif name == "TM-NORM SNN": ctrl = make_tmnorm(seed)
    else: raise ValueError(f"Unknown controller {name}")

    outcomes = []  # one of "reached" | "collision" | "fell" | "timeout" per episode
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed * 100 + ep)
        while True:
            a = ctrl.act(obs)
            nobs, r, term, trunc, info = env.step(a)
            ctrl.observe(r, nobs, term or trunc)
            obs = nobs
            if term or trunc:
                break
        if info["reached"]:
            outcomes.append("reached")
        elif info["fell"]:
            outcomes.append("fell")
        elif info["collision"]:
            outcomes.append("collision")
        else:
            outcomes.append("timeout")

    env.close()
    return name, seed, outcomes


def main():
    seeds = list(range(6000, 6010))
    controllers = ["Frozen MLP", "Online MLP", "Frozen SNN", "Pure-STDP SNN",
                   "R-STDP SNN", "TM-NORM SNN"]

    shift_cfg = Go2NavConfig(shift_type="sensor", shift_time_s=0.5,
                             sensor_dropout_frac=0.30, sensor_dropout_start=8,
                             episode_len_s=60.0)

    tasks = [(name, s, shift_cfg, 30) for name in controllers for s in seeds]

    results = {name: {} for name in controllers}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(evaluate_seed, *t): t for t in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            name, seed, outcomes = future.result()
            results[name][seed] = outcomes
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{len(tasks)} tasks...", flush=True)

    lines = ["# M5 Diagnostics: per-seed breakdown + termination reasons\n"]
    for name in controllers:
        lines.append(f"\n## {name}\n")
        lines.append("| seed | reached | collision | fell | timeout | success% |")
        lines.append("|---|---|---|---|---|---|")
        all_outcomes = []
        for seed in seeds:
            oc = results[name][seed]
            all_outcomes.extend(oc)
            n = len(oc)
            c = {k: oc.count(k) for k in ("reached", "collision", "fell", "timeout")}
            lines.append(f"| {seed} | {c['reached']} | {c['collision']} | {c['fell']} | "
                        f"{c['timeout']} | {c['reached']/n:.1%} |")
        n = len(all_outcomes)
        c = {k: all_outcomes.count(k) for k in ("reached", "collision", "fell", "timeout")}
        lines.append(f"| **TOTAL ({n})** | {c['reached']} ({c['reached']/n:.1%}) | "
                     f"{c['collision']} ({c['collision']/n:.1%}) | "
                     f"{c['fell']} ({c['fell']/n:.1%}) | "
                     f"{c['timeout']} ({c['timeout']/n:.1%}) | |")

    out_path = OUT / "diagnostics.md"
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
