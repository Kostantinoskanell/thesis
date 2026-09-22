"""Validate the sweep's best candidate (eta=0.1, anchor=0.02) at full rigor:
10 seeds x 30 episodes, same seeds/shift as the main M5 run, so it's directly
comparable (paired by seed) against the already-measured frozen SNN (11.0% +-
2.6%) and the M4-recipe R-STDP (8.7% +- 3.7%)."""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import scipy.stats

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from m5_full_comparison import OUT, make_frozen_snn
from m5_rstdp_sweep import make_rstdp_variant

import concurrent.futures

# From the m5_full_comparison.py run (10 seeds x 30 eps, same shift/seeds).
FROZEN_SNN_SUCCESS = np.array([0.1000, 0.1333, 0.0667, 0.1000, 0.1000,
                               0.1000, 0.1667, 0.1000, 0.1000, 0.1333])
M4_RECIPE_SUCCESS = None  # filled from archive if needed; not used in the test below


def evaluate(seed, shift_cfg, num_episodes):
    env = Go2NavEnv(shift_cfg)
    ctrl = make_rstdp_variant(seed, eta=0.1, anchor=0.02)
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
    return seed, np.array(succ).mean()


def main():
    seeds = list(range(6000, 6010))
    shift_cfg = Go2NavConfig(shift_type="sensor", shift_time_s=0.5,
                             sensor_dropout_frac=0.30, sensor_dropout_start=8,
                             episode_len_s=60.0)
    tasks = [(s, shift_cfg, 30) for s in seeds]
    print(f"Validating eta=0.1, anchor=0.02 over {len(seeds)} seeds x 30 eps...", flush=True)

    means = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(evaluate, *t): t for t in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            seed, mean = future.result()
            means[seed] = mean
            print(f"Completed {i+1}/{len(tasks)}: seed {seed} -> {mean:.1%}", flush=True)

    vals = np.array([means[s] for s in seeds])
    t, p = scipy.stats.ttest_ind(vals, FROZEN_SNN_SUCCESS, equal_var=False)

    lines = ["# R-STDP (eta=0.1, anchor=0.02) validation: 10 seeds x 30 eps\n",
             f"mean {vals.mean():.1%} +/- {vals.std():.1%} (per-seed: "
             + ", ".join(f'{v:.0%}' for v in vals) + ")\n",
             f"frozen SNN reference: 11.0% +/- 2.6%\n",
             f"Welch t-test vs frozen SNN: t={t:.3f}, p={p:.4f} "
             f"({'significant' if p < 0.05 else 'NOT significant'})\n"]
    out_path = OUT / "rstdp_validate.md"
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
