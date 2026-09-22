"""M5 follow-up: does plasticity SCOPE or the third-factor signal explain why
R-STDP (input+readout, TD-critic) ties frozen SNN at the corrected dropout=0.20
severity? Tests the M4b ablation-grid cells that were never actually run:
all-layers plasticity, and reward_mode='rpe' (reward-prediction-error) instead
of the TD-critic -- keeping eta=0.05, anchor=0.005 (M4's recipe) fixed so only
scope/reward-mode vary. Full rigor (10 seeds x 30 eps) from the start, since
the earlier 5-seed coarse sweep gave a false-positive that didn't replicate.

Reference (dropout=0.20, 10 seeds x 30 eps, archive/M5_full_comparison_dropout0.2):
  frozen SNN     21.3% +/- 7.2%
  R-STDP (input+readout, td)  19.3% +/- 5.3%   <- the recipe tested so far
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

from nmc.controllers.snn import SNNNavController
from nmc.plasticity.stdp import STDPConfig
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from m5_full_comparison import load_snn_net, readout_bounds, OUT

import concurrent.futures

VARIANTS = {
    "all-layers + td":       dict(plastic_layers=[0, 1, 2], reward_mode="td"),
    "input+readout + rpe":   dict(plastic_layers=[0, -1],   reward_mode="rpe"),
    "all-layers + rpe":      dict(plastic_layers=[0, 1, 2], reward_mode="rpe"),
}

# already measured at this severity, included for the printed table only
REFERENCE = {
    "frozen SNN": (0.213, 0.072),
    "R-STDP (input+readout, td) [baseline]": (0.193, 0.053),
}


def make_variant(seed, plastic_layers, reward_mode):
    net, T = load_snn_net()
    w_min, w_max = readout_bounds(net)
    cfg = STDPConfig(reward_modulated=True, eta=0.05, tau_e_ms=200.0, w_min=w_min, w_max=w_max)
    return SNNNavController(net, n_steps=T, plasticity_enabled=True,
                            stdp_cfg=cfg, seed=seed, reward_mode=reward_mode,
                            gate_threshold=0.0, plastic_layers=plastic_layers, anchor=0.005)


def evaluate(variant_name, seed, shift_cfg, num_episodes):
    kwargs = VARIANTS[variant_name]
    env = Go2NavEnv(shift_cfg)
    ctrl = make_variant(seed, **kwargs)
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
    return variant_name, seed, np.array(succ).mean()


def main():
    seeds = list(range(6000, 6010))
    shift_cfg = Go2NavConfig(shift_type="sensor", shift_time_s=0.5,
                             sensor_dropout_frac=0.20, sensor_dropout_start=8,
                             episode_len_s=60.0)
    tasks = [(name, s, shift_cfg, 30) for name in VARIANTS for s in seeds]
    print(f"Testing {len(VARIANTS)} variants x {len(seeds)} seeds = {len(tasks)} tasks...", flush=True)

    results = {name: [] for name in VARIANTS}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(evaluate, *t): t for t in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            name, seed, mean = future.result()
            results[name].append(mean)
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{len(tasks)} tasks...", flush=True)

    arrs = {name: np.array(vals) for name, vals in results.items()}

    lines =["# M5 plasticity-scope / third-factor follow-up (dropout=0.20, 10 seeds x 30 eps)\n",
             "| variant | mean success | sd |",
             "|---|---|---|"]
    for name, (m, sd) in REFERENCE.items():
        lines.append(f"| {name} | {m:.1%} | {sd:.1%} |")
    for name, arr in arrs.items():
        lines.append(f"| {name} | {arr.mean():.1%} | {arr.std():.1%} |")

    out_path = OUT.parent / "M5_full_comparison_dropout0.2" / "plasticity_scope.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
