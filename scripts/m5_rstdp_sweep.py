"""M5 R-STDP hyperparameter re-sweep across SEEDS (not just M4's single seed=7).

M4's recipe (eta=0.05, anchor=0.005, input+readout, TD-critic) was tuned on one
model-seed/env-seed combination and does not replicate at n=10 seeds (M5:
R-STDP 8.7% vs frozen SNN 11.0%, not significant -- see archive/M5_full_comparison).
This sweeps eta x anchor across 5 seeds x 15 episodes (a fast/coarse screen) to
check whether some other setting in the neighborhood is robustly better than
frozen SNN, before concluding H1 doesn't hold for this shift/network.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nmc.controllers.snn import LIFNet, SNNNavController
from nmc.plasticity.stdp import STDPConfig
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from m5_full_comparison import load_snn_net, readout_bounds, OUT

import concurrent.futures


def make_rstdp_variant(seed, eta, anchor, gate=0.0):
    net, T = load_snn_net()
    w_min, w_max = readout_bounds(net)
    cfg = STDPConfig(reward_modulated=True, eta=eta, tau_e_ms=200.0, w_min=w_min, w_max=w_max)
    return SNNNavController(net, n_steps=T, plasticity_enabled=True,
                            stdp_cfg=cfg, seed=seed, reward_mode="td",
                            gate_threshold=gate, plastic_layers=[0, -1], anchor=anchor)


def evaluate(eta, anchor, seed, shift_cfg, num_episodes):
    env = Go2NavEnv(shift_cfg)
    ctrl = make_rstdp_variant(seed, eta, anchor)
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
    return eta, anchor, seed, np.array(succ)


def main():
    etas = [0.01, 0.02, 0.05, 0.1]
    anchors = [0.0, 0.005, 0.02]
    seeds = list(range(6000, 6005))   # 5 seeds, coarse screen
    num_episodes = 15

    shift_cfg = Go2NavConfig(shift_type="sensor", shift_time_s=0.5,
                             sensor_dropout_frac=0.30, sensor_dropout_start=8,
                             episode_len_s=60.0)

    tasks = [(eta, anchor, s, shift_cfg, num_episodes)
             for eta in etas for anchor in anchors for s in seeds]
    print(f"Sweeping {len(etas)*len(anchors)} configs x {len(seeds)} seeds "
          f"= {len(tasks)} tasks x {num_episodes} eps...", flush=True)

    results = {}  # (eta, anchor) -> list of per-seed mean success
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(evaluate, *t): t for t in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            eta, anchor, seed, succ = future.result()
            results.setdefault((eta, anchor), []).append(succ.mean())
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{len(tasks)} tasks...", flush=True)

    lines = ["# M5 R-STDP hyperparameter re-sweep (5 seeds x 15 eps, coarse screen)\n",
             "Reference (from the full M5 run, 10 seeds x 30 eps): "
             "frozen SNN 11.0%, R-STDP(eta=0.05,anchor=0.005) 8.7%.\n",
             "| eta | anchor | mean success | sd | per-seed |",
             "|---|---|---|---|---|"]
    rows = []
    for (eta, anchor), vals in results.items():
        vals = np.array(vals)
        rows.append((eta, anchor, vals.mean(), vals.std(), vals))
    rows.sort(key=lambda r: -r[2])
    for eta, anchor, mean, sd, vals in rows:
        per_seed = ", ".join(f"{v:.0%}" for v in vals)
        lines.append(f"| {eta} | {anchor} | {mean:.1%} | {sd:.1%} | {per_seed} |")

    out_path = OUT / "rstdp_sweep.md"
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
