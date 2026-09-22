"""M4b pilot-robustness extensions, run at M5's statistical standard.

Four parts (select with --part; each writes its own artifact into
archive/M4b_extras/ so they can be run independently):

  severity  (M4b-1) dropout-severity sweep across ALL SIX controllers with CIs.
            M5 only screened the frozen SNN. The open question after M5 is
            whether R-STDP's advantage exists in SOME severity window and the
            two points we tested (0.20, 0.30) simply missed it.
  masks     (M4b-2) is the result mask-specific? M4/M5 always killed the SAME
            beam block (start=8). Re-runs frozen-SNN vs R-STDP on four
            different dead-beam positions.
  neuron    (M4b-7) neuron-model-alone ablation: frozen-LIF vs frozen-ALIF vs
            R-STDP-ALIF under identical dropout, to separate what the adaptive
            threshold buys from what plasticity buys (per Zhao et al. Table VII).
  traj      (M4b-3) trajectory-overlay figure: frozen path vs R-STDP's first
            post-shift episode vs after adaptation.

Run:  conda run -n nmc python scripts/m4b_extras.py --part severity
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats

from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from nmc.controllers.snn import LIFNet, SNNNavController
from nmc.plasticity.stdp import STDPConfig
from nmc.eval.metrics import mean_ci95
from m5_full_comparison import readout_bounds
from m6_energy import build, MLP_CONTROLLERS, SNN_CONTROLLERS

import concurrent.futures

OUT = ROOT / "archive" / "M4b_extras"
ALL_CONTROLLERS = MLP_CONTROLLERS + SNN_CONTROLLERS


def shift_cfg(dropout, start=8):
    return Go2NavConfig(shift_type="sensor", shift_time_s=0.5,
                        sensor_dropout_frac=dropout, sensor_dropout_start=start,
                        episode_len_s=60.0)


def rollout(env, ctrl, seed, num_episodes, log_positions=False):
    succ, paths = [], []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed * 100 + ep)
        pos = [env._robot_pose()[0].copy()] if log_positions else None
        while True:
            a = ctrl.act(obs)
            nobs, r, term, trunc, info = env.step(a)
            ctrl.observe(r, nobs, term or trunc)
            obs = nobs
            if log_positions:
                pos.append(env._robot_pose()[0].copy())
            if term or trunc:
                break
        succ.append(int(info["reached"]))
        if log_positions:
            paths.append(np.array(pos))
    return np.array(succ), paths


# ---------------------------------------------------------------- severity
def _sev_task(name, seed, dropout, n_eps):
    env = Go2NavEnv(shift_cfg(dropout))
    succ, _ = rollout(env, build(name, seed), seed, n_eps)
    env.close()
    return name, dropout, seed, float(succ.mean())


def part_severity(args):
    severities = [0.10, 0.15, 0.20, 0.30]
    seeds = list(range(6000, 6005))
    tasks = [(n, s, d, args.episodes) for n in ALL_CONTROLLERS
             for d in severities for s in seeds]
    print(f"severity sweep: {len(tasks)} tasks x {args.episodes} eps", flush=True)
    res = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_sev_task, *t): t for t in tasks}
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            name, d, seed, rate = f.result()
            res.setdefault((name, d), []).append(rate)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)

    lines = ["# M4b-1 — dropout-severity sweep, all six controllers\n",
             f"{len(seeds)} seeds x {args.episodes} eps per point, 95% CI (t) across seeds.\n",
             "| controller | " + " | ".join(f"drop={d}" for d in severities) + " |",
             "|---" * (len(severities) + 1) + "|"]
    curves = {}
    for name in ALL_CONTROLLERS:
        ms, hs = zip(*[mean_ci95(res[(name, d)]) for d in severities])
        curves[name] = (np.array(ms), np.array(hs))
        lines.append(f"| {name} | " + " | ".join(f"{m:.1%}±{h:.1%}" for m, h in zip(ms, hs)) + " |")

    # the question M5 leaves open: is there ANY severity where R-STDP > frozen SNN?
    lines += ["\n## R-STDP vs frozen SNN at each severity (Welch t-test, per-seed)\n",
              "| severity | frozen SNN | R-STDP | delta | p |", "|---|---|---|---|---|"]
    verdicts = []
    for d in severities:
        fr = np.array(res[("Frozen SNN", d)]); rs = np.array(res[("R-STDP SNN", d)])
        t, p = scipy.stats.ttest_ind(rs, fr, equal_var=False)
        verdicts.append(p < 0.05 and rs.mean() > fr.mean())
        lines.append(f"| {d} | {fr.mean():.1%} | {rs.mean():.1%} | "
                     f"{(rs.mean()-fr.mean())*100:+.1f} pts | {p:.3f} |")
    lines.append(f"\n**Any severity where R-STDP significantly beats frozen SNN? "
                 f"{'YES' if any(verdicts) else 'NO'}**")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    colours = {"Frozen MLP": "#b0b0b0", "Online MLP": "#1950a0", "Frozen SNN": "#7f8c8d",
               "Pure-STDP SNN": "#e67e22", "R-STDP SNN": "#c0392b", "TM-NORM SNN": "#8e44ad"}
    for name in ALL_CONTROLLERS:
        m, h = curves[name]
        ax.errorbar(severities, m, yerr=h, marker="o", capsize=3, lw=1.8,
                    color=colours[name], label=name)
    ax.set_xlabel("sensor-dropout fraction"); ax.set_ylabel("success rate")
    ax.set_ylim(0, None); ax.legend(frameon=False, fontsize=8)
    ax.set_title("M4b-1: severity sweep, all controllers (95% CI)")
    fig.tight_layout(); fig.savefig(OUT / "fig_severity_sweep.png", bbox_inches="tight", dpi=150)
    (OUT / "severity_sweep.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# ------------------------------------------------------------------- masks
def _mask_task(name, seed, start, n_eps):
    env = Go2NavEnv(shift_cfg(0.20, start=start))
    succ, _ = rollout(env, build(name, seed), seed, n_eps)
    env.close()
    return name, start, seed, float(succ.mean())


def part_masks(args):
    starts = [0, 8, 16, 24]
    seeds = list(range(6000, 6005))
    names = ["Frozen SNN", "R-STDP SNN"]
    tasks = [(n, s, st, args.episodes) for n in names for st in starts for s in seeds]
    print(f"multi-mask: {len(tasks)} tasks x {args.episodes} eps", flush=True)
    res = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_mask_task, *t): t for t in tasks}
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            name, st, seed, rate = f.result()
            res.setdefault((name, st), []).append(rate)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)

    lines = ["# M4b-2 — is the result mask-specific?\n",
             f"dropout=0.20, dead-beam block starting at four different positions "
             f"(M4/M5 always used start=8). {len(seeds)} seeds x {args.episodes} eps.\n",
             "| dead-beam start | frozen SNN | R-STDP | delta | p |", "|---|---|---|---|---|"]
    for st in starts:
        fr = np.array(res[("Frozen SNN", st)]); rs = np.array(res[("R-STDP SNN", st)])
        _, p = scipy.stats.ttest_ind(rs, fr, equal_var=False)
        lines.append(f"| {st}{' (M4/M5)' if st == 8 else ''} | {fr.mean():.1%}±{mean_ci95(fr)[1]:.1%} | "
                     f"{rs.mean():.1%}±{mean_ci95(rs)[1]:.1%} | "
                     f"{(rs.mean()-fr.mean())*100:+.1f} pts | {p:.3f} |")
    fr_all = np.concatenate([res[("Frozen SNN", st)] for st in starts])
    rs_all = np.concatenate([res[("R-STDP SNN", st)] for st in starts])
    lines.append(f"\nPooled over masks: frozen {fr_all.mean():.1%}, R-STDP {rs_all.mean():.1%}. "
                 f"Across-mask spread (sd of per-mask means): frozen "
                 f"{np.std([np.mean(res[('Frozen SNN', st)]) for st in starts]):.1%}, "
                 f"R-STDP {np.std([np.mean(res[('R-STDP SNN', st)]) for st in starts]):.1%}.")
    (OUT / "multi_mask.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# ------------------------------------------------------------------ neuron
def _load(ckpt_path, seed, plastic):
    ckpt = torch.load(ckpt_path, weights_only=True)
    net = LIFNet(in_dim=ckpt["in_dim"], hidden=tuple(ckpt["hidden"]),
                 n_pops=ckpt["n_pops"], pop_size=ckpt["pop_size"],
                 neuron=ckpt.get("neuron", "lif"))
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    if not plastic:
        return SNNNavController(net, n_steps=ckpt["tsteps"], plasticity_enabled=False, seed=seed)
    w_min, w_max = readout_bounds(net)
    cfg = STDPConfig(reward_modulated=True, eta=0.05, tau_e_ms=200.0, w_min=w_min, w_max=w_max)
    return SNNNavController(net, n_steps=ckpt["tsteps"], plasticity_enabled=True,
                            stdp_cfg=cfg, seed=seed, reward_mode="td",
                            gate_threshold=0.0, plastic_layers=[0, -1], anchor=0.005)


NEURON_VARIANTS = {
    "frozen LIF": ("assets/snn_seeds_lif/snn_seed0.pt", False),
    "frozen ALIF": ("assets/snn_seeds/snn_seed0.pt", False),
    "R-STDP ALIF": ("assets/snn_seeds/snn_seed0.pt", True),
}


def _neuron_task(variant, seed, n_eps):
    path, plastic = NEURON_VARIANTS[variant]
    env = Go2NavEnv(shift_cfg(0.20))
    succ, _ = rollout(env, _load(ROOT / path, seed, plastic), seed, n_eps)
    env.close()
    return variant, seed, float(succ.mean())


def part_neuron(args):
    seeds = list(range(6000, 6010))
    tasks = [(v, s, args.episodes) for v in NEURON_VARIANTS for s in seeds]
    print(f"neuron ablation: {len(tasks)} tasks x {args.episodes} eps", flush=True)
    res = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_neuron_task, *t): t for t in tasks}
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            v, seed, rate = f.result()
            res.setdefault(v, []).append(rate)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)

    lines = ["# M4b-7 — neuron model alone vs plasticity\n",
             f"dropout=0.20, {len(seeds)} seeds x {args.episodes} eps. Isolates what the "
             f"ALIF adaptive threshold buys from what R-STDP buys.\n",
             "| variant | success | 95% CI |", "|---|---|---|"]
    for v in NEURON_VARIANTS:
        m, h = mean_ci95(res[v])
        lines.append(f"| {v} | {m:.1%} | ±{h:.1%} |")
    lif, alif, rstdp = (np.array(res[v]) for v in
                        ("frozen LIF", "frozen ALIF", "R-STDP ALIF"))
    _, p_na = scipy.stats.ttest_ind(alif, lif, equal_var=False)
    _, p_ar = scipy.stats.ttest_ind(rstdp, alif, equal_var=False)
    lines += [f"\n- ALIF vs LIF (neuron model alone): {(alif.mean()-lif.mean())*100:+.1f} pts, p={p_na:.3f}",
              f"- R-STDP-ALIF vs frozen ALIF (plasticity alone): "
              f"{(rstdp.mean()-alif.mean())*100:+.1f} pts, p={p_ar:.3f}"]
    (OUT / "neuron_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# -------------------------------------------------------------------- traj
def part_traj(args):
    """Frozen path vs R-STDP's first post-shift episode vs after adaptation."""
    seed = 6000
    env = Go2NavEnv(shift_cfg(0.20))
    _, frozen_paths = rollout(env, build("Frozen SNN", seed), seed, 1, log_positions=True)
    rstdp = build("R-STDP SNN", seed)
    _, first = rollout(env, rstdp, seed, 1, log_positions=True)
    rollout(env, rstdp, seed + 1, 12)                       # adaptation block
    _, late = rollout(env, rstdp, seed, 1, log_positions=True)
    obst = [(o[0], o[1], o[2]) for o in env.privileged_state()[3]]
    goal, radius, half = env.goal, env.cfg.goal_radius_m, env.cfg.arena_size_m / 2
    env.close()

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for (x, y, rad) in obst:
        ax.add_patch(plt.Circle((x, y), rad, color="#2c5f9e", alpha=0.45))
    ax.add_patch(plt.Circle(goal, radius, color="#1f9d3a", alpha=0.5))
    for path, colour, lab in [(frozen_paths[0], "#7f8c8d", "frozen SNN"),
                              (first[0], "#e67e22", "R-STDP, first post-shift ep"),
                              (late[0], "#c0392b", "R-STDP, after 12 adaptation eps")]:
        ax.plot(path[:, 0], path[:, 1], lw=2, color=colour, label=lab)
        ax.scatter([path[-1, 0]], [path[-1, 1]], color=colour, marker="x", s=60)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("M4b-3: trajectories under 20% sensor dropout (seed 6000)")
    fig.tight_layout(); fig.savefig(OUT / "fig_trajectory_overlay.png",
                                    bbox_inches="tight", dpi=150)
    print(f"wrote {OUT/'fig_trajectory_overlay.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["severity", "masks", "neuron", "traj"])
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"severity": part_severity, "masks": part_masks,
     "neuron": part_neuron, "traj": part_traj}[args.part](args)


if __name__ == "__main__":
    main()
