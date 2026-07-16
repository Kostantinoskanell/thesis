"""Generate milestone figures into archive/<milestone>/.

Run per milestone to build the thesis "journey" archive. These figures are
publication-quality and reusable directly in the thesis.

Usage:  python scripts/make_figures.py m0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nmc.plasticity.stdp import STDPConfig, STDPLearner, kernel_reference  # noqa: E402

ARCHIVE = ROOT / "archive"
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})


def fig_stdp_kernel(outdir: Path) -> Path:
    """The classic biexponential STDP window Delta-w vs Delta-t (proposal Eq. 5)."""
    cfg = STDPConfig(a_plus=1.0, a_minus=1.0, tau_plus_ms=20.0, tau_minus_ms=20.0)
    dt = np.linspace(-80, 80, 800)
    dt = dt[dt != 0]
    dw = kernel_reference(dt, cfg)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dt[dt > 0], dw[dt > 0], color="#1950a0", lw=2, label="LTP ($\\Delta t>0$)")
    ax.plot(dt[dt < 0], dw[dt < 0], color="#c0392b", lw=2, label="LTD ($\\Delta t<0$)")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("$\\Delta t = t_{post} - t_{pre}$  (ms)")
    ax.set_ylabel("$\\Delta w$  (normalized)")
    ax.set_title("STDP learning window (golden reference)")
    ax.legend(frameon=False)
    p = outdir / "fig_stdp_kernel.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_rstdp_demo(outdir: Path) -> Path:
    """Show eligibility trace + reward-gated weight change for a single synapse."""
    cfg = STDPConfig(reward_modulated=True, a_plus=0.05, a_minus=0.05,
                     tau_e_ms=50.0, eta=0.8, dt_ms=1.0, w_min=-1, w_max=1)
    L = STDPLearner(n_pre=1, n_post=1, cfg=cfg)
    W = np.array([[0.0]])
    T = 200
    reward = np.zeros(T)
    reward[120] = 1.0  # a single reward event at t=120 ms
    elig, weight = [], []
    rng = np.random.default_rng(0)
    for t in range(T):
        # correlated pre-before-post pairs during 0..100ms to build LTP eligibility
        pre = np.array([1.0]) if (t < 100 and t % 5 == 0) else np.array([0.0])
        post = np.array([1.0]) if (t < 100 and t % 5 == 1) else np.array([0.0])
        L.step(W, pre, post, reward=float(reward[t]))
        elig.append(L.elig[0, 0])
        weight.append(W[0, 0])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    a1.plot(elig, color="#e08a1e", lw=1.8)
    a1.axvline(120, color="#2e7d32", ls="--", lw=1.2, label="reward $r=1$")
    a1.set_ylabel("eligibility $e_{ij}$")
    a1.set_title("R-STDP: eligibility accumulates, reward consolidates")
    a1.legend(frameon=False)
    a2.plot(weight, color="#1950a0", lw=1.8)
    a2.axvline(120, color="#2e7d32", ls="--", lw=1.2)
    a2.set_ylabel("weight $w_{ij}$")
    a2.set_xlabel("time (ms)")
    p = outdir / "fig_rstdp_demo.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def _reactive_action(obs, n_beams):
    """Tiny gap-follower: brake-free obstacle avoider used only to move the robot
    somewhere interesting for the snapshot (seed of the M1b scripted expert)."""
    lidar = obs[:n_beams]
    front = np.r_[lidar[:n_beams // 8], lidar[-n_beams // 8:]].min()
    if front > 0.35:
        return 0  # forward
    # turn toward the more open side
    left = lidar[n_beams // 4:n_beams // 2].mean()
    right = lidar[n_beams // 2:3 * n_beams // 4].mean()
    return 1 if left > right else 2


def _snapshot(env, ax, title):
    """Top-down schematic of the current env state with LiDAR beams."""
    import pybullet as p
    s = env.cfg.arena_size_m / 2.0
    ax.add_patch(plt.Rectangle((-s, -s), 2 * s, 2 * s, fill=False, ec="#555", lw=2))
    for bid in env.obstacles:
        (x, y, _), _ = p.getBasePositionAndOrientation(bid)
        ax.add_patch(plt.Circle((x, y), env.OBST_RADIUS, color="#2c66c4", alpha=0.85))
    for d in env.dynamic:
        (x, y, _), _ = p.getBasePositionAndOrientation(d["id"])
        ax.add_patch(plt.Circle((x, y), env.OBST_RADIUS, color="#e8811a", alpha=0.9))
    pos, yaw = env._robot_pose()
    lidar = env._raycast_lidar()
    angles = yaw + np.linspace(-np.pi, np.pi, env.cfg.n_lidar_beams, endpoint=False)
    for a, dn in zip(angles, lidar):
        r = dn * env.cfg.lidar_max_range_m
        ax.plot([pos[0], pos[0] + r * np.cos(a)], [pos[1], pos[1] + r * np.sin(a)],
                color="#bbb", lw=0.5, zorder=1)
    ax.add_patch(plt.Circle(tuple(pos), env.ROBOT_RADIUS, color="#1f9d3a", zorder=3))
    ax.arrow(pos[0], pos[1], 0.6 * np.cos(yaw), 0.6 * np.sin(yaw),
             head_width=0.2, color="k", zorder=4)
    ax.plot(*env.goal, marker="*", ms=18, color="#f2c200", mec="k", zorder=3)
    ax.set_xlim(-s - 0.5, s + 0.5)
    ax.set_ylim(-s - 0.5, s + 0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def build_m1():
    from nmc.envs.nav_env import NavEnv, NavConfig  # noqa: E402
    outdir = ARCHIVE / "M1_env_shift"
    outdir.mkdir(parents=True, exist_ok=True)
    env = NavEnv(NavConfig(seed=7, shift_time_s=30.0, episode_len_s=60.0))
    obs, _ = env.reset(seed=7)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.6))
    pre_done = False
    while True:
        obs, _, terminated, truncated, info = env.step(_reactive_action(obs, env.cfg.n_lidar_beams))
        if (not pre_done) and info["t"] >= 28.0:
            _snapshot(env, a1, "Phase 1  (t=28 s):  8 static + 3 dynamic")
            pre_done = True
        if info["t"] >= 33.0:
            _snapshot(env, a2, "Phase 2  (t=33 s):  shift injected — 16 static + 6 dynamic")
            break
        if terminated:      # reactive policy may still clip a corner; keep the clock
            pass
        if truncated:
            break
    env.close()
    fig.suptitle("M1 — navigation arena with mid-episode distribution shift", fontsize=13)
    legend = [plt.Line2D([], [], marker="o", ls="", color="#2c66c4", label="static obstacle"),
              plt.Line2D([], [], marker="o", ls="", color="#e8811a", label="dynamic obstacle"),
              plt.Line2D([], [], marker="o", ls="", color="#1f9d3a", label="robot"),
              plt.Line2D([], [], marker="*", ls="", color="#f2c200", mec="k", label="goal")]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    out = outdir / "fig_arena_shift.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    milestone_readme(
        outdir,
        "M1 — PyBullet nav env with mid-episode distribution shift",
        [
            "Differential-drive (kinematic unicycle) robot, 32-beam 360° LiDAR, 10x10 m arena.",
            "Smoke test PASS: 60 s episode, shift fires at exactly t=30 s, static 8->16, dynamic 3->6.",
            "`fig_arena_shift.png` — top-down view before (Phase 1) and after (Phase 2) the shift, "
            "with LiDAR beams; the obstacle-density doubling is the adaptivity test.",
        ],
        "2026-07-17",
    )
    print("wrote", out.relative_to(ROOT))


def milestone_readme(outdir: Path, title: str, bullets: list[str], date: str) -> Path:
    p = outdir / "README.md"
    lines = [f"# {title}", "", f"_Archived {date}_", ""]
    lines += [f"- {b}" for b in bullets]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def build_m0():
    outdir = ARCHIVE / "M0_scaffold"
    outdir.mkdir(parents=True, exist_ok=True)
    figs = [fig_stdp_kernel(outdir), fig_rstdp_demo(outdir)]
    milestone_readme(
        outdir,
        "M0 — Repo scaffold + verified plasticity golden reference",
        [
            "STDP/R-STDP online learner implemented in NumPy (golden reference for the FPGA).",
            "6/6 unit tests green: LTP/LTD signs, weight saturation, R-STDP reward gating.",
            "`fig_stdp_kernel.png` — the biexponential STDP window from `kernel_reference`.",
            "`fig_rstdp_demo.png` — eligibility trace accumulates, then a reward event "
            "consolidates it into a weight change (the mechanism H1 rests on).",
        ],
        "2026-07-17",
    )
    for f in figs:
        print("wrote", f.relative_to(ROOT))


if __name__ == "__main__":
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "m0"
    {"m0": build_m0, "m1": build_m1}[which]()
