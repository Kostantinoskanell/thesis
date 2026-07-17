"""D3 visual verification: A* teacher drives the Go2 through the nav env.

Runs one full episode WITH the mid-episode shift, rendering the real dynamics:
the Go2 trots between obstacles toward the goal disc, at t=30 s the obstacle
field doubles (visible in the GIF), moving obstacles drift and bounce.

Saves to archive/D3_go2_nav_env/:
  * nav_episode.gif        -- tracking-camera render
  * fig_nav_episode.png    -- top-down trajectory (pre/post-shift), lidar min,
                              goal distance, action histogram

Run on Windows (conda nmc):  python scripts/render_go2_nav.py [--seed N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig, OBST_RADIUS
from nmc.controllers.privileged_expert import PrivilegedExpert, PrivilegedConfig

OUT = ROOT / "archive" / "D3_go2_nav_env"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gif-every", type=int, default=5)
    # Default 30 s often ends before the shift fires (teacher is good); pull it
    # forward for the demo so the injection is visible in the GIF.
    ap.add_argument("--shift-time", type=float, default=30.0)
    ap.add_argument("--show-collision", action="store_true",
                    help="overlay collision geometry (robot capsules + obstacle/wall "
                         "boxes) and contact points")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gif_name = "nav_episode_collision.gif" if args.show_collision else "nav_episode.gif"

    env = Go2NavEnv(Go2NavConfig(seed=args.seed, shift_time_s=args.shift_time))
    # Inflation sized for the Go2 footprint (bounding circle 0.45 m + margin),
    # with progressive fallback for the post-shift dense field (see debug-log).
    # Floor at 0.55: tighter than that leaves zero real margin for the 0.45 m
    # bounding circle of a robot that sways while trotting (collided at 0.45).
    expert = PrivilegedExpert(PrivilegedConfig(
        inflate_m=0.9, fallback_inflations_m=(0.9, 0.7, 0.55)))

    obs, _ = env.reset(seed=args.seed)
    traj, lidar_min, goal_dist, actions, times = [], [], [], [], []
    frames, shift_frame = [], None
    obst_snapshots = {}

    step = 0
    while True:
        a = expert.act(obs, env)
        obs, r, term, trunc, info = env.step(a)
        pos, _ = env._robot_pose()
        traj.append(pos.copy()); lidar_min.append(obs[:32].min())
        goal_dist.append(env._goal_distance()); actions.append(a)
        times.append(info["t"])
        if step % args.gif_every == 0:
            frames.append(Image.fromarray(env.render(w=480, h=360, cam_dist=4.5,
                                                     azimuth=90, elevation=-40,
                                                     show_collision=args.show_collision)))
            if info["phase"] == 2 and shift_frame is None:
                shift_frame = len(frames) - 1
        if info["phase"] == 2 and 2 not in obst_snapshots:
            obst_snapshots[2] = [o[:3] for o in env.privileged_state()[3]]
        if 1 not in obst_snapshots:
            obst_snapshots[1] = [o[:3] for o in env.privileged_state()[3]]
        step += 1
        if term or trunc:
            break

    verdict = ("REACHED" if info["reached"] else
               "FELL" if info["fell"] else
               f"COLLIDED({info['collision_kind']})" if info["collision"] else "TIMEOUT")
    print(f"episode: {verdict} at t={info['t']:.1f}s  steps={step}  phase={info['phase']}")

    gif = OUT / gif_name
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f"wrote {gif} ({len(frames)} frames, shift at frame {shift_frame})")

    # -- figure -------------------------------------------------------------
    traj = np.array(traj); times = np.array(times)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    (a1, a2), (a3, a4) = axes

    shift_t = env.cfg.shift_time_s
    pre = times <= shift_t
    a1.plot(traj[pre, 0], traj[pre, 1], color="#1950a0", lw=1.8, label="pre-shift path")
    if (~pre).any():
        a1.plot(traj[~pre, 0], traj[~pre, 1], color="#c0392b", lw=1.8, label="post-shift path")
    for (x, y, rad) in obst_snapshots.get(1, []):
        a1.add_patch(plt.Circle((x, y), rad, color="#2c5f9e", alpha=0.55))
    for (x, y, rad) in obst_snapshots.get(2, []):
        a1.add_patch(plt.Circle((x, y), rad, color="#e67e22", alpha=0.30))
    a1.add_patch(plt.Circle(env.goal, env.cfg.goal_radius_m, color="#1f9d3a", alpha=0.5))
    a1.scatter([env.start_pos[0]], [env.start_pos[1]], color="k", marker="s",
               zorder=3, label="start")
    s = env.cfg.arena_size_m / 2
    a1.set_xlim(-s, s); a1.set_ylim(-s, s); a1.set_aspect("equal")
    a1.legend(frameon=False, fontsize=8)
    a1.set_title(f"top-down: blue=base obstacles, orange=shift-injected — {verdict}")

    a2.plot(times, goal_dist, color="#1950a0", lw=1.5)
    a2.axvline(shift_t, color="#c0392b", ls="--", lw=1.2, label="shift")
    a2.axhline(env.cfg.goal_radius_m, color="#1f9d3a", ls="--", lw=1, label="goal radius")
    a2.set_xlabel("time (s)"); a2.set_ylabel("distance to goal (m)")
    a2.legend(frameon=False, fontsize=9); a2.set_title("goal approach")

    a3.plot(times, lidar_min, color="#8e44ad", lw=1.2)
    a3.axvline(shift_t, color="#c0392b", ls="--", lw=1.2)
    a3.set_xlabel("time (s)"); a3.set_ylabel("min lidar (normalized)")
    a3.set_title("closest obstacle reading (drops after shift = denser field)")

    a4.hist(actions, bins=np.arange(5) - 0.5, rwidth=0.7, color="#1950a0")
    a4.set_xticks(range(4), ["fwd", "left", "right", "brake"])
    a4.set_title("action distribution (A* teacher)")

    fig.suptitle(f"Go2 nav env (full dynamics) — A* teacher episode, seed {args.seed}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_nav_episode.png", bbox_inches="tight", dpi=150)
    print(f"wrote {OUT / 'fig_nav_episode.png'}")


if __name__ == "__main__":
    main()
