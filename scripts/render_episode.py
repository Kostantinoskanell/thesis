"""Render a navigation episode to an animated GIF (visual verification + docs).

Drives the env with a chosen controller, records per-step state, and animates:
robot + heading, moving/static obstacles, LiDAR beams, goal, and the path so far.
Watching this is the ground-truth check that the sim actually behaves correctly.

Run:  conda run -n nmc python scripts/render_episode.py --milestone M1b --seed 2007
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nmc.envs.nav_env import NavEnv, NavConfig  # noqa: E402
from nmc.controllers.privileged_expert import PrivilegedExpert  # noqa: E402


def record(env, expert, seed, max_frames, stride):
    obs, _ = env.reset(seed=seed)
    frames = []
    traj = []
    step = 0
    while True:
        pos, yaw = env._robot_pose()
        _, _, goal, obst = env.privileged_state()
        traj.append(pos.copy())
        if step % stride == 0:
            frames.append(dict(pos=pos.copy(), yaw=yaw, goal=goal.copy(),
                               statics=[(x, y, r) for (x, y, r, vx, vy) in obst if vx == 0 and vy == 0],
                               dynamics=[(x, y, r) for (x, y, r, vx, vy) in obst if vx or vy],
                               lidar=obs[:env.cfg.n_lidar_beams].copy(),
                               traj=np.array(traj), t=env.t,
                               collided=False, reached=False))
        obs, _, term, trunc, info = env.step(expert.act(obs, env))
        collided = info["collision"]
        reached = info["reached"]
        if frames:
            frames[-1]["collided"] = collided
            frames[-1]["reached"] = reached
        step += 1
        if term or trunc or len(frames) >= max_frames:
            break
    return frames, (reached and not collided)


def animate(env, frames, out, fps):
    s = env.cfg.arena_size_m / 2.0
    n_beams = env.cfg.n_lidar_beams
    beam_ang = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
    fig, ax = plt.subplots(figsize=(6, 6))

    def draw(i):
        ax.clear()
        f = frames[i]
        ax.add_patch(plt.Rectangle((-s, -s), 2 * s, 2 * s, fill=False, ec="#555", lw=2))
        for (x, y, r) in f["statics"]:
            ax.add_patch(plt.Circle((x, y), r, color="#2c66c4", alpha=0.85))
        for (x, y, r) in f["dynamics"]:
            ax.add_patch(plt.Circle((x, y), r, color="#e8811a", alpha=0.9))
        pos, yaw = f["pos"], f["yaw"]
        for a, dn in zip(beam_ang, f["lidar"]):
            rr = dn * env.cfg.lidar_max_range_m
            ax.plot([pos[0], pos[0] + rr * np.cos(yaw + a)],
                    [pos[1], pos[1] + rr * np.sin(yaw + a)], color="#ccc", lw=0.4, zorder=1)
        if len(f["traj"]) > 1:
            ax.plot(f["traj"][:, 0], f["traj"][:, 1], color="#1f9d3a", lw=1.6, zorder=2)
        ax.add_patch(plt.Circle(tuple(pos), env.ROBOT_RADIUS, color="#1f9d3a", zorder=3))
        ax.arrow(pos[0], pos[1], 0.6 * np.cos(yaw), 0.6 * np.sin(yaw),
                 head_width=0.22, color="k", zorder=4)
        ax.plot(*f["goal"], marker="*", ms=20, color="#f2c200", mec="k", zorder=3)
        status = "REACHED" if f["reached"] else ("COLLISION" if f["collided"] else "")
        ax.set_title(f"t = {f['t']:4.1f} s    {status}")
        ax.set_xlim(-s - 0.5, s + 0.5); ax.set_ylim(-s - 0.5, s + 0.5)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", default="M1b")
    ap.add_argument("--seed", type=int, default=2007)
    ap.add_argument("--max-frames", type=int, default=90)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--shift", action="store_true", help="enable the mid-episode shift")
    args = ap.parse_args()

    outdir = ROOT / "archive" / (f"{args.milestone}_expert" if args.milestone == "M1b"
                                 else f"{args.milestone}")
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = NavConfig(episode_len_s=45.0,
                    shift_time_s=30.0 if args.shift else 1e9)
    env = NavEnv(cfg)
    expert = PrivilegedExpert()
    # Search for a successful (collision-free, goal-reaching) episode to showcase.
    frames = None
    for seed in range(args.seed, args.seed + 60):
        frames, ok = record(env, expert, seed, args.max_frames, args.stride)
        if ok:
            print(f"showcasing successful episode at seed={seed}")
            break
    env.close()
    out = outdir / "episode.gif"
    animate(env, frames, str(out), args.fps)
    last = frames[-1]
    print(f"frames={len(frames)}  final t={last['t']:.1f}s  "
          f"reached={last['reached']}  collided={last['collided']}")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
