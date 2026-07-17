"""Demo + verification: Go2 walks in the WINDOWS MuJoCo loop via the NumPy policy.

Runs a multi-phase velocity-command schedule (walk fwd -> turn -> fast fwd ->
stop) through nmc.platform.go2_rl_walker -- i.e. the exact interface the SNN
navigator will drive in D3 -- and saves, to archive/D3a_policy_export_windows/:
  * walk_windows.gif       -- rendered episode
  * fig_windows_tracking.png -- cmd-vs-actual vx, yaw rate, height
  * top-down trajectory subplot (turn arc should be visible)

Run on Windows (conda env nmc):  python scripts/render_go2_rl_windows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nmc.platform.go2_rl_walker import Go2RLWalker, CTRL_DT

OUT = ROOT / "archive" / "D3a_policy_export_windows"

# (duration_s, vx, vy, omega) -- the phases the SNN will later produce live.
SCHEDULE = [
    (4.0, 0.8, 0.0, 0.0),    # walk forward
    (4.0, 0.6, 0.0, 0.7),    # arc left
    (4.0, 1.0, 0.0, 0.0),    # fast forward
    (3.0, 0.0, 0.0, 0.0),    # stop & stand
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bot = Go2RLWalker()

    t, cmd_vx, act_vx, cmd_w, act_w, heights, xy = [], [], [], [], [], [], []
    frames = []
    now = 0.0
    for dur, vx, vy, om in SCHEDULE:
        bot.set_command(vx, vy, om)
        for i in range(int(round(dur / CTRL_DT))):
            s = bot.step()
            now += CTRL_DT
            t.append(now)
            cmd_vx.append(vx); act_vx.append(s["vx"])
            cmd_w.append(om);  act_w.append(s["yaw_rate"])
            heights.append(s["height"]); xy.append(s["pos"][:2])
            if len(t) % 7 == 0:                      # compact GIF
                frames.append(Image.fromarray(bot.render()))
    bot.close()

    gif = OUT / "walk_windows.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=140, loop=0)

    t = np.array(t); xy = np.array(xy)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    (a1, a2), (a3, a4) = axes
    a1.plot(t, act_vx, color="#1950a0", lw=1.4, label="actual vx")
    a1.plot(t, cmd_vx, color="#888", ls="--", lw=1.3, label="commanded vx")
    a1.set_ylabel("forward speed (m/s)"); a1.set_xlabel("time (s)")
    a1.legend(frameon=False, fontsize=9)
    a1.set_title("velocity tracking (Windows NumPy runtime)")

    a2.plot(t, act_w, color="#8e44ad", lw=1.4, label="actual yaw rate")
    a2.plot(t, cmd_w, color="#888", ls="--", lw=1.3, label="commanded ω")
    a2.set_ylabel("yaw rate (rad/s)"); a2.set_xlabel("time (s)")
    a2.legend(frameon=False, fontsize=9)
    a2.set_title("turn tracking")

    a3.plot(t, heights, color="#1f9d3a", lw=1.4)
    a3.axhspan(0.22, 0.32, color="#2e7d32", alpha=0.12, label="upright band")
    a3.axhline(0.15, color="#c0392b", ls="--", lw=1, label="fall")
    a3.set_ylabel("base height (m)"); a3.set_xlabel("time (s)")
    a3.legend(frameon=False, fontsize=9)
    a3.set_title("stability")

    a4.plot(xy[:, 0], xy[:, 1], color="#1950a0", lw=1.6)
    a4.scatter([xy[0, 0]], [xy[0, 1]], color="#1f9d3a", zorder=3, label="start")
    a4.scatter([xy[-1, 0]], [xy[-1, 1]], color="#c0392b", zorder=3, label="end")
    a4.set_aspect("equal"); a4.set_xlabel("x (m)"); a4.set_ylabel("y (m)")
    a4.legend(frameon=False, fontsize=9)
    a4.set_title("top-down trajectory (arc = turn phase)")

    fig.suptitle("Go2 RL policy on WINDOWS (pure NumPy, no JAX) — same policy as WSL training",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_windows_tracking.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # numeric verdicts per phase
    t = np.asarray(t)
    print(f"wrote {gif} ({len(frames)} frames) + fig_windows_tracking.png")
    edges = np.cumsum([0] + [d for d, *_ in SCHEDULE])
    for k, (dur, vx, vy, om) in enumerate(SCHEDULE):
        m = (t > edges[k] + 1.0) & (t <= edges[k + 1])     # skip 1s transient
        print(f"phase {k} (vx={vx}, w={om}): "
              f"vx {np.mean(np.array(act_vx)[m]):+.2f}±{np.std(np.array(act_vx)[m]):.2f}  "
              f"w {np.mean(np.array(act_w)[m]):+.2f}±{np.std(np.array(act_w)[m]):.2f}  "
              f"hmin {np.min(np.array(heights)[m]):.3f}")
    print(f"upright throughout: {np.min(heights) > 0.15}")


if __name__ == "__main__":
    main()
