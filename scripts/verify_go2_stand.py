"""Verify the MuJoCo Go2 stands stably under PD control; render it + height plot.

Foundation check for the dynamics migration: before walking, confirm the dynamic
Go2 holds its stance (doesn't collapse like the naive open-loop attempt).

Run:  conda run -n nmc python scripts/verify_go2_stand.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nmc.platform.go2_mujoco import Go2MuJoCo, Go2Config  # noqa: E402


def main():
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = ROOT / "archive" / "P1_go2_mujoco"
    outdir.mkdir(parents=True, exist_ok=True)

    robot = Go2MuJoCo(Go2Config())
    robot.reset()
    z0 = robot.base_state()["height"]

    heights, frames = [], []
    for k in range(300):                       # 300 control ticks
        st = robot.stand_step()
        heights.append(st["height"])
        if k % 10 == 0:
            frames.append(robot.render())
    zf = robot.base_state()["height"]

    # Rendered still of the standing Go2.
    Image.fromarray(frames[-1]).save(outdir / "stand.png")
    # Short GIF (proves it holds, not falls).
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(outdir / "stand.gif", save_all=True, append_images=imgs[1:], duration=120, loop=0)

    # Height-over-time stability plot.
    t = np.arange(len(heights)) * robot.model.opt.timestep * robot.cfg.control_decimation
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(t, heights, color="#1950a0", lw=1.8)
    ax.axhspan(0.26, 0.34, color="#2e7d32", alpha=0.12, label="nominal stance")
    ax.axhline(0.15, color="#c0392b", ls="--", lw=1, label="collapse")
    ax.set_xlabel("time (s)"); ax.set_ylabel("base height (m)")
    ax.set_title("MuJoCo Go2 — PD stance hold (full dynamics)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outdir / "fig_stand_stability.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    robot.close()

    stable = zf > 0.25
    print(f"stance height: start z={z0:.3f} -> after 300 ticks z={zf:.3f}")
    print(f"height range: [{min(heights):.3f}, {max(heights):.3f}]")
    print("wrote archive/P1_go2_mujoco/{stand.png, stand.gif, fig_stand_stability.png}")
    print("Go2 STAND:", "PASS (stable)" if stable else "FAIL (collapsed)")
    return 0 if stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
