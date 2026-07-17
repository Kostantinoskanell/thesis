"""Render the saved Go2 rollout to a GIF + make a velocity-tracking graph.

Same shape as render_go1_rollout.py, pointed at our ported Go2JoystickFlatTerrain
env. Uses Playground's exact Go2 mj_model (so qpos matches), replays the saved
qpos trajectory through an offscreen renderer. Runs in WSL2 (needs EGL:
MUJOCO_GL=egl).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import nmc.rl.envs.go2  # noqa: F401  (registers Go2JoystickFlatTerrain)

import numpy as np
import mujoco
from mujoco_playground import registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Go2JoystickFlatTerrain")
    ap.add_argument("--rollout", default="/mnt/c/Users/hapos/Desktop/thesis/assets/go2_rollout.npz")
    ap.add_argument("--outdir", default="/mnt/c/Users/hapos/Desktop/thesis/archive/D2_go2_rl_go2model")
    args = ap.parse_args()

    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.rollout)
    qpos, act_vx, cmd_vx, heights = data["qpos"], data["act_vx"], data["cmd_vx"], data["heights"]

    env = registry.load(args.env)
    model = getattr(env, "mj_model", None) or env._mj_model
    d = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=240, width=320)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 1.8, 130, -20

    frames = []
    for t in range(0, len(qpos), 7):          # subsample for a compact GIF
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(model, d)
        cam.lookat[:] = d.qpos[:3]
        renderer.update_scene(d, camera=cam)
        frames.append(Image.fromarray(renderer.render()))
    gif = outdir / "walk.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=60, loop=0)

    dt = float(model.opt.timestep) * 10        # control decimation ~10
    t = np.arange(len(act_vx)) * dt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    a1.plot(t, act_vx, color="#1950a0", lw=1.6, label="actual vx")
    a1.plot(t, cmd_vx, color="#888", ls="--", lw=1.3, label="commanded vx")
    a1.set_ylabel("forward speed (m/s)"); a1.legend(frameon=False, fontsize=9)
    a1.set_title("Go2 RL policy — velocity tracking + stability (MuJoCo, full dynamics)")
    a2.plot(t, heights, color="#1f9d3a", lw=1.6)
    a2.axhspan(0.22, 0.30, color="#2e7d32", alpha=0.12, label="upright band")
    a2.axhline(0.15, color="#c0392b", ls="--", lw=1, label="fall")
    a2.set_xlabel("time (s)"); a2.set_ylabel("base height (m)"); a2.legend(frameon=False, fontsize=9)
    fig.savefig(outdir / "fig_rl_tracking.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {gif} ({len(frames)} frames) + fig_rl_tracking.png")


if __name__ == "__main__":
    main()
