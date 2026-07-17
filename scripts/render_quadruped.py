"""Render the CPG quadruped walking to a 3D GIF (Platform track P1 visual check).

Walks forward, then turns, following the robot with the camera. Prints net
displacement + heading change so we can verify it actually locomotes (not shuffle
in place or walk backwards) rather than trusting the video alone.

Run:  conda run -n nmc python scripts/render_quadruped.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nmc.platform.quadruped import QuadrupedLocomotion, GaitConfig  # noqa: E402


def grab_frame(p, target, w=460, h=360):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[target[0], target[1], 0.3],
        distance=1.7, yaw=40, pitch=-28, roll=0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=w / h, nearVal=0.1, farVal=100)
    img = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.reshape(img[2], (h, w, 4))[:, :, :3].astype(np.uint8)
    return rgb


def main():
    from PIL import Image
    outdir = ROOT / "archive" / "P1_quadruped"
    outdir.mkdir(parents=True, exist_ok=True)

    robot = QuadrupedLocomotion(GaitConfig())
    robot.reset()
    p = robot._p

    frames = []
    start = robot.base_state()["pos"].copy()
    yaw0 = robot.base_state()["yaw"]
    log = {"t": [], "z": [], "v": [], "vx_cmd": [], "omega": [], "omega_cmd": []}

    # Forward, then forward + turn-left, to exercise the full [vx, omega] interface.
    schedule = [(0.6, 0.0, 220), (0.5, 0.9, 200)]
    tick = 0
    for vx, omega, nticks in schedule:
        robot.set_command(vx=vx, omega=omega)
        for _ in range(nticks):
            st = robot.step()
            log["t"].append(tick * robot.dt)
            log["z"].append(st["pos"][2]); log["v"].append(st["v"])
            log["vx_cmd"].append(vx); log["omega"].append(st["omega"]); log["omega_cmd"].append(omega)
            if tick % 3 == 0:
                frames.append(grab_frame(p, st["pos"]))
            tick += 1

    end = robot.base_state()
    robot.close()

    disp = end["pos"][:2] - start[:2]
    dyaw = np.degrees(np.arctan2(np.sin(end["yaw"] - yaw0), np.cos(end["yaw"] - yaw0)))
    print(f"net displacement: dx={disp[0]:+.2f} dy={disp[1]:+.2f} "
          f"|d|={np.linalg.norm(disp):.2f} m")
    print(f"heading change: {dyaw:+.0f} deg    final base height z={end['pos'][2]:.3f}")
    fell = end["pos"][2] < 0.2
    walked = np.linalg.norm(disp) > 0.5
    print(f"walked>0.5m: {walked}   fell: {fell}")

    imgs = [Image.fromarray(f) for f in frames]
    out = outdir / "walk.gif"
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=90, loop=0)
    print(f"frames={len(frames)}  wrote {out.relative_to(ROOT)}")

    # Diagnostic graph (the second visualization way): stability + command tracking.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array(log["t"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    a1.plot(t, log["z"], color="#1950a0", lw=1.8)
    a1.axhspan(0.25, 0.32, color="#2e7d32", alpha=0.12, label="stable stance band")
    a1.axhline(0.2, color="#c0392b", ls="--", lw=1, label="fall threshold")
    a1.set_ylabel("base height z (m)"); a1.legend(frameon=False, fontsize=8)
    a1.set_title("P1 gait stability + command tracking (A1 CPG trot)")
    a2.plot(t, log["v"], color="#1950a0", lw=1.6, label="actual speed")
    a2.plot(t, log["vx_cmd"], color="#888", ls="--", lw=1.2, label="commanded vx")
    a2.plot(t, log["omega"], color="#e08a1e", lw=1.6, label="actual ω")
    a2.plot(t, log["omega_cmd"], color="#e0b080", ls="--", lw=1.2, label="commanded ω")
    a2.set_xlabel("time (s)"); a2.set_ylabel("speed / ω"); a2.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(outdir / "fig_gait_diagnostics.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", (outdir / "fig_gait_diagnostics.png").relative_to(ROOT))
    print("P1 walk:", "PASS" if (walked and not fell) else "NEEDS TUNING")
    return 0 if (walked and not fell) else 1


if __name__ == "__main__":
    raise SystemExit(main())
