"""Walk the dynamic MuJoCo Go2 (CPG trot) and render a GIF + tracking graph.

Interim locomotion layer (CPG-through-PD) on the real Go2 with full dynamics.
Verifies forward walking + turn while staying upright.

Run:  conda run -n nmc python scripts/render_go2_walk.py
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

    outdir = ROOT / "archive" / "D2_go2_walk"
    outdir.mkdir(parents=True, exist_ok=True)

    robot = Go2MuJoCo(Go2Config())
    robot.reset()
    start = robot.base_state()["pos"][:2].copy()
    yaw0 = robot.base_state()["yaw"]

    log = {"t": [], "z": [], "v": [], "vx_cmd": [], "omega": [], "omega_cmd": []}
    frames = []
    schedule = [(0.6, 0.0, 200), (0.5, 0.8, 150)]   # forward, then forward+turn-left
    tick = 0
    for vx, omega, nticks in schedule:
        robot.set_command(vx=vx, omega=omega)
        for _ in range(nticks):
            st = robot.walk_step()
            log["t"].append(tick * robot.dt_ctrl)
            log["z"].append(st["height"]); log["v"].append(st["v"])
            log["vx_cmd"].append(vx)
            wz = robot.data.qvel[5]
            log["omega"].append(float(wz)); log["omega_cmd"].append(omega)
            if tick % 8 == 0:
                frames.append(robot.render(w=340, h=260))
            tick += 1

    end = robot.base_state()
    disp = end["pos"][:2] - start
    dyaw = np.degrees(np.arctan2(np.sin(end["yaw"] - yaw0), np.cos(end["yaw"] - yaw0)))

    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(outdir / "walk.gif", save_all=True, append_images=imgs[1:], duration=80, loop=0)

    t = np.array(log["t"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    a1.plot(t, log["z"], color="#1950a0", lw=1.6)
    a1.axhspan(0.24, 0.34, color="#2e7d32", alpha=0.12, label="upright band")
    a1.axhline(0.15, color="#c0392b", ls="--", lw=1, label="collapse")
    a1.set_ylabel("base height (m)"); a1.legend(frameon=False, fontsize=8)
    a1.set_title("MuJoCo Go2 CPG trot — stability + command tracking (full dynamics)")
    a2.plot(t, log["v"], color="#1950a0", lw=1.4, label="actual speed")
    a2.plot(t, log["vx_cmd"], color="#888", ls="--", lw=1.1, label="cmd vx")
    a2.plot(t, log["omega"], color="#e08a1e", lw=1.4, label="actual ω")
    a2.plot(t, log["omega_cmd"], color="#e0b080", ls="--", lw=1.1, label="cmd ω")
    a2.set_xlabel("time (s)"); a2.set_ylabel("speed / ω"); a2.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(outdir / "fig_walk_tracking.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    robot.close()

    walked = np.linalg.norm(disp) > 0.5
    fell = end["height"] < 0.18
    print(f"net displacement dx={disp[0]:+.2f} dy={disp[1]:+.2f} |d|={np.linalg.norm(disp):.2f} m")
    print(f"heading change {dyaw:+.0f} deg   final height z={end['height']:.3f}")
    print(f"walked>0.5m: {walked}   fell: {fell}   frames={len(frames)}")
    print("wrote archive/D2_go2_walk/{walk.gif, fig_walk_tracking.png}")
    print("Go2 CPG WALK:", "PASS" if (walked and not fell) else "NEEDS TUNING")
    return 0 if (walked and not fell) else 1


if __name__ == "__main__":
    raise SystemExit(main())
