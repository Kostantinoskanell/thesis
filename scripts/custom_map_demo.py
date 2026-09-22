"""Render the custom map to verify it looks right before running tests on it."""
import sys, os
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
from PIL import Image
import numpy as np

from custom_map import apply_custom_map, make_custom_env
from m6_energy import build

OUT = Path("archive/custom_map_demo")
OUT.mkdir(parents=True, exist_ok=True)


def run(name, seed, dropout=0.20, tag=""):
    env = make_custom_env(shift_type="sensor", shift_time_s=0.5,
                          sensor_dropout_frac=dropout, sensor_dropout_start=8)
    obs, _ = env.reset(seed=seed)
    apply_custom_map(env)
    obs = env._observation()
    ctrl = build(name, seed)
    frames, positions = [], [env._robot_pose()[0].copy()]
    step = 0
    while True:
        a = ctrl.act(obs)
        nobs, r, term, trunc, info = env.step(a)
        ctrl.observe(r, nobs, term or trunc)
        obs = nobs
        positions.append(env._robot_pose()[0].copy())
        if step % 5 == 0:
            frames.append(Image.fromarray(env.render(w=480, h=360, cam_dist=7.0,
                                                     azimuth=90, elevation=-60)))
        step += 1
        if term or trunc:
            break
    verdict = ("REACHED" if info["reached"] else "FELL" if info["fell"]
              else f"COLLIDED({info['collision_kind']})" if info["collision"] else "TIMEOUT")
    print(f"{name} seed={seed}: {verdict} at t={info['t']:.1f}s steps={step}")
    if frames:
        path = OUT / f"custom_map_{tag or name.lower().replace(' ', '_')}.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(f"  wrote {path} ({len(frames)} frames)")
    env.close()
    return info, np.array(positions)


if __name__ == "__main__":
    run("Frozen SNN", 6000, tag="frozen_snn")
    run("R-STDP SNN", 6000, tag="rstdp_snn")
    run("Frozen MLP", 6000, tag="frozen_mlp")
