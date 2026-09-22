"""M5 Render: Save GIFs for R-STDP and TM-NORM controllers to verify behavior visually."""
import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from m5_full_comparison import make_rstdp, make_tmnorm, OUT

def record_gif(name, make_fn, seed=6000):
    print(f"Recording {name}...")
    shift_cfg = Go2NavConfig(shift_type="sensor", shift_time_s=0.5,
                             sensor_dropout_frac=0.30, episode_len_s=60.0)
    env = Go2NavEnv(shift_cfg)
    ctrl = make_fn(seed)
    
    obs, _ = env.reset(seed=seed)
    frames = []
    step = 0
    while True:
        a = ctrl.act(obs)
        nobs, r, term, trunc, info = env.step(a)
        ctrl.observe(r, nobs, term or trunc)
        obs = nobs
        if step % 5 == 0:
            frame = env.render(w=480, h=360, cam_dist=4.5, azimuth=90, elevation=-40)
            frames.append(Image.fromarray(frame))
        step += 1
        if term or trunc:
            break
            
    env.close()
    if frames:
        gif_path = OUT / f"{name.lower().replace(' ', '_')}.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(f"Saved {gif_path}")

if __name__ == "__main__":
    record_gif("TM-NORM SNN", make_tmnorm)
    record_gif("R-STDP SNN", make_rstdp)
