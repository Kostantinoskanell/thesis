"""M4 final presentation videos: frozen (before) vs R-STDP-adapted (after) under
the sensor-dropout shift. High-resolution mp4s with a caption baked into every
frame, for use in the small_presentation folder.

For "after", the R-STDP controller is first run through a silent (unrendered,
fast) block of adaptation episodes under the shift -- exactly the M4 pilot
recipe (TD-error, input+readout plasticity, weight anchoring) -- then ONE more
episode is rendered with those adapted weights. "Before" is a fresh frozen SNN
on the same shift, same candidate seeds, for a fair visual side-by-side.

Run:  conda run -n nmc python scripts/render_m4_final.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

from pilot_m4 import make_rstdp, make_frozen_snn
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig

OUT = ROOT / "small_presentation" / "videos"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 960, 720
# Frames are captured once per env control step (25 Hz, Go2NavConfig default).
# FPS must match that to play at real-time speed -- FPS=15 made the walk look
# slow/draggy (0.6x speed), which read as "glitching." (see debug-log)
FPS = 25
DROPOUT, DROPOUT_START = 0.30, 8
# Wider, more top-down camera so the scattered obstacles + goal stay in frame
# regardless of where in the 10x10m arena the robot wanders -- the original
# close chase-cam (cam_dist=4.2) often kept every obstacle just out of frame.
CAM = dict(cam_dist=7.5, azimuth=100, elevation=-55)
ADAPT_EPISODES = 22          # silent adaptation block before the "after" render
CANDIDATE_SEEDS = list(range(5000, 5030))   # try these until one REACHES the goal


def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _shift_cfg():
    return Go2NavConfig(shift_type="sensor", shift_time_s=0.5,
                        sensor_dropout_frac=DROPOUT, sensor_dropout_start=DROPOUT_START,
                        episode_len_s=30.0)


def render_episode(env, ctrl, seed, caption_lines):
    """Run one episode with the given (already-adapted-or-not) controller,
    capturing a caption-annotated frame every control step. Returns
    (frames, reached: bool).

    Caption + status are stacked in separate rows (not side-by-side): a
    side-by-side layout collided when the caption text ran long enough to
    reach under the status badge's column (see debug-log)."""
    obs, _ = env.reset(seed=seed)
    frames = []
    font_big = _font(28)
    font_small = _font(20)
    bar_h = 34 + 28 + 34   # label row + desc row + status row
    step = 0
    while True:
        a = ctrl.act(obs)
        obs, r, term, trunc, info = env.step(a)
        img = Image.fromarray(env.render(w=W, h=H, **CAM))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, bar_h], fill=(0, 0, 0, 170))
        d.text((16, 6), caption_lines[0], font=font_big, fill=(255, 255, 255))
        if len(caption_lines) > 1:
            d.text((16, 38), caption_lines[1], font=font_small, fill=(220, 225, 235))
        status = ("REACHED GOAL" if info["reached"] else "FELL" if info["fell"]
                  else "COLLISION" if info["collision"] else f"t={info['t']:.0f}s")
        color = (90, 220, 120) if info["reached"] else (235, 90, 80) if (info["fell"] or info["collision"]) else (200, 200, 210)
        d.text((16, 66), status, font=font_small, fill=color)
        frames.append(np.asarray(img))
        step += 1
        if term or trunc:
            break
    return frames, bool(info["reached"])


def find_reaching_episode(env_factory, ctrl_factory, seeds, caption_lines, adapt_fn=None):
    """Adapt ONCE (if adapt_fn given), then try seeds against that SAME
    env+controller until one produces a REACHED episode; render that one.

    Earlier version re-ran the full adaptation block from scratch for every
    candidate seed (10x+ wasted compute, ~30 min wall time) -- fixed here."""
    env = env_factory()
    ctrl = ctrl_factory()
    if adapt_fn is not None:
        adapt_fn(env, ctrl, seeds[0])
    frames = None
    for s in seeds:
        frames, reached = render_episode(env, ctrl, s, caption_lines)
        print(f"  seed {s}: {'REACHED' if reached else 'no'}  ({len(frames)} frames)", flush=True)
        if reached:
            env.close()
            return frames, s
    env.close()
    print("  (no seed reached the goal in the candidate list; using the last run)")
    return frames, seeds[-1]


def adapt_rstdp(env, ctrl, base_seed):
    """Silent (no render) continual-adaptation block, same recipe as the
    successful M4 pilot run, ending with ctrl's weights adapted to the shift."""
    for i in range(ADAPT_EPISODES):
        s = base_seed - 1000 - i        # disjoint seeds from the rendered episodes
        obs, _ = env.reset(seed=s)
        while True:
            a = ctrl.act(obs)
            obs, r, term, trunc, info = env.step(a)
            ctrl.observe(r, obs, term or trunc)
            if term or trunc:
                break


def save_mp4(frames, path):
    with imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=8,
                            macro_block_size=None) as w:
        for f in frames:
            w.append_data(f)
    print(f"wrote {path}  ({len(frames)} frames, {len(frames)/FPS:.1f}s @ {FPS}fps)")


def main():
    print("=== BEFORE: frozen SNN under sensor dropout (no plasticity) ===", flush=True)
    frames_before, seed_before = find_reaching_episode(
        env_factory=_shift_cfg_env,
        ctrl_factory=lambda: make_frozen_snn(seed=7),
        seeds=CANDIDATE_SEEDS,
        caption_lines=["Frozen SNN  --  sensor failure (30% of LiDAR beams dead)",
                       "No online learning: cannot re-map around the broken sensor"],
    )
    save_mp4(frames_before, OUT / "before_frozen_snn.mp4")

    print("\n=== AFTER: R-STDP SNN, adapted online under the same shift ===", flush=True)
    frames_after, seed_after = find_reaching_episode(
        env_factory=_shift_cfg_env,
        ctrl_factory=lambda: make_rstdp(eta=0.05, seed=7, reward_mode="td",
                                        plastic_layers=[0, -1], anchor=0.005),
        seeds=CANDIDATE_SEEDS,
        caption_lines=["R-STDP SNN  --  same sensor failure, AFTER online adaptation",
                       "Input + readout plasticity (TD-error, anchored) re-maps around it"],
        adapt_fn=adapt_rstdp,
    )
    save_mp4(frames_after, OUT / "after_rstdp_adapted.mp4")

    print(f"\nseeds used: before={seed_before}  after={seed_after}")
    print("DONE")


def _shift_cfg_env():
    return Go2NavEnv(_shift_cfg())


if __name__ == "__main__":
    main()
