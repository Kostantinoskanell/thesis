"""M4b ice deep-dive: render_terrain_videos.py found R-STDP gave ZERO improvement
over the frozen SNN on ice (mu=0.20, 20% both), unlike sand where it fully closed
the gap to the MLP. Two hypotheses, tested here:

  (a) SLOW, not incapable -- a 15-episode warm-up wasn't enough; try 40.
  (b) TOO SEVERE, not a wrong shift -- mu=0.20 (25pt drop) may leave too little
      floor to adapt from; try the gentler mu=0.28 (15pt drop, still "usable"
      per the friction sweep) with the standard 15-episode warm-up.

Each condition reruns its OWN frozen MLP/SNN baselines too (frozen performance
also depends on mu), so every comparison is apples-to-apples at that mu.

Run:  conda run -n nmc python scripts/ice_deep_dive.py --episodes 15
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nmc.controllers.privileged_expert import PrivilegedExpert, PrivilegedConfig
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from eval_mlp_go2 import run_episode
from pilot_m4 import make_rstdp, make_frozen_snn, make_frozen_mlp, run_block

OUT = ROOT / "archive" / "M4b_terrain_walk_compare"

CONDITIONS = [
    {"label": "ice mu=0.20, warmup=40 (slower?)", "mu": 0.20, "warmup": 40, "tag": "ice_mu020_warm40"},
    {"label": "ice mu=0.28, warmup=15 (gentler)", "mu": 0.28, "warmup": 15, "tag": "ice_mu028_warm15"},
]


def make_env(mu, episode_len_s=45.0):
    return Go2NavEnv(Go2NavConfig(shift_type="terrain", terrain_mode="ice",
                                 terrain_friction={"ice": mu},
                                 shift_time_s=0.5, episode_len_s=episode_len_s))


def save_gif(frames, path, fps=10):
    if not frames:
        print(f"  (no frames captured for {path.name}, skipping)")
        return
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"  wrote {path} ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--seed0", type=int, default=8000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    expert = PrivilegedExpert(PrivilegedConfig(inflate_m=0.9, fallback_inflations_m=(0.9, 0.7, 0.55)))

    for cond in CONDITIONS:
        mu, warmup, tag = cond["mu"], cond["warmup"], cond["tag"]
        print(f"\n=== {cond['label']} ===")

        env = make_env(mu)
        mlp = make_frozen_mlp()
        succ = 0
        for ep in range(args.episodes):
            info, *_ = run_episode(env, mlp, seed=args.seed0 + ep, expert=expert, record_frames=False)
            succ += int(info["reached"])
        env.close()
        print(f"  frozen MLP: {succ}/{args.episodes} = {succ/args.episodes:.0%}")

        env = make_env(mu)
        snn = make_frozen_snn(seed=7)
        succ = 0
        for ep in range(args.episodes):
            info, *_ = run_episode(env, snn, seed=args.seed0 + ep, expert=expert, record_frames=False)
            succ += int(info["reached"])
        env.close()
        print(f"  frozen SNN: {succ}/{args.episodes} = {succ/args.episodes:.0%}")

        env = make_env(mu)
        rstdp = make_rstdp(eta=0.05, seed=7, reward_mode="td",
                           plastic_layers=[0, -1], anchor=0.005)
        warm_seeds = list(range(args.seed0 + 500, args.seed0 + 500 + warmup))
        run_block(env, rstdp, warm_seeds, adapt=True)
        succ, gif_frames = 0, None
        for ep in range(args.episodes):
            info, _step, frames, _apl, _sp = run_episode(
                env, rstdp, seed=args.seed0 + 1000 + ep, expert=expert, record_frames=(ep == 0))
            succ += int(info["reached"])
            if ep == 0:
                gif_frames = frames
        env.close()
        print(f"  R-STDP SNN (post-{warmup}ep warmup): {succ}/{args.episodes} = {succ/args.episodes:.0%}")
        save_gif(gif_frames, OUT / f"snn_rstdp_{tag}.gif")


if __name__ == "__main__":
    main()
