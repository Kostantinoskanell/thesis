"""M4b terrain walk comparison: MLP vs frozen-SNN vs R-STDP-SNN ("memristor")
walking on ice/sand, with GIFs + a success-rate graph.

Reuses pilot_m4.py's controller builders and eval_mlp_go2.py's rollout helper
so this is apples-to-apples with the existing M2-M4 metric suite, just applied
to the terrain shift instead of sensor dropout, with video capture added.

For R-STDP, a short warm-up block (continual adaptation, no recording) runs
first so the recorded/measured episodes show its POST-adaptation behavior on
this terrain, not its naive first reaction -- mirrors M4's "compare" protocol.

Run:  conda run -n nmc python scripts/render_terrain_videos.py --episodes 15 --warmup 15
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

import numpy as np

from nmc.controllers.privileged_expert import PrivilegedExpert, PrivilegedConfig
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from eval_mlp_go2 import run_episode
from pilot_m4 import make_rstdp, make_frozen_snn, make_frozen_mlp, run_block

OUT = ROOT / "archive" / "M4b_terrain_walk_compare"


def make_env(terrain_mode, mu_override=None, episode_len_s=45.0):
    kwargs = dict(shift_type="terrain", terrain_mode=terrain_mode,
                 shift_time_s=0.5, episode_len_s=episode_len_s)
    if mu_override is not None:
        kwargs["terrain_friction"] = {terrain_mode: mu_override}
    return Go2NavEnv(Go2NavConfig(**kwargs))


def save_gif(frames, path, fps=10):
    if not frames:
        print(f"  (no frames captured for {path.name}, skipping)")
        return
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"  wrote {path} ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=15, help="episodes/controller for the success bar chart")
    ap.add_argument("--warmup", type=int, default=15, help="R-STDP continual-adaptation episodes before recording")
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--ice-mu", type=float, default=0.08)
    ap.add_argument("--sand-mu", type=float, default=1.6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    mu_by_mode = {"ice": args.ice_mu, "sand": args.sand_mu}
    expert = PrivilegedExpert(PrivilegedConfig(inflate_m=0.9, fallback_inflations_m=(0.9, 0.7, 0.55)))

    all_rates = {}   # {(controller, terrain): success_rate}
    for terrain in ("ice", "sand"):
        mu = mu_by_mode[terrain]
        print(f"\n=== terrain: {terrain} (mu={mu}) ===")

        # -- frozen MLP: independent episodes, no adaptation --------------------
        env = make_env(terrain, mu)
        mlp = make_frozen_mlp()
        succ, gif_frames = 0, None
        for ep in range(args.episodes):
            info, _step, frames, _apl, _sp = run_episode(env, mlp, seed=args.seed0 + ep, expert=expert,
                                                         record_frames=(ep == 0))
            succ += int(info["reached"])
            if ep == 0:
                gif_frames = frames
        env.close()
        rate = succ / args.episodes
        all_rates[("frozen MLP", terrain)] = rate
        print(f"  frozen MLP: {succ}/{args.episodes} = {rate:.0%}")
        save_gif(gif_frames, OUT / f"mlp_{terrain}.gif")

        # -- frozen SNN: independent episodes, no adaptation ---------------------
        env = make_env(terrain, mu)
        snn = make_frozen_snn(seed=7)
        succ, gif_frames = 0, None
        for ep in range(args.episodes):
            info, _step, frames, _apl, _sp = run_episode(env, snn, seed=args.seed0 + ep, expert=expert,
                                                         record_frames=(ep == 0))
            succ += int(info["reached"])
            if ep == 0:
                gif_frames = frames
        env.close()
        rate = succ / args.episodes
        all_rates[("frozen SNN", terrain)] = rate
        print(f"  frozen SNN: {succ}/{args.episodes} = {rate:.0%}")
        save_gif(gif_frames, OUT / f"snn_frozen_{terrain}.gif")

        # -- R-STDP SNN ("memristor"): warm up (continual), THEN measure/record --
        env = make_env(terrain, mu)
        rstdp = make_rstdp(eta=0.05, seed=7, reward_mode="td",
                           plastic_layers=[0, -1], anchor=0.005)
        warm_seeds = list(range(args.seed0 + 500, args.seed0 + 500 + args.warmup))
        run_block(env, rstdp, warm_seeds, adapt=True)   # no recording, just adapts
        succ, gif_frames = 0, None
        for ep in range(args.episodes):
            info, _step, frames, _apl, _sp = run_episode(env, rstdp, seed=args.seed0 + 1000 + ep, expert=expert,
                                                         record_frames=(ep == 0))
            succ += int(info["reached"])
            if ep == 0:
                gif_frames = frames
        env.close()
        rate = succ / args.episodes
        all_rates[("R-STDP SNN", terrain)] = rate
        print(f"  R-STDP SNN (post-{args.warmup}ep warmup): {succ}/{args.episodes} = {rate:.0%}")
        save_gif(gif_frames, OUT / f"snn_rstdp_{terrain}.gif")

    # -- comparison bar chart -------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    controllers = ["frozen MLP", "frozen SNN", "R-STDP SNN"]
    terrains = ["ice", "sand"]
    x = np.arange(len(controllers)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"ice": "#5aa9e6", "sand": "#c9a35c"}
    for i, terrain in enumerate(terrains):
        vals = [all_rates[(c, terrain)] for c in controllers]
        ax.bar(x + (i - 0.5) * w, vals, w, color=colors[terrain], label=terrain)
    ax.set_xticks(x); ax.set_xticklabels(controllers)
    ax.set_ylabel("success rate"); ax.set_ylim(0, 1)
    ax.set_title("MLP vs frozen-SNN vs R-STDP-SNN walking on ice/sand")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig_path = OUT / "fig_terrain_walk_compare.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nwrote {fig_path}")


if __name__ == "__main__":
    main()
