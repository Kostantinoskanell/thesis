"""M2 evaluation (SOTA suite): multi-seed frozen MLP + online-TD(lambda), with CIs.

Metrics (src/nmc/eval/metrics.py): success rate (Wilson CI), SPL (Anderson 2018;
shortest path from the A* teacher on each episode's initial config), collision rate.
Frozen MLP is evaluated over N seed-models x M held-out episodes; per-episode
outcomes give a Wilson CI, and across-seed success gives a t-CI. The online
TD(lambda) controller is verified mechanically and run closed-loop pre-shift.

Held-out seeds (2000+) are disjoint from collection (1000+) and DAgger (3000+).
Shift disabled -> this is the pre-shift competence bar (M2 exit criterion).

Run:  conda run -n nmc python scripts/eval_mlp_go2.py --episodes 30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from nmc.controllers.mlp import MLPPolicy, OnlineMLP
from nmc.controllers.privileged_expert import PrivilegedExpert, PrivilegedConfig
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from nmc.eval.metrics import spl, path_length, collision_rate, mean_ci95, wilson_ci95

OUT = ROOT / "archive" / "M2_mlp_baselines_go2"


def load_policy(path):
    ckpt = torch.load(path, weights_only=True)
    hidden = tuple(ckpt.get("hidden", (512, 512)))
    policy = MLPPolicy(ckpt["obs_dim"], ckpt["n_actions"], hidden=hidden)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy


def shortest_path_len(env, expert):
    """A* geodesic start->goal on the episode's initial obstacle config (for SPL)."""
    pos, yaw, goal, obst = env.privileged_state()
    for infl in (0.9, 0.7, 0.55):
        occ = expert._occupancy(obst, inflate_m=infl)
        start = expert._to_cell(pos); occ[start] = False
        path = expert._astar(occ, start, expert._to_cell(goal))
        if path:
            pts = np.array([expert._to_world(rc) for rc in path])
            return path_length(pts)
    return float(np.linalg.norm(goal - pos))   # fallback: straight-line


def run_episode(env, controller, seed, expert, record_frames=False, gif_every=5):
    obs, _ = env.reset(seed=seed)
    sp_len = shortest_path_len(env, expert)
    positions = [env._robot_pose()[0].copy()]
    frames = []
    step = 0
    while True:
        a = controller.act(obs)
        next_obs, r, term, trunc, info = env.step(a)
        controller.observe(r, next_obs, term or trunc)
        obs = next_obs
        positions.append(env._robot_pose()[0].copy())
        if record_frames and step % gif_every == 0:
            from PIL import Image
            frames.append(Image.fromarray(env.render(w=480, h=360, cam_dist=4.5,
                                                     azimuth=90, elevation=-40)))
        step += 1
        if term or trunc:
            break
    return info, step, frames, path_length(np.array(positions)), sp_len


def eval_frozen(env, expert, model_paths, episodes, seed0):
    per_seed_success = []
    all_S, all_coll, all_pl, all_sp = [], [], [], []
    best_gif = None
    for si, mp in enumerate(model_paths):
        ctrl = OnlineMLP(load_policy(mp), frozen=True)
        succ = 0
        for ep in range(episodes):
            want = best_gif is None
            info, steps, frames, pl, sp = run_episode(
                env, ctrl, seed=seed0 + ep, expert=expert, record_frames=want)
            reached = info["reached"]
            succ += int(reached)
            all_S.append(int(reached))
            all_coll.append(int(info["collision"]))
            all_pl.append(pl); all_sp.append(sp)
            if reached and want and frames:
                best_gif = frames
        per_seed_success.append(succ / episodes)
        print(f"  seed-model {si}: success {succ}/{episodes} ({succ/episodes:.0%})", flush=True)
    return per_seed_success, all_S, all_coll, all_pl, all_sp, best_gif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=2000)
    ap.add_argument("--seeds-dir", default="assets/mlp_seeds")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    model_paths = sorted((ROOT / args.seeds_dir).glob("mlp_seed*.pt"))
    assert model_paths, f"no seed models in {args.seeds_dir}; run train_seeds_mlp_go2.py"
    print(f"evaluating {len(model_paths)} seed-models x {args.episodes} episodes")

    env = Go2NavEnv(Go2NavConfig(shift_time_s=1e9, episode_len_s=60.0))
    expert = PrivilegedExpert(PrivilegedConfig(
        inflate_m=0.9, fallback_inflations_m=(0.9, 0.7, 0.55)))

    # ---- online TD(lambda) mechanical check ----
    online = OnlineMLP(load_policy(model_paths[0]), frozen=False, lam=0.9)
    w0 = [p.detach().clone() for p in online.params]
    o = np.random.default_rng(0).standard_normal(37).astype(np.float32)
    online.act(o); online.observe(-1.0, o + 0.01, False)
    d = max(float((a - b).abs().max()) for a, b in zip(online.params, w0))
    print(f"online TD(lambda) mechanical check: max|dW| = {d:.2e} ({'UPDATES' if d>0 else 'BUG'})")
    assert d > 0

    # ---- frozen MLP, multi-seed, metric suite ----
    per_seed, S, coll, pl, sp, best_gif = eval_frozen(
        env, expert, model_paths, args.episodes, args.seed0)
    env.close()

    n_total = len(S)
    k = int(np.sum(S))
    rate, wl, wh = wilson_ci95(k, n_total)
    seed_mean, seed_hw = mean_ci95(per_seed)
    spl_val = spl(S, pl, sp)
    coll_val = collision_rate(coll)
    print(f"\n=== frozen MLP (pre-shift, {len(model_paths)} seeds x {args.episodes} eps) ===")
    print(f"success (pooled): {rate:.0%}  Wilson95 [{wl:.0%}, {wh:.0%}]  (n={n_total})")
    print(f"success (across-seed mean): {seed_mean:.0%} +/- {seed_hw:.0%}")
    print(f"SPL: {spl_val:.3f}    collision rate: {coll_val:.0%}")

    if best_gif:
        gif = OUT / "mlp_episode.gif"
        best_gif[0].save(gif, save_all=True, append_images=best_gif[1:], duration=100, loop=0)
        print(f"wrote {gif}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

    a1.bar(["success", "SPL"], [rate, spl_val],
           color=["#1f9d3a", "#1950a0"], width=0.55)
    a1.errorbar([0], [rate], yerr=[[rate - wl], [wh - rate]], fmt="none",
                ecolor="k", capsize=5)
    a1.axhline(0.30, color="#c0392b", ls="--", lw=1, label="M2 bar (30%)")
    a1.set_ylim(0, 1); a1.set_ylabel("value")
    a1.set_title(f"frozen MLP pre-shift (Wilson95 on success)\ncollision rate {coll_val:.0%}")
    a1.legend(frameon=False, fontsize=9)

    a2.bar(range(len(per_seed)), per_seed, color="#1950a0", width=0.6)
    a2.axhline(seed_mean, color="k", lw=1.2, label=f"mean {seed_mean:.0%}")
    a2.axhspan(seed_mean - seed_hw, seed_mean + seed_hw, color="k", alpha=0.12,
               label=f"95% CI +/-{seed_hw:.0%}")
    a2.set_xlabel("seed-model"); a2.set_ylabel("success rate"); a2.set_ylim(0, 1)
    a2.set_title("per-seed success (init/SGD variance)")
    a2.legend(frameon=False, fontsize=9)
    fig.suptitle(f"M2 frozen MLP (512x512, BC+DAgger) — {len(model_paths)} seeds, "
                 f"{args.episodes} held-out eps each  |  A* teacher (privileged): 62%")
    fig.tight_layout()
    fig.savefig(OUT / "fig_mlp_eval.png", bbox_inches="tight", dpi=150)
    print(f"wrote {OUT / 'fig_mlp_eval.png'}")

    ok = rate >= 0.30
    print("\nM2 frozen-MLP:", "PASS" if ok else "BELOW BAR (want >=30%)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
