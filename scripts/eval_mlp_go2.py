"""M2 evaluation: frozen MLP (imitation) closed-loop on the dynamic Go2 nav env.

Held-out seeds (2000+, disjoint from collection's 1000+), shift disabled --
this measures PRE-shift competence, the M2 exit criterion ("frozen MLP reaches
goal pre-shift"). Also runs the mechanical check that the online-TD variant
actually updates its weights from environment reward alone (controller 2's
defining property), plus a short closed-loop online run.

Saves eval bar figure + one successful-episode GIF to archive/M2_mlp_baselines_go2/.

Run:  conda run -n nmc python scripts/eval_mlp_go2.py --episodes 30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch+matplotlib on Windows

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from nmc.controllers.mlp import MLPPolicy, OnlineMLP
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig

OUT = ROOT / "archive" / "M2_mlp_baselines_go2"


def load_policy(path):
    ckpt = torch.load(path, weights_only=True)
    policy = MLPPolicy(ckpt["obs_dim"], ckpt["n_actions"])
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy


def run_episode(env, controller, seed, record_frames=False, gif_every=5):
    obs, _ = env.reset(seed=seed)
    frames = []
    step = 0
    while True:
        a = controller.act(obs)
        next_obs, r, term, trunc, info = env.step(a)
        controller.observe(r, next_obs, term or trunc)
        obs = next_obs
        if record_frames and step % gif_every == 0:
            from PIL import Image
            frames.append(Image.fromarray(env.render(w=480, h=360, cam_dist=4.5,
                                                     azimuth=90, elevation=-40)))
        step += 1
        if term or trunc:
            break
    return info, step, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=2000)
    ap.add_argument("--model", default="assets/mlp_frozen_go2.pt")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1) mechanical check: online-TD updates weights from reward only ----
    policy = load_policy(ROOT / args.model)
    online = OnlineMLP(load_policy(ROOT / args.model), frozen=False)
    w0 = [p.detach().clone() for p in online.policy.parameters()]
    obs = np.random.default_rng(0).standard_normal(37).astype(np.float32)
    online.act(obs)
    online.observe(-1.0, obs + 0.01, False)     # one TD update from a reward
    delta = max(float((a - b).abs().max()) for a, b in
                zip([p.detach() for p in online.policy.parameters()], w0))
    print(f"online-TD mechanical check: max|dW| after one observe() = {delta:.2e} "
          f"({'UPDATES' if delta > 0 else 'FROZEN — BUG'})")
    assert delta > 0

    frozen_check = OnlineMLP(load_policy(ROOT / args.model), frozen=True)
    w0 = [p.detach().clone() for p in frozen_check.policy.parameters()]
    frozen_check.act(obs); frozen_check.observe(-1.0, obs, False)
    delta_f = max(float((a - b).abs().max()) for a, b in
                  zip([p.detach() for p in frozen_check.policy.parameters()], w0))
    print(f"frozen check: max|dW| = {delta_f:.2e} ({'ok' if delta_f == 0 else 'BUG'})")
    assert delta_f == 0

    # ---- 2) closed-loop eval: frozen MLP, pre-shift distribution ----
    env = Go2NavEnv(Go2NavConfig(shift_time_s=1e9, episode_len_s=60.0))
    frozen = OnlineMLP(policy, frozen=True)

    results = {"reached": 0, "collided": 0, "fell": 0, "timeout": 0}
    best_gif = None
    for ep in range(args.episodes):
        want_frames = best_gif is None    # record until first success captured
        info, steps, frames = run_episode(env, frozen, seed=args.seed0 + ep,
                                          record_frames=want_frames)
        if info["reached"]:
            results["reached"] += 1
            if want_frames and frames:
                best_gif = frames
        elif info["fell"]:
            results["fell"] += 1
        elif info["collision"]:
            results["collided"] += 1
        else:
            results["timeout"] += 1
        print(f"  ep {ep:3d}: {'REACHED' if info['reached'] else 'FELL' if info['fell'] else 'COLL' if info['collision'] else 'TIME'}"
              f" steps={steps}", flush=True)
    env.close()

    n = args.episodes
    rate = results["reached"] / n
    print(f"\nfrozen MLP pre-shift: success {results['reached']}/{n} ({rate:.0%})  "
          f"collided={results['collided']} fell={results['fell']} timeout={results['timeout']}")

    if best_gif:
        gif = OUT / "mlp_episode.gif"
        best_gif[0].save(gif, save_all=True, append_images=best_gif[1:], duration=100, loop=0)
        print(f"wrote {gif}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    keys = list(results)
    ax.bar(keys, [results[k] / n for k in keys], color=["#1f9d3a", "#c0392b", "#8e44ad", "#888"], width=0.6)
    ax.set_ylim(0, 1); ax.set_ylabel("fraction of episodes")
    ax.set_title(f"M2 frozen MLP (student, lidar-only) — pre-shift, {n} held-out episodes\n"
                 f"success {rate:.0%} (A* teacher w/ privileged info: 62%)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_mlp_eval.png", bbox_inches="tight", dpi=150)
    print(f"wrote {OUT / 'fig_mlp_eval.png'}")

    # M2 exit criterion: the frozen student reaches goals pre-shift.
    ok = results["reached"] >= max(3, int(0.3 * n))
    print("\nM2 frozen-MLP:", "PASS" if ok else "BELOW BAR (want >=30% success)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
