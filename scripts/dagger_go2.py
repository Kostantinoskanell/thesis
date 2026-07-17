"""M2: DAgger iterations to fix behavior-cloning covariate shift.

The BC student hit 23% closed-loop success at 86.5% step accuracy -- the classic
compounding-error failure: the student drifts into states absent from the
teacher's demonstrations and acts badly there. DAgger fixes exactly this: roll
out the STUDENT (so we visit the student's own state distribution), label every
visited state with the A* teacher's action, aggregate into the dataset, retrain.

Run:  conda run -n nmc python scripts/dagger_go2.py --iters 2 --episodes 40
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from nmc.controllers.mlp import MLPPolicy
from nmc.controllers.privileged_expert import PrivilegedExpert, PrivilegedConfig
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig

ACTION_NAMES = ["fwd", "left", "right", "brake"]


def retrain(X, Y, epochs=60, batch=512, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    counts = np.bincount(Y, minlength=4)
    w = torch.as_tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
    w = w / w.mean()
    policy = MLPPolicy(obs_dim=X.shape[1], n_actions=4)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for k in range(0, len(Xt), batch):
            b = perm[k:k + batch]
            logits, _ = policy(Xt[b])
            loss = F.cross_entropy(logits, Yt[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    policy.eval()
    with torch.no_grad():
        acc = float((policy(Xt)[0].argmax(-1) == Yt).float().mean())
    return policy, acc


def student_rollout_with_teacher_labels(env, policy, expert, seed):
    """Student drives; teacher labels every visited state."""
    obs, _ = env.reset(seed=seed)
    X, Y = [], []
    while True:
        teacher_a = expert.act(obs, env)         # label for THIS state
        with torch.no_grad():
            logits, _ = policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        student_a = int(logits.argmax(-1))
        X.append(obs.copy()); Y.append(teacher_a)
        obs, _, term, trunc, info = env.step(student_a)   # STUDENT acts
        if term or trunc:
            return np.array(X, np.float32), np.array(Y, np.int64), info


def evaluate(env, policy, episodes, seed0):
    hits = {"reached": 0, "collided": 0, "fell": 0, "timeout": 0}
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        while True:
            with torch.no_grad():
                logits, _ = policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            obs, _, term, trunc, info = env.step(int(logits.argmax(-1)))
            if term or trunc:
                break
        k = ("reached" if info["reached"] else "fell" if info["fell"]
             else "collided" if info["collision"] else "timeout")
        hits[k] += 1
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=40, help="student rollouts per iter")
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--data", default="data/imitation_go2.npz")
    ap.add_argument("--out-data", default="data/imitation_go2_dagger.npz")
    ap.add_argument("--out-model", default="assets/mlp_frozen_go2.pt")
    args = ap.parse_args()

    z = np.load(ROOT / args.data)
    X, Y = z["obs"], z["action"]
    print(f"base dataset: {len(X)} steps")

    ckpt = torch.load(ROOT / args.out_model, weights_only=True)
    policy = MLPPolicy(ckpt["obs_dim"], ckpt["n_actions"])
    policy.load_state_dict(ckpt["state_dict"]); policy.eval()

    env = Go2NavEnv(Go2NavConfig(shift_time_s=1e9, episode_len_s=60.0))
    expert = PrivilegedExpert(PrivilegedConfig(
        inflate_m=0.9, fallback_inflations_m=(0.9, 0.7, 0.55)))

    history = []
    for it in range(args.iters):
        t0 = time.time()
        newX, newY, succ = [], [], 0
        for ep in range(args.episodes):
            x, y, info = student_rollout_with_teacher_labels(
                env, policy, expert, seed=3000 + it * 1000 + ep)
            newX.append(x); newY.append(y)
            succ += int(info["reached"])
        X = np.concatenate([X] + newX)
        Y = np.concatenate([Y] + newY)
        policy, acc = retrain(X, Y)
        print(f"iter {it}: +{sum(len(x) for x in newX)} labeled steps "
              f"(student success during rollouts {succ}/{args.episodes}), "
              f"dataset {len(X)}, retrain acc {acc:.3f}, wall {time.time()-t0:.0f}s", flush=True)
        history.append(succ / args.episodes)

    print("final eval on held-out seeds:", flush=True)
    hits = evaluate(env, policy, args.eval_episodes, seed0=2000)
    env.close()
    n = args.eval_episodes
    print(f"frozen MLP after DAgger: success {hits['reached']}/{n} ({hits['reached']/n:.0%})  "
          f"collided={hits['collided']} fell={hits['fell']} timeout={hits['timeout']}")

    np.savez_compressed(ROOT / args.out_data, obs=X, action=Y)
    torch.save({"state_dict": policy.state_dict(), "obs_dim": X.shape[1],
                "n_actions": 4, "hidden": [512, 512]}, ROOT / args.out_model)
    print(f"saved dataset -> {args.out_data}  model -> {args.out_model}")
    ok = hits["reached"] >= max(3, int(0.3 * n))
    print("M2 frozen-MLP:", "PASS" if ok else "STILL BELOW BAR")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
