"""M1b: run the scripted expert, log imitation demonstrations, report success.

Collects on the BASE distribution (shift disabled) so the frozen MLP is trained
on the pre-shift task; the mid-episode shift is a *test-time* perturbation used
later in the recovery experiment, not part of the training data.

Run:  conda run -n nmc python scripts/collect_expert.py --episodes 60 --out data/imitation.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nmc.controllers.expert import ScriptedExpert, ExpertConfig  # noqa: E402
from nmc.controllers.privileged_expert import PrivilegedExpert  # noqa: E402
from nmc.envs.nav_env import NavEnv, NavConfig  # noqa: E402


def run_episode(env, expert, seed):
    obs, _ = env.reset(seed=seed)
    obs_log, act_log = [], []
    collided = reached = False
    kind = None
    steps = 0
    while True:
        a = expert.act(obs, env)
        obs_log.append(obs.copy())
        act_log.append(a)
        obs, _, terminated, truncated, info = env.step(a)
        steps += 1
        if info["collision"]:
            collided = True
            kind = info["collision_kind"]
        reached = reached or info["reached"]
        if terminated or truncated:
            break
    success = reached and not collided
    return (np.array(obs_log, dtype=np.float32), np.array(act_log, dtype=np.int64),
            success, steps, collided, reached, kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--out", type=str, default="data/imitation.npz")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--n-static", type=int, default=8)
    ap.add_argument("--n-dynamic", type=int, default=3)
    ap.add_argument("--expert", choices=["astar", "reactive"], default="astar")
    args = ap.parse_args()

    # Shift disabled for demonstration collection (base distribution only).
    cfg = NavConfig(shift_time_s=1e9, episode_len_s=45.0,
                    n_static_obstacles=args.n_static, n_dynamic_obstacles=args.n_dynamic)
    env = NavEnv(cfg)
    expert = PrivilegedExpert() if args.expert == "astar" else ScriptedExpert(ExpertConfig())

    X, Y = [], []
    successes = collided_n = timeout_n = 0
    coll_static = coll_dynamic = 0
    total_steps = 0
    for ep in range(args.episodes):
        obs_log, act_log, success, steps, collided, reached, kind = run_episode(env, expert, seed=args.seed + ep)
        total_steps += steps
        if success:
            successes += 1
            X.append(obs_log)   # keep only successful demonstrations for imitation
            Y.append(act_log)
        elif collided:
            collided_n += 1
            if kind == "dynamic":
                coll_dynamic += 1
            else:
                coll_static += 1
        else:
            timeout_n += 1      # ran out of time without reaching (and no collision)
    env.close()

    rate = successes / args.episodes
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if X:
        Xall = np.concatenate(X)
        Yall = np.concatenate(Y)
        np.savez_compressed(out, obs=Xall, action=Yall, success_rate=rate)
        n_demo = len(Xall)
    else:
        n_demo = 0

    print(f"episodes={args.episodes}  success={successes} ({rate:.0%})")
    print(f"failures: collided={collided_n} (static={coll_static}, dynamic={coll_dynamic})"
          f"  timed-out(no reach)={timeout_n}")
    print(f"avg steps/episode={total_steps / args.episodes:.0f}")
    print(f"imitation demos (successful-episode steps)={n_demo}")
    print(f"saved -> {args.out}" if X else "no successful episodes; nothing saved")
    # M1b's deliverable is a set of CLEAN imitation demonstrations (drawn only from
    # collision-free successful episodes), not a teacher success rate. The residual
    # dynamic-obstacle failures are hard for a discrete-action, speed-limited robot
    # and are simply discarded. Bar: enough clean demo steps to train the MLP.
    ok = n_demo >= 5000
    print("\nM1b:", "PASS" if ok else f"NEED MORE DEMOS (have {n_demo}, want >=5000)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
