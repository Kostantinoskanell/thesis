"""M1b-on-dynamics: A* teacher demos on the Go2 nav env (D3).

Same protocol as collect_expert.py (kinematic): collect on the BASE distribution
(shift disabled) so the frozen student is trained pre-shift; keep only clean
(collision-free, no-fall, reached) episodes. New failure mode vs kinematic: falls.

Run:  conda run -n nmc python scripts/collect_expert_go2.py --episodes 60 --out data/imitation_go2.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nmc.controllers.privileged_expert import PrivilegedExpert, PrivilegedConfig  # noqa: E402
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig  # noqa: E402


def run_episode(env, expert, seed):
    obs, _ = env.reset(seed=seed)
    obs_log, act_log = [], []
    collided = reached = fell = False
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
        fell = fell or info["fell"]
        reached = reached or info["reached"]
        if terminated or truncated:
            break
    success = reached and not collided and not fell
    return (np.array(obs_log, dtype=np.float32), np.array(act_log, dtype=np.int64),
            success, steps, collided, reached, fell, kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--out", type=str, default="data/imitation_go2.npz")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    # Shift disabled for demonstration collection (base distribution only).
    cfg = Go2NavConfig(shift_time_s=1e9, episode_len_s=60.0)
    env = Go2NavEnv(cfg)
    # Inflation for the Go2 footprint + progressive fallback (floor 0.55 -- the
    # trotting robot needs real margin; 0.45 collided; see debug-log).
    expert = PrivilegedExpert(PrivilegedConfig(
        inflate_m=0.9, fallback_inflations_m=(0.9, 0.7, 0.55)))

    X, Y = [], []
    successes = collided_n = timeout_n = fell_n = 0
    coll_static = coll_dynamic = 0
    total_steps = 0
    t0 = time.time()
    for ep in range(args.episodes):
        (obs_log, act_log, success, steps, collided,
         reached, fell, kind) = run_episode(env, expert, seed=args.seed + ep)
        total_steps += steps
        if success:
            successes += 1
            X.append(obs_log)
            Y.append(act_log)
        elif fell:
            fell_n += 1
        elif collided:
            collided_n += 1
            if kind == "dynamic":
                coll_dynamic += 1
            else:
                coll_static += 1
        else:
            timeout_n += 1
        print(f"  ep {ep:3d}: {'OK  ' if success else 'FELL' if fell else 'COLL' if collided else 'TIME'}"
              f"  steps={steps}", flush=True)
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

    print(f"\nepisodes={args.episodes}  success={successes} ({rate:.0%})  wall={time.time()-t0:.0f}s")
    print(f"failures: collided={collided_n} (static={coll_static}, dynamic={coll_dynamic})"
          f"  fell={fell_n}  timed-out={timeout_n}")
    print(f"avg steps/episode={total_steps / args.episodes:.0f}")
    print(f"imitation demos (successful-episode steps)={n_demo}")
    print(f"saved -> {args.out}" if X else "no successful episodes; nothing saved")
    ok = n_demo >= 5000
    print("\nD3 demos:", "PASS" if ok else f"NEED MORE (have {n_demo}, want >=5000)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
