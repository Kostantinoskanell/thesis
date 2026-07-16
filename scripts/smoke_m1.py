"""M1 smoke test: run a full episode, verify the mid-episode shift fires.

Run in the nmc env:  conda run -n nmc python scripts/smoke_m1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nmc.envs.nav_env import NavEnv, NavConfig  # noqa: E402


def main():
    cfg = NavConfig(seed=1, gui=False, episode_len_s=60.0, shift_time_s=30.0)
    env = NavEnv(cfg)
    obs, _ = env.reset(seed=1)
    rng = np.random.default_rng(0)

    assert obs.shape == (env.obs_dim,), f"bad obs shape {obs.shape}"
    n_static_before = len(env.obstacles)
    n_dyn_before = len(env.dynamic)

    phase1_seen = phase2_seen = False
    shift_t = None
    n_static_after = n_dyn_after = -1
    collisions = 0
    reached = False
    steps = 0
    lidar_min = 1.0

    # Run ONE continuous 60 s episode. We deliberately ignore collision-
    # termination here: the point of the smoke test is to confirm the sim
    # clock advances to 60 s and the distribution shift fires at t=30 s, not
    # to survive (a random policy never would). Collisions are still counted.
    while True:
        a = int(rng.integers(env.n_actions))
        obs, reward, terminated, truncated, info = env.step(a)
        steps += 1
        lidar_min = min(lidar_min, float(obs[:cfg.n_lidar_beams].min()))
        if info["phase"] == 1:
            phase1_seen = True
        if info["phase"] == 2 and not phase2_seen:
            phase2_seen = True
            shift_t = info["t"]
            n_static_after = len(env.obstacles)
            n_dyn_after = len(env.dynamic)
        collisions += int(info["collision"])
        reached = reached or info["reached"]
        if truncated:
            break

    env.close()

    print(f"steps={steps}  sim_time={info['t']:.1f}s (dt={env.dt:.3f})")
    print(f"obs_dim={env.obs_dim}  lidar range seen: min_norm={lidar_min:.3f}")
    print(f"shift fired at t={shift_t:.1f}s  (target {cfg.shift_time_s}s)")
    print(f"static obstacles: {n_static_before} -> {n_static_after} (density doubled)")
    print(f"dynamic obstacles: {n_dyn_before} -> {n_dyn_after}")
    print(f"collisions logged={collisions}  goal reached at least once={reached}")

    ok = (phase1_seen and phase2_seen
          and abs(shift_t - cfg.shift_time_s) < 2 * env.dt
          and n_static_after == 2 * n_static_before
          and 0.0 <= lidar_min <= 1.0)
    print("\nM1 SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
