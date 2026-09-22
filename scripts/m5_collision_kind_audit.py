"""How much of the M5 sensor-dropout failure rate is static-obstacle collision
(directly caused by the shift the experiment studies) vs dynamic-obstacle
collision (a roaming mover that has nothing to do with the shift)?

If dynamic collisions dominate, the environment is adding outcome noise that
has nothing to do with the phenomenon under study -- a legitimate reason to
build a "cleaner" test map (fewer/no dynamic obstacles) for future work.
"""
import sys, os
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from m6_energy import build

import concurrent.futures

def shift_cfg(dropout=0.20, start=8):
    return Go2NavConfig(shift_type="sensor", shift_time_s=0.5,
                        sensor_dropout_frac=dropout, sensor_dropout_start=start,
                        episode_len_s=60.0)

def task(name, seed, n_eps):
    env = Go2NavEnv(shift_cfg())
    ctrl = build(name, seed)
    outcomes = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed * 100 + ep)
        while True:
            a = ctrl.act(obs)
            nobs, r, term, trunc, info = env.step(a)
            ctrl.observe(r, nobs, term or trunc)
            obs = nobs
            if term or trunc:
                break
        if info["reached"]:
            outcomes.append("reached")
        elif info["fell"]:
            outcomes.append("fell")
        elif info["collision"]:
            outcomes.append(f"collision_{info['collision_kind']}")
        else:
            outcomes.append("timeout")
    env.close()
    return name, seed, outcomes

def main():
    names = ["Frozen SNN", "R-STDP SNN", "Frozen MLP"]
    seeds = list(range(6000, 6008))
    n_eps = 20
    tasks = [(n, s, n_eps) for n in names for s in seeds]
    print(f"{len(tasks)} tasks x {n_eps} eps...", flush=True)
    res = {n: [] for n in names}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(task, *t): t for t in tasks}
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            name, seed, outcomes = f.result()
            res[name].extend(outcomes)
            if (i + 1) % 6 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)

    for name in names:
        oc = res[name]
        n = len(oc)
        from collections import Counter
        c = Counter(oc)
        print(f"\n{name} (n={n} episodes):")
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f"  {k}: {v} ({v/n:.1%})")
        static_share = c.get("collision_static", 0)
        dynamic_share = c.get("collision_dynamic", 0)
        total_coll = static_share + dynamic_share
        if total_coll:
            print(f"  -> of collisions specifically: {dynamic_share}/{total_coll} "
                  f"({dynamic_share/total_coll:.1%}) were with a MOVING obstacle")

if __name__ == "__main__":
    main()
