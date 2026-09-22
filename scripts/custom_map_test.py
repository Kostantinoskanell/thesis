"""Test all 6 controllers on the fixed custom map (wall + forced right-turn
detour through the dead-beam sector + one patrolling mover), under the same
sensor-dropout shift M5 studies. Since the layout is now fixed, the only
seed-to-seed variation is controller stochasticity (spike-encoding noise,
plasticity evolution) -- a cleaner instrument than the random procedural map.
"""
import sys, os
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
from collections import Counter
import numpy as np

from custom_map import apply_custom_map, make_custom_env
from m6_energy import build, MLP_CONTROLLERS, SNN_CONTROLLERS

import concurrent.futures

OUT = Path("archive/custom_map_demo")
CONTROLLERS = MLP_CONTROLLERS + SNN_CONTROLLERS


def task(name, seed, n_eps, dropout):
    env = make_custom_env(shift_type="sensor", shift_time_s=0.5,
                          sensor_dropout_frac=dropout, sensor_dropout_start=8)
    ctrl = build(name, seed)
    outcomes = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed * 100 + ep)
        apply_custom_map(env)
        obs = env._observation()
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
    seeds = list(range(6000, 6005))   # 5 controller-seeds
    n_eps = 20                        # x20 episodes each = 100 per controller
    dropout = 0.20
    tasks = [(n, s, n_eps, dropout) for n in CONTROLLERS for s in seeds]
    print(f"custom-map test: {len(tasks)} tasks x {n_eps} eps...", flush=True)

    res = {n: [] for n in CONTROLLERS}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(task, *t): t for t in tasks}
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            name, seed, outcomes = f.result()
            res[name].extend(outcomes)
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)

    lines = ["# Custom map test: forced detour through the dead-beam sector + one mover\n",
             f"5 controller-seeds x {n_eps} eps = {5*n_eps} episodes/controller, "
             f"dropout={dropout}, fixed layout (see scripts/custom_map.py).\n",
             "| controller | success | collision(static) | collision(dynamic) | fell | timeout |",
             "|---|---|---|---|---|---|"]
    for name in CONTROLLERS:
        oc = res[name]
        n = len(oc)
        c = Counter(oc)
        lines.append(f"| {name} | {c.get('reached',0)/n:.1%} | "
                     f"{c.get('collision_static',0)/n:.1%} | "
                     f"{c.get('collision_dynamic',0)/n:.1%} | "
                     f"{c.get('fell',0)/n:.1%} | {c.get('timeout',0)/n:.1%} |")

    import scipy.stats
    rs = np.array([1 if o == "reached" else 0 for o in res["R-STDP SNN"]])
    fr = np.array([1 if o == "reached" else 0 for o in res["Frozen SNN"]])
    # per-seed means for a fair Welch t-test (n=5 seeds, not n=100 episodes)
    def per_seed(name):
        oc = res[name]
        per_ep = n_eps
        return np.array([np.mean([1 if o == "reached" else 0 for o in oc[i*per_ep:(i+1)*per_ep]])
                         for i in range(len(seeds))])
    rs_seed = per_seed("R-STDP SNN"); fr_seed = per_seed("Frozen SNN")
    t, p = scipy.stats.ttest_ind(rs_seed, fr_seed, equal_var=False)
    lines.append(f"\nR-STDP vs Frozen SNN (per-seed means, Welch t-test): "
                 f"{rs_seed.mean():.1%} vs {fr_seed.mean():.1%}, p={p:.3f}")

    (OUT / "custom_map_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT/'custom_map_results.md'}")


if __name__ == "__main__":
    main()
