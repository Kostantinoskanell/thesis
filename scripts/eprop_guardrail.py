"""e-prop guardrail (mirrors pilot_m4.py's guardrail() for R-STDP): sweep eta
on the BASE (unshifted) distribution and confirm pre-shift performance
survives plasticity before trusting any shift-recovery number. Numerically
verified first (scripts/ was used ad hoc; see eprop_parity_check pattern in
the module docstrings) that EPropNavController's forward pass is bit-exact
vs the real PyTorch LIFNet with eta=0.

Run:  conda run -n nmc python scripts/eprop_guardrail.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np

from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from nmc.controllers.eprop_snn import EPropNavController

import concurrent.futures

OUT = Path("archive/D1_eprop")
CKPT = "assets/snn_seeds/snn_seed0.pt"


def run_block(eta, seed, n_eps=24):
    env = Go2NavEnv(Go2NavConfig(shift_time_s=1e9, episode_len_s=60.0))
    ctrl = EPropNavController(CKPT, seed=seed, eta=eta, anchor=0.005, plastic_layers=(0, -1))
    succ = []
    finite = True
    for ep in range(n_eps):
        obs, _ = env.reset(seed=4000 + ep)
        while True:
            a = ctrl.act(obs)
            nobs, r, term, trunc, info = env.step(a)
            ctrl.observe(r, nobs, term or trunc)
            obs = nobs
            if term or trunc:
                break
        succ.append(int(info["reached"]))
        for layer in ctrl.layers:
            if not np.all(np.isfinite(layer.W)):
                finite = False
    env.close()
    return eta, seed, float(np.mean(succ)), finite


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # extended downward: an ad hoc single-seed probe (0.01/0.05/0.1) showed
    # degradation even at the smallest value tested, with weight norms
    # growing smoothly (finite, not diverging) -- suggesting e-prop's
    # eligibility trace has a larger natural scale than R-STDP's (no small
    # hand-tuned amplitude constant like STDP's a_plus=0.008), so the same
    # eta produces a bigger update. Scan low enough to actually find a safe
    # point rather than defaulting to "least-bad of an insufficient range".
    etas = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    seeds = list(range(6000, 6005))
    tasks = [(eta, s) for eta in etas for s in seeds]
    print(f"guardrail: {len(tasks)} tasks...", flush=True)

    # frozen-SNN reference on the same base distribution
    from m6_energy import build
    env = Go2NavEnv(Go2NavConfig(shift_time_s=1e9, episode_len_s=60.0))
    frozen_succ = []
    ctrl = build("Frozen SNN", 7)
    for ep in range(24):
        obs, _ = env.reset(seed=4000 + ep)
        while True:
            a = ctrl.act(obs)
            nobs, r, term, trunc, info = env.step(a)
            ctrl.observe(r, nobs, term or trunc)
            obs = nobs
            if term or trunc:
                break
        frozen_succ.append(int(info["reached"]))
    env.close()
    frozen_rate = float(np.mean(frozen_succ))

    res = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_block, *t): t for t in tasks}
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            eta, seed, rate, finite = f.result()
            res.setdefault(eta, []).append((rate, finite))
            if (i + 1) % 6 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)

    lines = ["# e-prop guardrail: base-distribution stability across eta\n",
             f"Frozen SNN reference (no plasticity): {frozen_rate:.0%} over 24 eps.\n",
             "| eta | mean success (5 seeds) | all weights finite? | verdict |",
             "|---|---|---|---|"]
    safe_etas = []
    for eta in etas:
        rates = [r for r, _ in res[eta]]
        finite_all = all(fi for _, fi in res[eta])
        m = np.mean(rates)
        verdict = "OK" if (finite_all and m >= frozen_rate - 0.10) else ("DIVERGED" if not finite_all else "DEGRADED")
        if verdict == "OK":
            safe_etas.append(eta)
        lines.append(f"| {eta} | {m:.0%} | {finite_all} | {verdict} |")
    picked = max(safe_etas) if safe_etas else min(etas)
    lines.append(f"\n**Recommended eta for the rigor test: {picked}** "
                 f"(largest that stays within 10pts of frozen SNN, weights finite)")

    (OUT / "guardrail.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT/'guardrail.md'}")


if __name__ == "__main__":
    main()
