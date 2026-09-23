"""e-prop vs R-STDP vs Frozen SNN on sensor dropout -- CORRECTED rigor pass.

Supersedes eprop_rigor.py, which had two real gaps: (1) it compared e-prop's
fresh per-seed data against OLD M5 aggregate (mean, SD) numbers via a normal
approximation, not a real paired test on matched raw data, and (2) it never
persisted its own raw per-seed array, so that approximation couldn't even be
fixed retroactively. Both are avoidable here: m5_full_comparison.py's
evaluate_seed() uses the IDENTICAL seed range (6000-6009) and IDENTICAL
env-reset scheme (seed*100+ep), so re-running Frozen SNN and R-STDP SNN fresh
in this same script gives genuinely matched per-seed data for a real Welch
t-test -- and this script writes the raw CSV so it never has to be redone.

Run:  conda run -n nmc python scripts/eprop_rigor2.py --eta 0.005
"""
from __future__ import annotations
import argparse
import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np
import scipy.stats
from statsmodels.stats.multitest import multipletests

from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from nmc.controllers.eprop_snn import EPropNavController
from m5_full_comparison import make_frozen_snn, make_rstdp

OUT = Path("archive/D1_eprop")
CKPT = "assets/snn_seeds/snn_seed0.pt"
SEEDS = list(range(6000, 6010))


def evaluate(name, seed, dropout, eta, n_eps):
    env = Go2NavEnv(Go2NavConfig(shift_type="sensor", shift_time_s=0.5,
                                 sensor_dropout_frac=dropout, sensor_dropout_start=8,
                                 episode_len_s=60.0))
    if name == "e-prop SNN":
        ctrl = EPropNavController(CKPT, seed=seed, eta=eta, anchor=0.005, plastic_layers=(0, -1))
    elif name == "Frozen SNN":
        ctrl = make_frozen_snn(seed)
    elif name == "R-STDP SNN":
        ctrl = make_rstdp(seed)
    else:
        raise ValueError(name)

    succ = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed * 100 + ep)
        while True:
            a = ctrl.act(obs)
            nobs, r, term, trunc, info = env.step(a)
            ctrl.observe(r, nobs, term or trunc)
            obs = nobs
            if term or trunc:
                break
        succ.append(int(info["reached"]))
    env.close()
    return name, seed, dropout, float(np.mean(succ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, default=0.005,
                    help="e-prop rigor guardrail recommendation")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    controllers = ["Frozen SNN", "R-STDP SNN", "e-prop SNN"]
    dropouts = (0.30, 0.20)
    n_eps = 30

    tasks = [(name, s, d, args.eta, n_eps) for d in dropouts for name in controllers for s in SEEDS]
    print(f"{len(tasks)} tasks total ({len(controllers)} controllers x {len(dropouts)} dropouts x "
          f"{len(SEEDS)} seeds), {n_eps} eps each...", flush=True)

    raw = {(name, d): {} for d in dropouts for name in controllers}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(evaluate, *t): t for t in tasks}
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            name, seed, dropout, rate = f.result()
            raw[(name, dropout)][seed] = rate
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)

    # persist raw per-seed data so this never has to be re-derived from a
    # summary approximation again
    csv_lines = ["controller,dropout,seed,success_rate"]
    for (name, d), by_seed in raw.items():
        for s in SEEDS:
            csv_lines.append(f"{name},{d},{s},{by_seed[s]:.4f}")
    (OUT / "raw_success_rates.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    arrs = {(name, d): np.array([raw[(name, d)][s] for s in SEEDS]) for name in controllers for d in dropouts}

    lines = ["# e-prop rigor test (CORRECTED): does e-prop recover where R-STDP did not?\n",
             f"eta={args.eta}, anchor=0.005, plastic_layers=(0,-1) -- identical protocol and seeds "
             f"(6000-6009, seed*100+ep resets) for all three controllers, run together in this pass "
             f"so the comparison is a real paired Welch t-test on matched raw data, not an "
             f"approximation against old summary statistics. Holm-Bonferroni corrected across all "
             f"4 comparisons below.\n"]

    pvals = []
    comparisons = []  # (dropout, other_name, delta)
    for d in dropouts:
        lines.append(f"\n## dropout = {d}\n")
        lines.append("| controller | success (mean+/-SD) |")
        lines.append("|---|---|")
        for name in controllers:
            a = arrs[(name, d)]
            lines.append(f"| {'**' + name + '**' if name == 'e-prop SNN' else name} | {a.mean():.1%}+/-{a.std(ddof=1):.1%} |")
        for other in ("Frozen SNN", "R-STDP SNN"):
            a_e = arrs[("e-prop SNN", d)]
            a_o = arrs[(other, d)]
            t, p = scipy.stats.ttest_ind(a_e, a_o, equal_var=False)
            pvals.append(p)
            comparisons.append((d, other, a_e.mean() - a_o.mean()))

    reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    lines.append("\n## Holm-Bonferroni corrected comparisons (e-prop vs baseline)\n")
    lines.append("| dropout | vs | delta | raw p | Holm p | significant? |")
    lines.append("|---|---|---|---|---|---|")
    for (d, other, delta), p_raw, p_h, sig in zip(comparisons, pvals, p_adj, reject):
        lines.append(f"| {d} | {other} | {delta:+.1%} | {p_raw:.3f} | {p_h:.3f} | {'YES' if sig else 'no'} |")

    any_sig_frozen = any(sig and other == "Frozen SNN" for (_, other, _), sig in zip(comparisons, reject))
    verdict = ("e-prop shows a statistically significant (Holm-corrected) advantage over frozen SNN"
               if any_sig_frozen else
               "e-prop does NOT show a Holm-corrected significant advantage over frozen SNN -- "
               "the null result generalizes beyond R-STDP specifically")
    lines.append(f"\n\n**Verdict: {verdict}**")

    (OUT / "rigor2.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT/'rigor2.md'} and {OUT/'raw_success_rates.csv'}")


if __name__ == "__main__":
    main()
