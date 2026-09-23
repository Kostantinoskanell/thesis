"""Correction: m4c_rigorous.py tested sand at mu=1.6 (render_terrain_videos.py's
bare CLI default), but M4c's actual headline claim ("sand: R-STDP fully closes
the gap to the MLP, 27%->47%, matches MLP's 47%") was measured at the RETUNED
mu=1.20 (see ROADMAP.md's M4c row) -- mu=1.6 was M4c's own FIRST, REJECTED
screening value ("sand *helps* it to 55%... unusable"). Re-runs the sand
condition only, at the correct mu=1.20, same 10-seed/30-eps/Holm-corrected
protocol as m4c_rigorous.py, to actually stress-test the claim that was made.
"""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from pathlib import Path
import numpy as np
import scipy.stats
from statsmodels.stats.multitest import multipletests

from m4c_rigorous import evaluate, CONTROLLERS

import concurrent.futures

OUT = Path("archive/M4c_rigorous")


def main():
    seeds = list(range(6000, 6010))
    mu = 1.20  # now a permanent entry in m4c_rigorous.CONDITIONS["sand_mu120"]
    tasks = [(n, "sand_mu120", s, 30) for n in CONTROLLERS for s in seeds]

    print(f"sand @ mu={mu} (the ACTUAL M4c-claimed severity): {len(tasks)} tasks...", flush=True)
    res = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(evaluate, *t): t for t in tasks}
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            name, terrain, seed, rate = f.result()
            res.setdefault(name, []).append(rate)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)

    arrs = {n: np.array(res[n]) for n in CONTROLLERS}
    rs = arrs["R-STDP SNN"]
    lines = [f"# M4c sand correction: re-run at the ACTUAL claimed mu=1.20\n",
             "The first m4c_rigorous.py pass used mu=1.6 (render_terrain_videos.py's bare "
             "CLI default) -- but that is the value M4c's OWN screening rejected as "
             "\"unusable\" (sand *helps* success to 55%). The real headline claim "
             "('closes the gap to the MLP') was measured at the RETUNED mu=1.20. This "
             "corrects that.\n",
             "| controller | success (mean+/-SD) | M4c's original claim (n=15, mu=1.20) |",
             "|---|---|---|"]
    m4c_orig = {"Frozen MLP": "47%", "Frozen SNN": "27%", "R-STDP SNN": "47%"}
    for n in CONTROLLERS:
        a = arrs[n]
        lines.append(f"| {n} | {a.mean():.1%}+/-{a.std():.1%} | {m4c_orig[n]} |")

    pvals, labels = [], []
    for other in ["Frozen SNN", "Frozen MLP"]:
        _, p = scipy.stats.ttest_ind(rs, arrs[other], equal_var=False)
        pvals.append(p); labels.append(other)
    _, p_corr, _, _ = multipletests(pvals, method="holm")
    lines.append("\n**R-STDP vs the others (Holm-corrected):**\n")
    lines.append("| comparison | delta | p (Holm) | significant? |")
    lines.append("|---|---|---|---|")
    for lab, p in zip(labels, p_corr):
        d = (rs.mean() - arrs[lab].mean()) * 100
        lines.append(f"| R-STDP vs {lab} | {d:+.1f} pts | {p:.3f} | {'YES' if p < 0.05 else 'no'} |")

    (OUT / "sand_mu120_correction.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT/'sand_mu120_correction.md'}")


if __name__ == "__main__":
    main()
