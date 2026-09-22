"""Stitch the two separate H4 noise-sweep runs (noise_sweep.md: sigma<=0.2,
noise_sweep_high.md: sigma>=0.3) into ONE figure and ONE summary table spanning
the full sigma in [0, 0.8] range.

Found during a visual audit: the site was showing fig_h4_noise_high.png (only
sigma=0.3-0.8) next to a text table covering the FULL 0-0.8 range, so the
figure's x-axis started at 0.3 with no visual context for the reader. No new
simulation needed -- both underlying datasets already exist; this just
re-plots them together and recomputes slope/N50 over the full stitched curve
rather than two disjoint half-range values.

Run:  conda run -n nmc python scripts/m6_noise_stitch.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from m6_noise_sweep import n50

OUT = Path("archive/M6_energy")

SIGMAS = [0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]

# (mean, half-width) per controller per sigma, read off noise_sweep.md + noise_sweep_high.md
DATA = {
    "Frozen MLP":     [(42.0,10.4),(42.0,10.4),(50.0,19.6),(52.0,10.4),(34.0,14.2),(42.0,16.2),(34.0,18.8),(38.0,23.9)],
    "Online MLP":     [(48.0,16.2),(42.0,10.4),(48.0, 5.6),(42.0,10.4),(36.0,28.6),(40.0,24.8),(36.0,38.9),(10.0,15.2)],
    "Frozen SNN":     [(40.0,23.2),(52.0,25.4),(42.0,26.9),(50.0,29.1),(40.0,12.4),(28.0,16.2),(18.0,18.4),(18.0,18.4)],
    "Pure-STDP SNN":  [(14.0, 6.8),(16.0,11.1),(16.0,22.6),( 8.0,10.4),( 4.0,11.1),(16.0,18.8),( 8.0,10.4),( 6.0, 6.8)],
    "R-STDP SNN":     [(42.0,10.4),(44.0,20.8),(54.0,14.2),(48.0,10.4),(30.0, 8.8),(28.0,13.6),(36.0,22.6),(34.0,14.2)],
    "TM-NORM SNN":    [( 2.0, 5.6),(16.0,14.2),(20.0,12.4),(20.0,17.6),(14.0,18.8),(14.0,11.1),( 4.0, 6.8),( 0.0, 0.0)],
}

colours = {"Frozen MLP": "#b0b0b0", "Online MLP": "#1950a0", "Frozen SNN": "#7f8c8d",
           "Pure-STDP SNN": "#e67e22", "R-STDP SNN": "#c0392b", "TM-NORM SNN": "#8e44ad"}

fig, ax = plt.subplots(figsize=(8, 5.2))
lines = ["# M6 / H4 -- graceful degradation under sensor noise (full range, stitched)\n",
         "Base distribution (no dropout shift), additive Gaussian LiDAR noise, "
         "5 seeds x 10 eps per point. CI = 95% (t, across seeds). Stitched from "
         "`noise_sweep.md` (sigma<=0.2) and `noise_sweep_high.md` (sigma>=0.3); "
         "slope/N50 below are recomputed over the FULL 0-0.8 curve.\n",
         "| controller | " + " | ".join(f"s={s}" for s in SIGMAS) + " | slope (pts/0.1s) | N50 |",
         "|---" * (len(SIGMAS) + 2) + "|"]

for name, vals in DATA.items():
    means = np.array([m for m, _ in vals])
    halfs = np.array([h for _, h in vals])
    ax.errorbar(SIGMAS, means / 100, yerr=halfs / 100, marker="o", capsize=3, lw=1.8,
                color=colours[name], label=name)
    slope = np.polyfit(SIGMAS, means, 1)[0] * 0.1
    n50v = n50(SIGMAS, (means / 100).tolist())
    n50s = "never" if np.isinf(n50v) else ("--" if np.isnan(n50v) else f"{n50v:.3f}")
    cells = " | ".join(f"{m:.1f}%+/-{h:.1f}%" for m, h in vals)
    lines.append(f"| {name} | {cells} | {slope:+.1f} | {n50s} |")

ax.set_xlabel("LiDAR noise sigma"); ax.set_ylabel("success rate")
ax.set_ylim(0, None); ax.legend(frameon=False, fontsize=8)
ax.set_title("H4: graceful degradation under sensor noise, full range (95% CI over seeds)")
fig.tight_layout()
fig.savefig(OUT / "fig_h4_noise_full.png", bbox_inches="tight", dpi=150)

(OUT / "noise_sweep_full.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\nwrote {OUT/'fig_h4_noise_full.png'} and {OUT/'noise_sweep_full.md'}")
