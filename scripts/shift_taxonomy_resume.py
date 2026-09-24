"""Resume shift_taxonomy.py --part all for the two shift types that never
finished when the host machine rebooted mid-run (sensor_bias completed
successfully -- see archive/shift_taxonomy/{severity,rigor}_sensor_bias.md --
but sensor_range died mid-severity-screen with no output written, and
goal_drift never started). Runs the same severity-then-rigor loop as the
original --part all path, just skipping sensor_bias.

Run:  conda run -n nmc python scripts/shift_taxonomy_resume.py
"""
from __future__ import annotations
import sys

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")

from shift_taxonomy import SEVERITY_GRID, run_severity, run_rigor, OUT

REMAINING = ["sensor_range", "goal_drift"]


def main():
    for shift in REMAINING:
        picked = run_severity(shift)
        run_rigor(shift, picked)

    summary = ["# Shift taxonomy: cross-shift summary\n",
               "| shift type | R-STDP beats frozen SNN? |", "|---|---|"]
    for shift in list(SEVERITY_GRID):
        f = OUT / f"rigor_{shift}.md"
        verdict = "?"
        if f.exists():
            txt = f.read_text(encoding="utf-8")
            verdict = "YES" if "beats frozen SNN on" in txt and "YES**" in txt else "NO"
        summary.append(f"| {shift} | {verdict} |")
    summary.append("\n(compare with: sensor dropout = NO (M5), terrain sand/ice = see M4c_rigorous)")
    (OUT / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
