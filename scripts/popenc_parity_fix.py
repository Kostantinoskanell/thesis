"""Re-run of the D4 PopEnc-SNN baseline parity check ONLY, after fixing a bug
in popenc_verify_and_h4.py's _parity_task: it ignored its own `seed` argument
(hardcoded env.reset(seed=4000+ep)), so all 3 "seeds" ran the identical 20
episodes and trivially agreed -- a fake +/-0.0% SD across what was actually
one 20-episode run, not 60 independent ones. This reruns with seed correctly
fed into env.reset(), same scale (3 seeds x 20 eps) as the original.

Run:  conda run -n nmc python scripts/popenc_parity_fix.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np

from popenc_verify_and_h4 import _parity_task
from nmc.eval.metrics import mean_ci95

import concurrent.futures

OUT = Path("archive/D4_popenc_snn")
CKPT = "assets/snn_popenc_seeds/snn_popenc_seed0.pt"


def main():
    seeds = list(range(6000, 6003))
    tasks = [(CKPT, s, 20) for s in seeds]
    print(f"parity check (fixed seeding): {len(tasks)} tasks...", flush=True)
    rates, firing = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as ex:
        for f in concurrent.futures.as_completed({ex.submit(_parity_task, *t): t for t in tasks}):
            seed, rate, fr = f.result()
            rates.append(rate); firing.append(fr)
            print(f"  seed={seed}: success={rate:.1%}", flush=True)
    m, h = mean_ci95(rates)
    print(f"\nCorrected parity: {m:.1%} +/- {h:.1%} (per-seed: {[f'{r:.1%}' for r in rates]}), "
          f"mean firing rate {np.mean(firing):.1%}")
    print("Reference: original rate/TTFS SNN (M3) 41% [32,51] Wilson-95; MLP 37% [29,45].")
    parity_ok = m > 0.25
    print(f"Parity gate ({'PASS' if parity_ok else 'FAIL'})")


if __name__ == "__main__":
    main()
