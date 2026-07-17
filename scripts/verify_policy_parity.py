"""Train<->deploy parity gate: NumPy policy (Windows) vs brax policy (WSL/JAX).

Loads the parity vectors exported by scripts/rl/export_go2_policy.py (64 obs +
the actions brax's own deterministic inference fn produced for them in WSL) and
checks our pure-NumPy reimplementation reproduces the same actions. This guards
against silent obs-normalization / weight-layout / activation mismatches that
would otherwise poison every experiment built on the Windows walking loop.

Run on Windows (conda env nmc):  python scripts/verify_policy_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from nmc.rl.numpy_policy import NumpyGo2Policy


def main():
    export = ROOT / "assets" / "go2_policy_export.npz"
    vectors = ROOT / "assets" / "go2_parity_vectors.npz"
    policy = NumpyGo2Policy(export)
    z = np.load(vectors)
    obs, ref = z["obs"], z["actions"]

    ours = np.stack([policy(o) for o in obs])
    err = np.abs(ours - ref)
    print(f"vectors: {len(obs)}   max|err| = {err.max():.3e}   mean|err| = {err.mean():.3e}")
    # float32 net evaluated in float64 vs jax float32: tolerance a few ulp above 1e-6
    ok = err.max() < 1e-4
    print("PARITY", "OK" if ok else "FAILED")
    if not ok:
        worst = np.unravel_index(err.argmax(), err.shape)
        print(f"worst: vector {worst[0]} dim {worst[1]}: ours {ours[worst]:.6f} ref {ref[worst]:.6f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
