"""Quick walking metrics from a dumped L4 trajectory npz (base_pos, base_lin_vel,
command). WALKING = base height sustained ~0.18-0.30 m + actual vx tracks the
commanded vx + travels many meters, with no collapse. Prints one summary line set."""
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "data/l4_sparse_t4_walktest.npz"
d = np.load(path, allow_pickle=True)
bp, bv, cmd = d["base_pos"], d["base_lin_vel"], d["command"]
n = bp.shape[0]
print(f"file: {path}   steps: {n}")
print("base height: mean %.3f  min %.3f  max %.3f" % (bp[:, 2].mean(), bp[:, 2].min(), bp[:, 2].max()))
print("commanded vx %.2f  actual vx: mean %.3f  steady(200:) %.3f" % (cmd[0, 0], bv[:, 0].mean(), bv[200:, 0].mean()))
print("vx err (steady 200:) %.3f" % abs(bv[200:, 0] - cmd[200:, 0]).mean())
# path length (sum of per-step displacements) is the true "did it travel" metric;
# net bp[-1]-bp[0] is misleading (the Play env wraps position on episode reset).
path_len = np.linalg.norm(np.diff(bp[:, :2], axis=0), axis=1).sum()
print("path length %.2f m  (net %.2f m)" % (path_len, np.linalg.norm(bp[-1, :2] - bp[0, :2])))
