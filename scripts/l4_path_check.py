import sys
import numpy as np

d = np.load(sys.argv[1], allow_pickle=True)
bp = d["base_pos"]
bv = d["base_lin_vel"]
steps = np.diff(bp[:, :2], axis=0)
path_len = np.linalg.norm(steps, axis=1).sum()
net = np.linalg.norm(bp[-1, :2] - bp[0, :2])
print("x range %.2f..%.2f   y range %.2f..%.2f" % (bp[:, 0].min(), bp[:, 0].max(), bp[:, 1].min(), bp[:, 1].max()))
print("path length %.2f m   net displacement %.2f m" % (path_len, net))
print("mean body vx %.3f  vy %.3f" % (bv[:, 0].mean(), bv[:, 1].mean()))
