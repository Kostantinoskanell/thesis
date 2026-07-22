"""Diagnose whether the spiking Go2 policy actually WALKS, from a dumped trajectory."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "data/l4_walk_traj.npz"
out = sys.argv[2] if len(sys.argv) > 2 else "archive/L4_gait_check/fig_gait_diagnostic.png"
import os; os.makedirs(os.path.dirname(out), exist_ok=True)

d = np.load(path, allow_pickle=True)
bp, bv, jp, cmd = d["base_pos"], d["base_lin_vel"], d["joint_pos"], d["command"]
T = bp.shape[0]; t = np.arange(T) * 0.02  # 50 Hz control

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0,0].plot(t, bp[:,2], color="#c0392b"); ax[0,0].axhline(0.30, ls="--", color="green", label="normal standing ~0.30 m")
ax[0,0].set_title("base height over time"); ax[0,0].set_xlabel("s"); ax[0,0].set_ylabel("height (m)")
ax[0,0].legend(frameon=False); ax[0,0].set_ylim(0, 0.45)

ax[0,1].plot(bp[:,0]-bp[0,0], bp[:,1]-bp[0,1], color="#1950a0")
ax[0,1].scatter([0],[0], c="green", s=40, label="start"); ax[0,1].set_title(f"base XY path (traveled {np.linalg.norm(bp[-1,:2]-bp[0,:2]):.2f} m)")
ax[0,1].set_xlabel("x (m)"); ax[0,1].set_ylabel("y (m)"); ax[0,1].legend(frameon=False); ax[0,1].axis("equal")

ax[1,0].plot(t, bv[:,0], color="#c0392b", label="actual vx")
ax[1,0].plot(t, cmd[:,0], ls="--", color="black", label="commanded vx")
ax[1,0].set_title("forward velocity tracking"); ax[1,0].set_xlabel("s"); ax[1,0].set_ylabel("m/s"); ax[1,0].legend(frameon=False)

for i,c in zip(range(4,8), ["#e74c3c","#e67e22","#f1c40f","#2ecc71"]):
    ax[1,1].plot(t[:200], jp[:200,i], color=c, lw=0.8)
ax[1,1].set_title("thigh-joint angles (first 4 s) -- periodic = real gait")
ax[1,1].set_xlabel("s"); ax[1,1].set_ylabel("rad")

fig.suptitle(f"L4 gait check: mean height {bp[:,2].mean():.2f} m, "
             f"mean |vx| {abs(bv[:,0]).mean():.3f} m/s, traveled {np.linalg.norm(bp[-1,:2]-bp[0,:2]):.2f} m",
             fontsize=13)
fig.tight_layout()
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
