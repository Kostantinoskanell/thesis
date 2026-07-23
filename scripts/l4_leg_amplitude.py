"""Per-leg joint amplitude/periodicity check -- is one leg dragging (not cycling) while
the others step normally? A dragging leg shows much smaller range-of-motion and/or no
periodic swing on its knee/calf joint vs the other three legs during steady walking."""
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "data/l4_dagger_upright_walktest.npz"
d = np.load(path, allow_pickle=True)
jp = d["joint_pos"]
names = [str(x) for x in d["joint_names"]]
n = jp.shape[0]
lo, hi = 200, min(900, n)  # steady-state window, skip startup transient + any tail fall

legs = ["FL", "FR", "RL", "RR"]
print(f"file: {path}  steady window [{lo}:{hi}] of {n} steps\n")
for leg in legs:
    idxs = [i for i, nm in enumerate(names) if nm.startswith(leg + "_")]
    print(f"-- {leg} --")
    for i in idxs:
        seg = jp[lo:hi, i]
        amp = seg.max() - seg.min()
        std = seg.std()
        vel = np.diff(seg)
        sign_changes = int((np.diff(np.sign(vel)) != 0).sum())  # rough swing-cycle count
        print(f"  {names[i]:14s} range [{seg.min():+.3f},{seg.max():+.3f}] amp {amp:.3f} "
              f"std {std:.3f}  sign_changes {sign_changes}")
