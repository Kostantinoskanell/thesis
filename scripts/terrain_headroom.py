"""M4b terrain retuning: M4 tried floor-friction shifts (ice mu=0.08, sand mu=1.6)
and found ice too harsh (floors success) and sand too mild (no real drop). This
sweeps intermediate friction values to find a "terrain" shift with real headroom,
the same screening idea as shift_headroom.py applied to a friction sweep instead
of a fixed two-point choice.

The RL walker was trained with floor friction ~U(0.4, 1.0) (see go2_nav_env.py's
_inject_terrain docstring) -- anything outside that band is out-of-distribution
for the low-level locomotion policy, which is the whole point of this shift.

Run:  conda run -n nmc python scripts/terrain_headroom.py --episodes 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from nmc.controllers.snn import LIFNet, SNNNavController
from nmc.controllers.privileged_expert import PrivilegedExpert, PrivilegedConfig
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from eval_mlp_go2 import run_episode

SNN_CKPT = ROOT / "assets" / "snn_seeds" / "snn_seed0.pt"

# mu values to sweep: below the training band (slippery, "ice"-like) and above it
# (grippy/draggy, "sand"-like). 0.08/1.6 are M4's rejected endpoints, kept for
# reference; the rest probe the space between the training band and those extremes.
FRICTION_SWEEP = [0.08, 0.15, 0.20, 0.28, 1.2, 1.6, 2.2, 3.0]


def load_frozen_snn(seed):
    ckpt = torch.load(SNN_CKPT, weights_only=True)
    net = LIFNet(in_dim=ckpt["in_dim"], hidden=tuple(ckpt["hidden"]),
                 n_pops=ckpt["n_pops"], pop_size=ckpt["pop_size"],
                 neuron=ckpt.get("neuron", "lif"))
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    return SNNNavController(net, n_steps=ckpt["tsteps"], plasticity_enabled=False, seed=seed)


def eval_condition(mu, episodes, seed0, shifted: bool):
    shift_time = 0.5 if shifted else 1e9
    cfg = Go2NavConfig(shift_time_s=shift_time, episode_len_s=60.0,
                       shift_type="terrain", terrain_mode="custom",
                       terrain_friction={"custom": mu})
    env = Go2NavEnv(cfg)
    expert = PrivilegedExpert(PrivilegedConfig(
        inflate_m=0.9, fallback_inflations_m=(0.9, 0.7, 0.55)))
    ctrl = load_frozen_snn(seed=1234)
    succ = 0
    for ep in range(episodes):
        info, *_ = run_episode(env, ctrl, seed=seed0 + ep, expert=expert, record_frames=False)
        succ += int(info["reached"])
    env.close()
    return succ / episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=6000)
    args = ap.parse_args()

    base = eval_condition(1.0, args.episodes, args.seed0, shifted=False)  # unshifted ref
    print(f"unshifted baseline (default floor mu): {base:.0%}\n")

    header = f"{'floor mu':>10s} {'shifted%':>10s} {'drop (pt)':>11s}  verdict"
    print(header); print("-" * len(header))
    usable = []
    for mu in FRICTION_SWEEP:
        shifted = eval_condition(mu, args.episodes, args.seed0, shifted=True)
        drop = (base - shifted) * 100
        if shifted < 0.05:
            verdict = "too harsh (floored)"
        elif drop < 8:
            verdict = "too mild (no headroom)"
        else:
            verdict = "USABLE"
            usable.append((mu, base, shifted, drop))
        print(f"{mu:10.2f} {shifted:10.0%} {drop:11.1f}  {verdict}", flush=True)

    print("\nUsable friction values (real headroom for a recovery experiment):")
    for mu, b, s, d in usable:
        print(f"  - mu={mu}: {b:.0%} -> {s:.0%} ({d:.1f} pt drop)")


if __name__ == "__main__":
    main()
