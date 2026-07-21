"""M4b shift screening: which candidate mid-episode distribution shifts give the
FROZEN pretrained ALIF-SNN real headroom to recover (a meaningful base->shifted
success drop, not floor/ceiling)? This is the permanent version of the scratch
check that selected "sensor dropout" over "obstacles" for M4 (see
archive/M4_pilot_go2/README.md's journey step 1) -- applied here to the new
M4b candidates (sensor_bias, sensor_range, goal_drift) plus the existing three
for a full side-by-side.

A shift with near-zero drop is too weak (no re-learning required, like
"obstacles"). A shift that floors success near 0% is too harsh (nothing to
measure recovery against, like "terrain-ice"). The useful zone is the same
~15-30 point drop with real floor left that "sensor" showed at M4.

Run:  conda run -n nmc python scripts/shift_headroom.py --episodes 20
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
from eval_mlp_go2 import run_episode   # shared rollout helper

SNN_CKPT = ROOT / "assets" / "snn_seeds" / "snn_seed0.pt"

# Same shift_time_s=0.5 convention as pilot_m4.py's --mode compare: the shift is
# active for essentially the whole episode, so success reflects post-shift
# competence directly (no per-episode continual adaptation here -- frozen model).
CANDIDATES = {
    "sensor (M4 adopted)":     dict(shift_type="sensor", sensor_dropout_frac=0.30),
    "obstacles (M4 rejected)": dict(shift_type="obstacles"),
    "terrain-ice (M4 harsh)":  dict(shift_type="terrain", terrain_mode="ice"),
    "terrain-sand (M4 mild)":  dict(shift_type="terrain", terrain_mode="sand"),
    "sensor_bias":             dict(shift_type="sensor_bias"),
    "sensor_range":            dict(shift_type="sensor_range"),
    "goal_drift":              dict(shift_type="goal_drift"),
}


def load_frozen_snn(seed):
    ckpt = torch.load(SNN_CKPT, weights_only=True)
    net = LIFNet(in_dim=ckpt["in_dim"], hidden=tuple(ckpt["hidden"]),
                 n_pops=ckpt["n_pops"], pop_size=ckpt["pop_size"],
                 neuron=ckpt.get("neuron", "lif"))
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    return SNNNavController(net, n_steps=ckpt["tsteps"], plasticity_enabled=False, seed=seed)


def eval_condition(shift_kwargs, episodes, seed0, shifted: bool):
    shift_time = 0.5 if shifted else 1e9
    cfg = Go2NavConfig(shift_time_s=shift_time, episode_len_s=60.0, **shift_kwargs)
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
    ap.add_argument("--seed0", type=int, default=5000)
    args = ap.parse_args()

    print(f"frozen ALIF-SNN, {args.episodes} episodes per condition\n")
    header = f"{'shift':26s} {'base':>7s} {'shifted':>9s} {'drop (pt)':>11s}  verdict"
    print(header); print("-" * len(header))
    results = []
    for name, kwargs in CANDIDATES.items():
        base = eval_condition(kwargs, args.episodes, args.seed0, shifted=False)
        shifted = eval_condition(kwargs, args.episodes, args.seed0, shifted=True)
        drop = (base - shifted) * 100
        if shifted < 0.05:
            verdict = "too harsh (floored)"
        elif drop < 8:
            verdict = "too weak (no headroom)"
        else:
            verdict = "USABLE"
        results.append((name, base, shifted, drop, verdict))
        print(f"{name:26s} {base:7.0%} {shifted:9.0%} {drop:11.1f}  {verdict}", flush=True)

    print("\nUsable candidates (real headroom for a recovery experiment):")
    for name, base, shifted, drop, verdict in results:
        if verdict == "USABLE":
            print(f"  - {name}: {base:.0%} -> {shifted:.0%} ({drop:.1f} pt drop)")


if __name__ == "__main__":
    main()
