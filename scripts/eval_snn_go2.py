"""M3 evaluation: frozen pretrained LIF-SNN closed-loop, vs the M2 MLP.

Same harness/metrics as eval_mlp_go2 (success + Wilson CI, SPL, collision rate),
so the SNN vs MLP comparison is apples-to-apples. Exit criterion: SNN pre-shift
success within a few % of the MLP's 37%. This frozen controller is also the
frozen-SNN ablation (proposal controller 3) reused in M4.

Run:  conda run -n nmc python scripts/eval_snn_go2.py --episodes 30
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

import numpy as np
import torch

from nmc.controllers.snn import LIFNet, SNNNavController
from nmc.controllers.privileged_expert import PrivilegedExpert, PrivilegedConfig
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from nmc.eval.metrics import spl, collision_rate, mean_ci95, wilson_ci95
from eval_mlp_go2 import shortest_path_len, run_episode   # shared helpers

OUT = ROOT / "archive" / "M3_snn_pretrain_go2"


def load_snn(path, seed):
    ckpt = torch.load(path, weights_only=True)
    net = LIFNet(in_dim=ckpt["in_dim"], hidden=tuple(ckpt["hidden"]),
                 n_pops=ckpt["n_pops"], pop_size=ckpt["pop_size"],
                 neuron=ckpt.get("neuron", "lif"))
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    return SNNNavController(net, n_steps=ckpt["tsteps"], plasticity_enabled=False, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=2000)
    ap.add_argument("--seeds-dir", default="assets/snn_seeds")
    ap.add_argument("--mlp-success", type=float, default=0.37, help="M2 MLP bar for the plot")
    ap.add_argument("--outdir", default=None, help="override archive dir (e.g. LIF reference)")
    args = ap.parse_args()
    global OUT
    if args.outdir:
        OUT = ROOT / args.outdir
    OUT.mkdir(parents=True, exist_ok=True)

    model_paths = sorted((ROOT / args.seeds_dir).glob("snn_seed*.pt"))
    assert model_paths, f"no SNN seed models in {args.seeds_dir}; run train_snn_go2.py"
    print(f"evaluating {len(model_paths)} SNN seed-models x {args.episodes} episodes")

    env = Go2NavEnv(Go2NavConfig(shift_time_s=1e9, episode_len_s=60.0))
    expert = PrivilegedExpert(PrivilegedConfig(
        inflate_m=0.9, fallback_inflations_m=(0.9, 0.7, 0.55)))

    per_seed, S, coll, pl, sp = [], [], [], [], []
    spikes_pd, fire_rate = [], []
    best_gif = None
    for si, mp in enumerate(model_paths):
        ctrl = load_snn(mp, seed=1234 + si)
        succ = 0
        for ep in range(args.episodes):
            want = best_gif is None
            info, steps, frames, apl, splen = run_episode(
                env, ctrl, seed=args.seed0 + ep, expert=expert, record_frames=want)
            succ += int(info["reached"])
            S.append(int(info["reached"])); coll.append(int(info["collision"]))
            pl.append(apl); sp.append(splen)
            if info["reached"] and want and frames:
                best_gif = frames
        st = ctrl.spike_stats()
        spikes_pd.append(st["spikes_per_decision"]); fire_rate.append(st["firing_rate"])
        per_seed.append(succ / args.episodes)
        print(f"  SNN seed-model {si}: success {succ}/{args.episodes} ({succ/args.episodes:.0%})"
              f"  | {st['spikes_per_decision']:.0f} spikes/decision, firing rate {st['firing_rate']:.1%}",
              flush=True)
    env.close()
    mean_spd = float(np.mean(spikes_pd)); mean_fr = float(np.mean(fire_rate))

    n = len(S); k = int(np.sum(S))
    rate, wl, wh = wilson_ci95(k, n)
    seed_mean, seed_hw = mean_ci95(per_seed)
    spl_val, coll_val = spl(S, pl, sp), collision_rate(coll)
    print(f"\n=== frozen LIF-SNN (pre-shift, {len(model_paths)} seeds x {args.episodes} eps) ===")
    print(f"success (pooled): {rate:.0%}  Wilson95 [{wl:.0%}, {wh:.0%}]  (n={n})")
    print(f"success (across-seed): {seed_mean:.0%} +/- {seed_hw:.0%}")
    print(f"SPL: {spl_val:.3f}   collision rate: {coll_val:.0%}   MLP ref: {args.mlp_success:.0%}")
    print(f"spikes/decision: {mean_spd:.0f}   firing rate: {mean_fr:.1%}  (H2 energy preview)")

    if best_gif:
        gif = OUT / "snn_episode.gif"
        best_gif[0].save(gif, save_all=True, append_images=best_gif[1:], duration=100, loop=0)
        print(f"wrote {gif}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.bar(["SNN success", "SNN SPL"], [rate, spl_val], color=["#8e44ad", "#1950a0"], width=0.55)
    a1.errorbar([0], [rate], yerr=[[rate - wl], [wh - rate]], fmt="none", ecolor="k", capsize=5)
    a1.axhline(args.mlp_success, color="#1f9d3a", ls="--", lw=1.3, label=f"MLP success ({args.mlp_success:.0%})")
    a1.set_ylim(0, 1); a1.set_ylabel("value")
    a1.set_title(f"frozen LIF-SNN pre-shift (Wilson95)\ncollision rate {coll_val:.0%}")
    a1.legend(frameon=False, fontsize=9)
    a2.bar(range(len(per_seed)), per_seed, color="#8e44ad", width=0.6)
    a2.axhline(seed_mean, color="k", lw=1.2, label=f"mean {seed_mean:.0%}")
    a2.axhspan(seed_mean - seed_hw, seed_mean + seed_hw, color="k", alpha=0.12,
               label=f"95% CI +/-{seed_hw:.0%}")
    a2.axhline(args.mlp_success, color="#1f9d3a", ls="--", lw=1.3, label="MLP")
    a2.set_xlabel("SNN seed-model"); a2.set_ylabel("success rate"); a2.set_ylim(0, 1)
    a2.set_title("per-seed success"); a2.legend(frameon=False, fontsize=8)
    fig.suptitle(f"M3 frozen LIF-SNN vs M2 MLP — {len(model_paths)} seeds x {args.episodes} eps")
    fig.tight_layout()
    fig.savefig(OUT / "fig_snn_eval.png", bbox_inches="tight", dpi=150)
    print(f"wrote {OUT / 'fig_snn_eval.png'}")

    # exit criterion: within 8 pts of the MLP (a "few %"), CI-aware
    ok = rate >= args.mlp_success - 0.08
    print("\nM3 SNN pre-shift parity:", "PASS" if ok else "BELOW MLP (needs work)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
