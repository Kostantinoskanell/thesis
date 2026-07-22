# L5 — energy of the spiking *locomotion* controller (H2)

_2026-07-22 · does the spiking Go2 policy actually cost less energy than the MLP?_

## Question (H2)
The whole point of a spiking controller is energy: it does sparse **accumulate** (AC)
ops only when neurons spike, vs the MLP's dense **multiply-accumulate** (MAC) every
forward pass. The nav layer's ALIF-SNN fired at ~1.4% and was dramatically cheaper. Does
the *locomotion* policy (the distilled PopSAN walker from L4) inherit that advantage?

## Method (Zhao et al. 2025 / Horowitz 2014, 45 nm)
`scripts/l5_energy_analysis.py`. Measure per-layer firing rates on 4096 **real walking
observations**, then:
- **spiking SynOps** = Σ_layer (in × out × pre-firing-rate × T) accumulate ops, +
  encoder Gaussian currents (48×in_pop MULs) + decoder (act×out_pop MACs).
- **MLP** = Σ_layer (in × out) = 40,448 MACs, once per decision.
- energy = SynOps·0.9 pJ (AC) + MUL·3.7 pJ + MAC·4.6 pJ.

Walking is re-verified for every policy by trajectory (`--force-command 0.5,0,0`,
`l4_traj_metrics.py`: sustained base height ≈0.19 m + body vx tracks 0.5 + multi-metre
**path length**) — never by reward. (Note: net displacement `bp[-1]-bp[0]` is a red
herring — the Play env wraps position on reset, so it reads ~0.49 m for *every* run,
walker or belly-flop; path length is the real travel metric.)

## Finding: the naive distilled walker is NOT efficient — it fires densely
| policy | mean hidden firing | T | val-MSE | SynOps | energy | vs MLP | walks? |
|---|---|---|---|---|---|---|---|
| **dense** (L4 DAgger walker) | 62.7% | 8 | 0.025 | 362,830 | 328.9 nJ | **0.57× (costlier)** | ✅ path 19.1 m |
| **+ firing penalty** (λ=0.05) | 44.2% | 8 | 0.029 | 294,011 | 266.9 nJ | **0.70× (costlier)** | ✅ path 19.0 m |
| **+ penalty, T=5** (λ=0.03) | 47.7% | 5 | 0.038 | 195,335 | 178.1 nJ | **1.04× cheaper** | ✅ path 18.3 m |
| **+ penalty, T=4** (λ=0.05) | 44.7% | 4 | 0.049 | 149,382 | 136.8 nJ | 1.36× cheaper | ❌ path 2.6 m (belly-flop) |
| MLP baseline | — | — | — | 40,448 MAC | 186.1 nJ | 1.0× | ✅ |

Two structural reasons the spiking policy starts out *more expensive*:
1. **Dense firing.** Behavior cloning only matches actions — nothing pressures the net to
   spike sparsely, so it fires at 62.7% (vs the nav ALIF-SNN's 1.4%). At that rate the
   AC-vs-MAC advantage evaporates.
2. **Population-coding + T overhead.** The encoder inflates obs 48→480 (in_pop=10), so
   `fc[0]` alone is 480×128, and every SynOp is paid **T times** per decision. At T=8 this
   dominates.

## What recovers H2 — and the energy↔fidelity tradeoff it exposes
- **Firing-rate penalty** (a differentiable mean-spike-activity term added to the
  distillation loss, `PopSpikingActorNet.forward_with_activity` + `--firing-penalty`)
  halves firing (62.7%→44%) at negligible gait cost (val-MSE 0.025→0.029, still walks). But
  that alone only reaches 0.70× — still costlier — because the T=8 population overhead
  dominates.
- **Reducing T is the dominant energy lever, but it trades against gait fidelity.** T=4
  makes the SNN 1.36× cheaper but **breaks walking** — the coarser rate code (val-MSE
  jumps to 0.049) sends it back into the belly-flop (base 0.167 m, vx 0.007, path 2.6 m).
- **T=5 is the minimum T that preserves the gait** → **1.04× cheaper than the MLP while
  still walking** (base 0.190 m, vx 0.476, path 18.3 m; `sparse_t5_walk.gif`). This is the
  sweet spot: **the first walking spiking Go2 locomotion policy that is also (marginally)
  more energy-efficient than its dense-MLP equivalent at 45 nm digital.**

## Honest framing
- The 1.04× margin is **thin** at 45 nm digital — locomotion does not get the nav layer's
  order-of-magnitude win. Continuous, precise motor control resists the sparse-firing
  regime (it needs denser firing and several timesteps for action fidelity), whereas coarse
  discrete navigation is naturally sparse. **This echoes the L4 finding** (R-STDP recovers
  navigation but destabilizes locomotion): the locomotion layer is a harder, more delicate
  substrate for neuromorphic methods than the navigation layer. Both milestones tell the
  same story — nav is the natural fit; loco is achievable but marginal.
- **45 nm digital SynOps is a conservative proxy.** On neuromorphic hardware (Loihi) the
  advantage is far larger — event-driven execution, much cheaper AC, and no dense
  weight-memory movement (PopSAN reports ~140× energy vs GPU on Loihi). So 1.04× is a
  digital-ASIC floor, not the neuromorphic ceiling.

## Reproduce
```
# firing-rate-regularized distillation at chosen T (nmc env, CPU):
python scripts/l4_distill_spiking.py --data data/l4_distill_data.npz,data/l4_dagger_data.npz \
   --init-weights data/l4_dagger_spiking.pt --out data/l4_sparse_t5_spiking.pt \
   --T 5 --firing-penalty 0.03 --epochs 40 --lr 5e-4
# energy:
python scripts/l5_energy_analysis.py --student data/l4_sparse_t5_spiking.pt --T 5
# walk-verify (WSL isaac env) + metrics:
wsl bash scripts/wsl_isaac_l4.sh --mode baseline --T 5 \
   --load-weights <abs>/data/l4_sparse_t5_spiking.pt --force-command 0.5,0,0 \
   --episodes 3 --dump-traj <abs>/data/l4_sparse_t5_walktest.npz
python scripts/l4_traj_metrics.py data/l4_sparse_t5_walktest.npz
python scripts/l4_render_traj_mujoco.py data/l4_sparse_t5_walktest.npz archive/L5_energy/sparse_t5_walk.gif
```
