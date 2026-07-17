# M2 — MLP baselines on the dynamic Go2 env (SOTA-upgraded: SPL, CIs, eligibility traces)

_Archived 2026-07-17 · controllers 1–2 of five, strong-and-fair (see
[sota_decisions.md D7](../../docs/references/sota_decisions.md))._

## Frozen MLP (controller 1)
Behavior cloning on D3's A* demos + **DAgger** to fix covariate shift (war story:
[bc-covariate-shift-dagger](../../docs/debug-log/2026-07-17_bc-covariate-shift-dagger.md)).
Architecture upgraded to **512×512 + LayerNorm** (capacity parity with the LIF-SNN),
dropout 0.1 during BC.

**Evaluation is now multi-seed with confidence intervals** (5 independently-trained
seed-models × 30 held-out episodes = 150 episodes; held-out seeds 2000+ disjoint from
collection 1000+ and DAgger 3000+; shift disabled = pre-shift competence bar):

| metric | value |
|---|---|
| success (pooled, n=150) | **37%**, Wilson-95% **[29%, 45%]** |
| success (across-seed mean) | **37% ± 8%** (t-95%; per-seed 30/33/33/40/47%) |
| **SPL** (Anderson 2018) | **0.346** |
| collision rate | 55% |
| falls | 0 |

Passes the 30% bar. Reference: A* teacher *with privileged full-map info* = 62%. The gap
is the expected price of a **reactive, lidar-only** student vs a global planner — SPL 0.35
says successful runs are also reasonably direct, not lucky detours. The 512×512 arch beat
the old 256×256 (single-seed 33% → DAgger 40%, pooled 37%).

Figures: `fig_mlp_training.png` (BC curve + per-class acc), `fig_mlp_eval.png` (success
with Wilson CI + SPL; per-seed spread with t-CI band), `mlp_episode.gif` (successful run).

## Online-TD(λ) MLP (controller 2) — the fair R-STDP competitor
Upgraded from one-step TD to an **actor-critic with eligibility traces** (TD(λ),
λ=0.9). This is the deliberate structural parallel to R-STDP: both accumulate a
per-parameter/synapse eligibility trace and consolidate it under an environment-reward
signal — differing only in substrate (SGD vs local plasticity). λ=0 exactly recovers the
proposal's original one-step baseline. Verified: one act→observe changes weights
(max|ΔW| = 1.5e-3); provably frozen when `frozen=True`. Closed-loop adaptation is measured
in M4/M5 (this milestone is the pre-shift bar).

## Variance / rigor notes
- CIs capture **init + SGD-order** variance (5 seed-models on the fixed DAgger dataset).
  DAgger-collection variance is not resampled (one aggregation, 61,082 steps) — a known,
  documented scope choice; M5's full comparison resamples the whole pipeline per seed.

## Reproduce (Windows, conda nmc)
```
python scripts/train_mlp_go2.py                       # BC pretrain (512x512)
python scripts/dagger_go2.py --iters 2 --episodes 40  # DAgger -> data/imitation_go2_dagger.npz
python scripts/train_seeds_mlp_go2.py --seeds 5       # 5 seed-models
python scripts/eval_mlp_go2.py --episodes 30          # SPL + CIs + figures + online check
```
