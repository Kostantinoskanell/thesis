# M3 — LIF-SNN surrogate-gradient pretraining matches the MLP pre-shift

_Archived 2026-07-18 · controller 3 (frozen SNN) + the substrate M4 releases to R-STDP._

The LIF-SNN is trained by **surrogate-gradient BPTT** (snnTorch, fast-sigmoid) on the
**same DAgger dataset** the MLP used (`data/imitation_go2_dagger.npz`, 61k steps), with the
same class-weighted cross-entropy — but on **population-vote logits** (one neuron
population per discrete action, proposal Sec. 3.2). Observations are spike-encoded with
`encode_nav_obs` (rate-coded LiDAR nearness + TTFS urgency + on/off-split goal/heading/
yaw-rate), so the SNN sees exactly the MLP's information.

## SOTA neuron: ALIF (D2)
Adopted **adaptive-LIF** (Bellec et al. 2020: spike-triggered decaying threshold) as a
custom `ALIFCell` — snnTorch has no built-in ALIF. Motivation is thesis-specific, not
buzzword: ALIF gives each neuron temporal memory, directly addressing the proposal's own
risk that "rate-coded LiDAR lacks temporal structure STDP can exploit." Vanilla LIF kept
as a one-seed reference.

## Result — frozen SNN, closed-loop, pre-shift (shift disabled)
Same harness/metrics as M2 (Wilson-95% on pooled success, SPL, held-out seeds 2000+):

| controller | seeds | success (pooled) | SPL | firing rate | spikes/decision |
|---|---|---|---|---|---|
| **ALIF SNN** | 3 × 30 eps | **41%**, Wilson [32, 51] | 0.318 | **1.4%** | **312** |
| LIF SNN (ref) | 1 × 30 eps | 37%, Wilson [22, 54] | 0.330 | 1.9% | 412 |
| MLP (M2) | 5 × 30 eps | 37%, Wilson [29, 45] | 0.346 | — (dense) | — |

**Exit criterion PASSED:** the SNN matches/beats the MLP pre-shift (41% vs 37%). Two
findings worth stating at a defense:
1. **Offline accuracy ≠ closed-loop success (again).** The SNN's offline val-accuracy was
   ~63% (ALIF) / 68% (LIF) — well below the MLP's ~86% — yet closed-loop it is on par.
   Reinforces M2's lesson: only closed-loop evaluation counts.
2. **ALIF is more spike-efficient than LIF** (1.4% vs 1.9% firing; ~25% fewer spikes) at
   equal-or-better success — the adaptive threshold integrates over time so fewer spikes
   carry the same decision. This is the H2 energy story in preview (the dense MLP does a
   full matrix-multiply every step; the SNN is ~99% silent).

**Honest caveat:** 3 SNN seeds → wide across-seed CI (±27%; per-seed 33/37/53%). The
pooled Wilson [32,51] is the number to quote. **M5 will run ≥10 seeds on the GPU
env** (`nmc-snn` WSL, torch-CUDA, verified) — CPU BPTT here was ~47 min for all 4 seeds
in parallel (20 cores, thread-capped; ~3 h if sequential).

Artifacts: `fig_snn_eval.png` (ALIF success+SPL with CI, per-seed spread vs MLP),
`snn_episode.gif` (successful run), `fig_snn_training_seed*.png` (training curves),
`lif_ref/` (LIF reference eval). Models: `assets/snn_seeds/snn_seed{0,1,2}.pt` (ALIF),
`assets/snn_seeds_lif/` (LIF).

## Reproduce
```
conda run -n nmc bash scripts/train_snn_seeds_parallel.sh   # 3 ALIF + 1 LIF, parallel
conda run -n nmc python scripts/eval_snn_go2.py --episodes 30 --seeds-dir assets/snn_seeds
```
Next: **M4** releases these weights to online R-STDP (`SNNNavController(plasticity_enabled=True)`)
and measures recovery vs frozen-SNN / online-MLP after the mid-episode shift — the
go/no-go gate.
