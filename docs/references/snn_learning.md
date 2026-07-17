# SNN Training & Online Learning Rules

## Foundational / already in the proposal
- **Maass (1997)**, *Neural Networks* — spiking neurons as the "third generation." Framing.
- **Bi & Poo (1998)**, *J. Neuroscience* — original STDP measurement. The rule in Eq. (5).
- **Eshraghian et al. (2023)**, *Proc. IEEE* — snnTorch; our primary SNN framework.
  https://arxiv.org/abs/2109.12894

## Surrogate-gradient pretraining (our SNN init)
- **Neftci, Mostafa & Zenke (2019)**, *IEEE Signal Processing Magazine* — "Surrogate
  Gradient Learning in SNNs." The canonical reference for the pretraining step.
  https://arxiv.org/abs/1901.09948

## Online learning — the SOTA axis (READ THESE)
- **Bellec et al. (2020)**, *Nature Communications* — **e-prop** ("A solution to the
  learning dilemma for recurrent networks of spiking neurons"). Three-factor,
  eligibility-trace, online approximation of BPTT. The SOTA alternative/upgrade to
  classic R-STDP. https://www.nature.com/articles/s41467-020-17236-y
- **Frenkel — eprop-PyTorch** — clean reference implementation we can build on.
  https://github.com/ChFrenkel/eprop-PyTorch
- **Including STDP in eligibility propagation (2022)**, arXiv:2201.07602 — explicitly
  bridges STDP and e-prop; the theoretical link that lets us treat R-STDP and e-prop
  as two instances of one three-factor rule on the *same* FPGA datapath.
  https://arxiv.org/abs/2201.07602
- **Juarez-Lora et al. (2022)**, *Front. Neurorobotics* — R-STDP for a changing-friction
  robot arm; the closest prior R-STDP-for-adaptation result. (In proposal.)

## Key insight for us
R-STDP and e-prop share the structure **eligibility trace (per synapse) × scalar
learning/​reward signal**. That is exactly the FPGA datapath we already scoped
(eligibility register + broadcast multiply), so upgrading the rule does not change the
hardware architecture — see [sota_decisions.md](sota_decisions.md).

## Lit-check findings (2026-07-18) — where our M4 logic sits vs the field
- **Reward-prediction-error modulation is standard, not exotic.** Multiple R-STDP-nav
  papers use dopamine-style reinforcement; baseline subtraction (our D8 RPE) is the
  mainstream three-factor formulation. Our approach is aligned.
- **Stability-plasticity dilemma is THE named risk** for online STDP: STDP *is*
  forgetting-prone (STDP susceptible to catastrophic forgetting under non-stationary
  input). Standard mitigations: data reinforcement/replay, controlled forgetting via
  dopaminergic modulation (Allred & Roy, *Front. Neurosci.* 2020), consolidation. Our
  guardrail (does plasticity destroy the policy?) IS a stability test; M4 now also
  measures **retention** (base-task success after shift-adaptation) — the "stability"
  half. Frame H1 explicitly in stability-plasticity terms.
- **Online test-time adaptation (TTA) for SNNs via *threshold* modulation**
  (arXiv:2505.05375) — a recent alternative mechanism for distribution-shift recovery
  that adapts neuron thresholds, not weights. Connects to our ALIF adaptive threshold;
  candidate extra baseline/ablation for M5.
- **Astrocyte-gated multi-timescale plasticity (AGMP, 2025)** — augments eligibility
  traces with a broadcast teaching signal; essentially the e-prop direction we deferred
  (D1). Confirms our framing is current.
- **Task-difficulty caveat:** canonical R-STDP-nav demos report ~95% but on *reactive*
  obstacle avoidance (turn away from near obstacle). Ours is goal-directed nav + A*
  teacher + mid-episode shift + full Go2 dynamics — a materially harder problem, so
  absolute success is not comparable; the *relative* R-STDP-vs-baseline gap is the claim.

Sources: R-STDP nav — [autonomous learning R-STDP robot](https://www.sciencedirect.com/science/article/abs/pii/S0925231221009310),
[R-STDP target-reaching vehicle (Frontiers 2019)](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2019.00018/full);
stability-plasticity — [Controlled Forgetting (Front. Neurosci. 2020)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00007/full),
[loss of plasticity in continual RL](https://arxiv.org/pdf/2303.07507);
SNN TTA — [threshold modulation](https://arxiv.org/pdf/2505.05375);
online continual SNN — [AGMP](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1768235/full).
