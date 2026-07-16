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
