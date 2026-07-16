# Memristors & Neuromorphic Hardware

## Foundational (in proposal)
- **Strukov et al. (2008)**, *Nature* — "The missing memristor found."
- **Biolek et al. (2009)** — SPICE memristor model with window function (our Eq. 6 reference).
- **Merolla et al. (2014)** TrueNorth; **Davies et al. (2018)** Loihi.

## Recent reviews (the hardware-bridge framing)
- **Shooshtari et al. (2025/2026)**, *Advanced Intelligent Systems* — "Review of
  Memristors for In-Memory Computing and SNNs." Our main framing citation for
  memristor↔crossbar↔STDP. https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202500806
- **Brain-Inspired Computing Based on Large-Scale Memristor Crossbar Arrays (2026)**,
  *Adv. Functional Materials* — crossbar structures & history.
  https://advanced.onlinelibrary.wiley.com/doi/10.1002/adfm.202528309
- **Memristor Synapse — A Device-Level Critical Review** — STP/LTP/STDP/SRDP device
  properties; good for our "three memristive properties" table.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12899887/

## Device non-idealities (grounds our stochasticity/noise modelling)
- **On-Chip Learning with Memristor-Based NNs: Accuracy & Efficiency Under Device
  Variations, Conductance Errors, and Input Noise (2024)**, arXiv:2408.14680 — directly
  supports our LFSR-noise / additive-weight-noise "stochasticity" property and the
  robustness sweep (H4). https://arxiv.org/abs/2408.14680

## Honest-scope reminder
We are **memristor-inspired**, not device-level. These device papers justify *why* the
structural analogy is faithful (bounded, history-dependent, stochastic conductance) —
they are not claims that our FPGA reproduces device physics. Keep that line in the text.
