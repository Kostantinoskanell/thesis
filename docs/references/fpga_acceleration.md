# FPGA SNN Acceleration & On-Chip Learning

## Surveys (start here — map the design space)
- **SNNs on FPGA: methodologies and recent advancements (2025)**, *Neural Networks*
  vol. 186 — the current survey of FPGA SNN methods. https://doi.org/10.1016/j.neunet.2025.107256
- **A Survey of SNN Accelerator on FPGA (2023)**, arXiv:2307.03910 — taxonomy of
  architectures, incl. resource multiplexing (our time-multiplexed PE design).
  https://arxiv.org/abs/2307.03910
- **A Quarter Century of Neuromorphic Architectures on FPGAs — Overview (2025)**,
  arXiv:2502.20415. https://arxiv.org/abs/2502.20415

## On-chip / local learning accelerators (our M7–M8 target)
- **Spiker-LL (2026)**, arXiv:2605.18003 — energy-efficient FPGA accelerator with
  *adaptive local learning* (extends the open-source Spiker+ inference core). Closest
  open architecture to what we build; check their local-learning datapath.
- **FireFly-P (2026)**, arXiv:2601.21222 — **FPGA-accelerated SNN plasticity for adaptive
  control** — the most direct comparison point for H3. Concrete numbers to beat/cite:
  **8 µs end-to-end latency (inference + plasticity update), 0.713 W, ~10K LUTs** on a
  tiny Cmod A7-35T. Uses a forward engine + a plasticity engine (same split we scoped);
  their plasticity rule is meta-learned offline via Evolutionary Strategy then run online.
  Use their latency/power/LUT figures as the H3 baseline our R-STDP datapath is measured
  against. https://arxiv.org/abs/2601.21222
- **Low-cost FPGA spiking ELM with on-chip R-STDP (2021)** — proves on-chip R-STDP on
  modest FPGA resources; a resource-budget reference point.
  https://www.researchgate.net/publication/355102502
  https://arxiv.org/abs/2605.18003
- **Neil & Liu (2014)**, Minitaur — FPGA SNN with on-chip STDP + resource multiplexing.
  (In proposal.)
- **Energy-Aware FPGA Implementation of SNN with LIF Neurons (2024)**, arXiv:2411.01628.
  https://arxiv.org/abs/2411.01628

## Context
- Loihi (Davies 2018, in proposal) supports reward-modulated STDP on-chip but is
  proprietary — this is exactly the gap an **open FPGA** R-STDP/e-prop accelerator fills.
