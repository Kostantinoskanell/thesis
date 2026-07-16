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
  https://arxiv.org/abs/2605.18003
- **Neil & Liu (2014)**, Minitaur — FPGA SNN with on-chip STDP + resource multiplexing.
  (In proposal.)
- **Energy-Aware FPGA Implementation of SNN with LIF Neurons (2024)**, arXiv:2411.01628.
  https://arxiv.org/abs/2411.01628

## Context
- Loihi (Davies 2018, in proposal) supports reward-modulated STDP on-chip but is
  proprietary — this is exactly the gap an **open FPGA** R-STDP/e-prop accelerator fills.
