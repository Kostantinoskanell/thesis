# M0 — Repo scaffold + verified plasticity golden reference

_Archived 2026-07-17_

- STDP/R-STDP online learner implemented in NumPy (golden reference for the FPGA).
- 6/6 unit tests green: LTP/LTD signs, weight saturation, R-STDP reward gating.
- `fig_stdp_kernel.png` — the biexponential STDP window from `kernel_reference`.
- `fig_rstdp_demo.png` — eligibility trace accumulates, then a reward event consolidates it into a weight change (the mechanism H1 rests on).
