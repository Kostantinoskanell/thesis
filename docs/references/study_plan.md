# Study Plan — what to learn, in order

Paced to the milestones so you study each topic just before you need it. ~ = priority.

## Now (while M1b/M2 are built) — foundations
- ~~ **Surrogate-gradient SNN training** — Neftci, Mostafa & Zenke (2019). Understand
  why we can't backprop through a spike and what the surrogate fixes. You'll need this
  for M3 (SNN pretraining).
- ~~ **snnTorch tutorials** (Eshraghian) — work through the LIF + surrogate-gradient
  notebooks end to end. This is the tool you'll live in.
- ~ **STDP basics** — Bi & Poo (1998) + any STDP tutorial. You already have the math;
  make sure you can explain LTP/LTD windows from memory (defense question).

## Before M4 (the pilot) — the core intellectual content
- ~~ **e-prop** — Bellec et al. (2020), *Nature Comms*. This is the SOTA online rule and
  the biggest single idea in the thesis. Read the paper, then read Frenkel's
  eprop-PyTorch code. Be able to explain the three factors (eligibility, learning
  signal, and the local pseudo-derivative).
- ~~ **Three-factor learning rules** — read the STDP↔e-prop bridge (arXiv:2201.07602)
  so you can defend "R-STDP and e-prop are the same hardware, different third factor."
- ~ **Reward shaping / TD targets** — enough RL to explain the online-MLP baseline's
  TD/advantage loss and the R-STDP reward signal (they use the same signal).
- ~ **Continual-learning SNN path planning** (arXiv:2404.15524) — the closest prior work
  to our recovery-under-shift experiment; know how they measure adaptation.

## Before M7–M8 (FPGA) — hardware
- ~~ **FPGA SNN survey** (arXiv:2307.03910) + **SNNs-on-FPGA survey** (Neural Networks
  2025). Understand DSP-slice budgeting, BRAM LUTs, and time-multiplexed PEs.
- ~~ **AXI4-Stream + DMA + PyNQ** — the Xilinx PYNQ overlay tutorials. This is where
  most students lose weeks; start the "hello-world DMA" overlay early (even before the
  board arrives, in Vivado sim).
- ~ **Fixed-point arithmetic & quantization** — enough to validate the 64-bin log LUT
  against our `kernel_reference` (<2% error gate).
- ~ **Spiker-LL / Minitaur** architectures — concrete local-learning FPGA datapaths.

## Cross-cutting (any time)
- **Memristor framing** — Shooshtari (2025) review; be able to justify the structural
  analogy *and* its limits (you'll be asked "is this really a memristor?" — answer: no,
  it's memristor-*inspired*, here's exactly why).
- **Experimental rigor** — how to report mean ± 95% CI over seeds and run a significance
  test on recovery time. A committee will push on statistical validity.

## Skills to build in parallel
- Git discipline (commit per milestone), Weights & Biases logging, LaTeX/TikZ for
  figures, and reproducible seeded experiments (already scaffolded in `configs/`).
