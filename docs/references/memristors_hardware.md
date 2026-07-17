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

## Real-device option (characterization add-on — see D5 in sota_decisions.md)
Real memristors are purchasable and thesis-viable as a **complement** to the FPGA, not a
replacement. Off-the-shelf source (2026) is **Knowm**: discrete SDC devices (8 in 16-DIP,
16 in 32-DIP) and small crossbars up to 32×32 on 64-pin edge boards. Ceiling ~1024
synapses, stochastic, drift over cycling — too small/noisy to *be* the controller, but
ideal to **measure real STDP conductance-change curves, fit the synapse model, and load
the measured parameters into the FPGA update rule** ("R-STDP validated against real device
physics"). Gating resource: an SMU (Keithley 2400/2600) for controlled pulsing + readout —
check the ECE Patras electronics lab. Est. ~2–4 weeks incl. instrument time.
- Store: https://knowm.com/collections/all
- 8-Discrete 16-DIP: https://knowm.com/products/m-sdc-memristor-8-discrete-16-dip
- W+SDC crossbars: https://knowm.com/products/w-sdc-memristor-crossbars

## Honest-scope reminder
We are **memristor-inspired**, not device-level. These device papers justify *why* the
structural analogy is faithful (bounded, history-dependent, stochastic conductance) —
they are not claims that our FPGA reproduces device physics. Keep that line in the text.
(The D5 real-device add-on *grounds* the analogy with measured parameters; it does not
change the memristor-inspired framing of the compute substrate.)
