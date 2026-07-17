# SOTA Decision Log

Where the state-of-the-art option differs from the conventional one. Standing rule:
**pick SOTA unless it conflicts with a locked decision — and if it does, surface the
trade-off here rather than silently choosing.**

---

## D1. Online learning rule: R-STDP  vs  e-prop  (OPEN — recommend upgrade)

**Conventional / currently locked:** reward-modulated STDP (R-STDP) — a three-factor
rule: Hebbian eligibility trace × global reward.

**SOTA:** **e-prop** (Bellec et al., *Nature Comms* 2020). Also a three-factor rule, but
the eligibility trace is derived as an online approximation of BPTT and the third factor
is a per-neuron *learning signal* (error projection), not just a global scalar reward.
e-prop approaches BPTT accuracy and is the current standard for online RSNN learning.

**Why this is a soft conflict, not a hard one:** both rules have the identical hardware
shape — *per-synapse eligibility register × broadcast third factor*. The FPGA datapath
we scoped (eligibility trace + scalar/vector gate) supports **both**. The STDP↔e-prop
bridge (arXiv:2201.07602) makes them formally two instances of one family.

**Recommendation (per SOTA directive):** frame the learning rule as a **general
three-factor eligibility-trace rule** and implement **two instances**:
1. **e-prop** — the SOTA instance, primary for the H1 recovery claim.
2. **classic R-STDP** — the biologically-simplest instance, kept as the
   "memristor-native" comparison (a global dopamine-like scalar).
3. **pure STDP** — the Hebbian ablation (already planned).

This *strengthens* the thesis: same hardware, three points on a
biological-plausibility ↔ learning-power axis, and it keeps the memristor-crossbar
story (three-factor local rule on a crossbar) fully intact.

**Cost:** e-prop needs a per-neuron learning signal (a small extra broadcast vector) and
the surrogate-derivative eligibility form — modest software work, ~1 extra register/PE
in hardware. **Decision needed from advisor/student before M4.**

---

## D2. LIF neuron model: vanilla LIF  vs  adaptive LIF (ALIF)

**SOTA:** e-prop's headline results use **adaptive LIF** (ALIF) neurons (with an
adaptive threshold), which give the network longer temporal memory. If we adopt e-prop
(D1), ALIF is the matching neuron model. Low cost in snnTorch. Consider for M3.

---

## D3. Robot platform: kinematic unicycle  vs  Unitree Go2 (RESOLVED — Go2, two-layer)

**Decision:** target the **Unitree Go2** quadruped (user has one) via a **two-layer,
decoupled** design:
- High-level: the SNN outputs velocity commands `[vx, vy, omega]` (unchanged interface).
- Low-level (sim): a convex-MPC locomotion controller walks the robot in PyBullet
  (prototype with the built-in A1 URDF; swap Go2 URDF later). See
  [locomotion.md](locomotion.md).
- Low-level (real): the Go2's onboard SDK sport mode walks it from the same velocity
  commands — no custom gait deployed to hardware.

**Why decoupled:** legged locomotion is a separate hard problem, orthogonal to the
neuromorphic-plasticity contribution. The plasticity science (M2–M8) stays on the fast
kinematic model; the Go2 layer is for realistic visuals + a final integrated/hardware
demo. This is the "Platform track" (P1–P3) in ROADMAP.md.

**Not chosen:** implementing/​training a learned locomotion policy ourselves (Isaac Lab /
unitree_rl_gym) — that would be a whole second thesis and the real Go2 already walks
itself.

---

## D4. Encoding: rate + TTFS  vs  learned/latency encoding

Rate coding is simple but throws away temporal structure STDP needs (a known risk).
SOTA event-driven work uses **learned or latency-based encoders**. Revisit if the M4
pilot shows the plasticity has no temporal signal to exploit.

---

## D5. Synapse substrate: memristor-inspired FPGA  vs  real memristor  (RESOLVED — FPGA primary, real device as characterization add-on)

**Question raised:** could a real memristor replace the memristor-inspired FPGA synapse
array?

**Decision:** **No full replacement — keep the FPGA as the synapse array, and add one
discrete real memristor as a bounded device-characterization study.** Frame it as a
*hybrid*: measure real device behaviour, fit the synapse model, load the measured
parameters into the FPGA update rule.

**What is actually purchasable (2026):** [Knowm](https://knowm.com/collections/all) is
the only off-the-shelf source. Self-Directed-Channel (SDC) discrete devices — 8 in a
16-DIP or 16 in a 32-DIP (breadboard-friendly) — and small crossbars up to **32×32** on
64-pin edge boards. "Burn & Learn" research chips are intermittently stocked at reduced
cost. Robot-controller-scale memristive crossbars exist only in fabs (IBM, HP, Tsinghua),
not for sale.

**Why full replacement fails for this thesis:**
- **Scale** — a 32×32 crossbar caps at 1024 synapses; an R-STDP locomotion network needs
  orders of magnitude more.
- **Variability / endurance** — SDC devices are stochastic with large device-to-device
  spread and conductance drift over potentiation/depression cycles. Fine for a
  characterization study, brutal for a controller training in closed loop for hours.
- **Instrumentation burden** — proper pulsing needs controlled amplitude/width, current
  compliance, and precise readout (an SMU like a Keithley 2400/2600, or a careful
  DAC/ADC + op-amp front-end off the FPGA). A mixed-signal sub-project on its own.

**Why the hybrid is the *stronger* thesis (per SOTA directive):** most memristor-inspired
FPGA work uses an idealized synapse model (linear / Biolek window). Measuring real STDP
conductance-change curves on an 8-device DIP (~$300–400), fitting the model, and loading
*those measured parameters* into the FPGA synapse update rule supports a claim few
undergrad theses can make: **"R-STDP controller validated against real device physics."**
This directly reinforces D1's memristor-native R-STDP instance and the crossbar story,
and keeps the honest-scope line in [memristors_hardware.md](memristors_hardware.md)
(we remain memristor-*inspired*; the real device grounds the analogy, it is not the
compute substrate).

**Cost / gating:** ~2–4 weeks including instrument time; **bounded add-on, not a
re-architecture.** Gating resource is an SMU — check the ECE Patras
electronics/microelectronics lab for a bookable unit *before* committing. Order the
Knowm chip early (ships from the US; research stock is intermittent). Crossbar-scale
all-analog version → **future work**.

Sources: [Knowm 8-Discrete 16-DIP](https://knowm.com/products/m-sdc-memristor-8-discrete-16-dip),
[W+SDC crossbars PCIE-64](https://knowm.com/products/w-sdc-memristor-crossbars),
[Burn & Learn 16-Discrete 32-DIP](https://knowm.com/products/burn-learn-m-sdc-memristor-16-discrete-32-dip).

---

_Update this log whenever a new SOTA option is identified. Every "we chose the simpler
thing" must have an entry saying why and when to revisit._
