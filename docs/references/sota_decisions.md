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

## D6. Real-hardware deployment: sport-mode SDK  vs  low-level SDK (custom policy)  (DEFERRED — sport-mode primary, low-level as end-of-thesis stretch goal)

**Question raised:** a classmate pointed out `unitree_sdk2_python` exposes a **low-level**
interface — `LowState_`/`LowCmd_` DDS topics (pub/sub, analogous to ROS topics) — giving
direct IMU + per-joint encoder readout and direct per-motor torque/position commands,
bypassing Unitree's onboard walking firmware entirely. Should we deploy through this
instead of sport-mode?

**What each path actually does:**
- **Sport-mode SDK** (current plan, [locomotion.md](locomotion.md), ROADMAP D2/D3): send
  `Move(vx, vy, vyaw)`; the Go2's **own onboard gait** walks it. Our trained MuJoCo
  Playground policy (D2, `Go2JoystickFlatTerrain`) is never touched on real hardware —
  only the high-level SNN's velocity commands reach the robot.
  Note: not currently exercised by any milestone — D3's real-Go2 deploy is listed as the
      final integrated/hardware demo, after the M2–M8 science.
- **Low-level SDK**: our own trained policy's joint targets are sent directly, at
  control-loop rate, using real IMU/joint sensors as the observation.

**Why this reopens a locked decision, not a free upgrade:** [locomotion.md](locomotion.md)
explicitly chose sport-mode and ruled out "*implementing/training a learned locomotion
policy ourselves — that would be a whole second thesis*." Low-level deployment requires,
at minimum:
- **A state estimator.** Our policy's `local_linvel` observation (D2's `go2/joystick.py`)
  is a ground-truth MJX velocimeter reading; a real IMU only gives gyro + accelerometer,
  not body-frame velocity — this must come from leg-odometry/EKF fusion, which does not
  exist yet in any form here.
- **Safety infrastructure**: torque limits, fall detection, E-stop — a real robot can be
  damaged by an under-randomized sim2real policy.
- **Sim2real robustness work** (more aggressive domain randomization / iterative tuning) —
  the actual bulk of effort in papers like unitree_rl_gym / Isaac Lab Go2 (see
  [locomotion.md](locomotion.md)'s SOTA list), not a quick add-on.

**Decision:** **keep sport-mode as the real-hardware target for the thesis's core
deliverable.** Log low-level SDK deployment (closing the sim2real loop with the D2 trained
policy) as a **documented stretch goal** for a closed-loop hardware demo near the end
(post-M9-ish, only if time remains) — it would make a substantially stronger demo, but
must not compete with the R-STDP pilot (M4, the actual go/no-go gate) for time.

**Practical field notes** (from two labmates who use the SDK, 2026-07-17):
- Package is `unitree_sdk2_python` (import `unitree_sdk2py`); low-level = `LowState_`
  (subscribe) / `LowCmd_` (publish) DDS topics.
- You must write **handler/callback plumbing** to consume the DDS streams correctly —
  it is not turnkey.
- Keep the **official Unitree handbook** at hand: some command flags are **silently
  ignored** by the firmware, which wastes days if you don't know which.
- Their "sim commands run directly on the real robot" claim is optimistic: `LowState`
  gives IMU (gyro/accel/quat), joint encoders, foot force — **not body-frame velocity**,
  which our policy's `local_linvel` observation requires (ground-truth velocimeter in
  MJX). Bridging needs leg-odometry/EKF estimation or retraining without that obs.
- Confirmed by lab usage: **sport mode's built-in gait is the standard tool when the
  work is path-planning-level** — which is exactly our SNN-navigator architecture.

**Revisit when:** the M2–M8 plasticity science is done and a hardware demo slot opens up
with time to spare; re-evaluate against how much of the FPGA track (D-track) is still
pending at that point. Note the D2 trained Go2 policy is the exact artifact a low-level
deployment would use — nothing in the current plan forks on this decision, so deferring
costs nothing.

---

## D7. Baseline strength: "SOTA baselines" vs "fair baselines"  (RESOLVED — strong *and* fair, upgrades that also improve fairness)

**Question raised:** should the M2 MLP baselines be upgraded with more SOTA machinery?

**Decision:** Yes — but the governing criterion for a *baseline* is **strong-and-fair, not
maximal**. The thesis contribution is the R-STDP controller; the baselines exist to make
the comparison rigorous. A baseline upgrade is worth doing when it (a) is genuinely SOTA
*and* (b) keeps the comparison apples-to-apples (or makes it fairer). Upgrades that would
make a baseline asymmetrically strong in ways the SNN can't match (privileged features,
recurrence/memory the SNN lacks, expert labels post-shift) are **rejected as unfair**, not
embraced as "more SOTA."

**Applied upgrades (M2):**
1. **Online MLP → eligibility-trace TD(λ) actor-critic** (was one-step TD). This is both
   more SOTA *and fairer*: R-STDP's mechanism is itself an eligibility trace gated by a
   global reward (proposal Eq. rstdp), so giving the online MLP eligibility traces makes
   the two adaptive controllers structurally parallel — they differ only in substrate.
   λ=0 exactly recovers the proposal's original one-step baseline, so nothing is lost.
2. **SPL** (Success weighted by Path Length, Anderson et al. 2018, arXiv:1807.06757) +
   collision rate — the standard embodied-navigation metric suite, reused across all five
   controllers in M5. Success rate alone can't see detour inefficiency.
3. **Multi-seed + 95% CIs** (Wilson for a single controller's success count; t-interval
   across seed-models). A single-seed point estimate is not defensible at a defense; the
   ROADMAP already mandates this for M5, adopted early here.
4. **MLP capacity parity with the LIF-SNN** (512×512 + LayerNorm). Removes the "the
   baseline was under-powered" objection.

**Rejected (this round):** continuous action space `[v, ω]`. More SOTA for navigation, but
the proposal locked discrete 4-action (population-vote SNN decoder) and it would cascade
through D3/M2–M8 — scope disproportionate to a part that isn't the contribution. See D4.

**Principle for the rest of the thesis:** apply the same test to every baseline/ablation
upgrade — SOTA *and* fair, symmetric across controllers.

---

_Update this log whenever a new SOTA option is identified. Every "we chose the simpler
thing" must have an entry saying why and when to revisit._
