# SOTA Decision Log

Where the state-of-the-art option differs from the conventional one. Standing rule:
**pick SOTA unless it conflicts with a locked decision — and if it does, surface the
trade-off here rather than silently choosing.**

---

## D1. Online learning rule: R-STDP  vs  e-prop  (RESOLVED — e-prop implemented and tested, see D19)

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
in hardware.

**Decision (2026-07-17): keep R-STDP primary for M4; add e-prop as a second three-factor
instance in M5, NOT before the pilot.** Reasoning: (1) M4 is the go/no-go gate that tests
whether reward-modulated plasticity works at all — that is the thesis's core claim and
must be tested cleanly on the memristor-native rule (R-STDP) first. (2) e-prop's
*per-neuron* learning signal is less memristor-crossbar-native than R-STDP's single
*global* reward broadcast, so making e-prop primary would partially undercut RQ2/RQ3 (the
FPGA/hardware contribution). (3) Rushing a large e-prop implementation in before M4 risks
delaying the pivotal experiment. The three-factor framing (this entry) is preserved: the
same eligibility-trace datapath serves both, so e-prop enters as a learning-power
reference in the M5 full comparison once R-STDP is proven. ALIF (D2) — the neuron model
e-prop would use — is adopted now, so the substrate is already e-prop-ready.

---

## D2. LIF neuron model: vanilla LIF  vs  adaptive LIF (ALIF)  (RESOLVED — ALIF adopted for M3)

**SOTA:** e-prop's headline results use **adaptive LIF** (ALIF) neurons (with an
adaptive threshold), which give the network longer temporal memory.

**Decision (2026-07-17):** **adopt ALIF as the M3 neuron model.** Motivation is not just
"more SOTA" — ALIF directly attacks the proposal's *own* open risk that "rate-coded LiDAR
lacks temporal structure STDP can exploit": the spike-triggered decaying threshold gives
each neuron memory across timesteps, so the plasticity has temporal signal to work with.
It *strengthens* the R-STDP story rather than competing with it, and is the natural
partner to R-STDP (unlike e-prop, D1).

**Correction to the earlier note:** "low cost in snnTorch" was wrong — snnTorch has **no
built-in ALIF** (`Leaky.learn_threshold` is a *learnable fixed* threshold, not adaptive).
Implemented as a compact custom `ALIFCell` (Bellec et al. 2020 dynamics) in
`src/nmc/controllers/snn.py`, drop-in via `LIFNet(neuron="alif")`, same fast-sigmoid
surrogate, same weights/decode. Vanilla LIF kept selectable (`neuron="lif"`) as a
one-seed reference so the LIF→ALIF benefit can be reported.

**Cost:** re-pretrain M3 with ALIF (~45 min/seed on CPU). Bounded.

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

## D8. R-STDP third factor: raw reward  vs  reward-prediction error (RPE)  (RESOLVED — RPE, raw kept as ablation)

**Question raised (M4 pilot):** what signal should modulate the R-STDP eligibility trace?

**Decision:** **reward-prediction error `r - r̄`** (r̄ = EMA baseline), i.e. dopamine-style
baseline subtraction, as the default third factor. Raw env reward kept selectable as an
ablation (`reward_mode="raw"`).

**Why it's likely necessary, not cosmetic:** the env reward is progress-to-goal, which is
**almost always slightly positive**. Raw-reward R-STDP then *uniformly potentiates*
whatever the network just did every step — it can't distinguish a good action from a
slightly-worse one, so it drifts rather than improves. Subtracting a running baseline
makes plasticity potentiate only **better-than-expected** outcomes and depress
worse-than-expected — the standard modern three-factor / neuromodulated-plasticity
formulation (dopamine RPE, Schultz).

**Also fairer:** the online-MLP baseline (D7) already uses an advantage (value-subtracted)
signal via its critic. Raw-reward R-STDP vs advantage-MLP would be an unfair mismatch;
RPE puts both adaptive controllers on a baseline-subtracted footing. Same strong-and-fair
test as D7.

**Cost:** one EMA scalar per controller (`rpe_alpha`), computed at consolidation in
`SNNController.learn`. STDPConfig (the FPGA golden reference) is untouched — RPE is a
controller-level modulation of the scalar third factor, so the hardware datapath is
unchanged. Implemented; raw is the ablation for the writeup ("why RPE was needed").

---

## D9. Plasticity site + stability: readout-only  vs  input-layer + anchoring  (RESOLVED — input+readout plasticity with weight anchoring)

**Question (M4 pilot):** where should R-STDP act, and how to keep it stable?

**Decision:** apply R-STDP to the **input layer AND the readout** (not readout-only), with
**elastic weight anchoring** (pull toward the pretrained weights each step) for stability.

**Why input-layer plasticity (evidence-driven):** the M4 sensor-dropout test corrupts the
navigator's *input* (dead LiDAR beams). Readout-only plasticity **provably cannot re-map a
broken input** — it stalled at 7% while the full-backprop online-MLP (which adapts input
layers) recovered early. Extending R-STDP to `fc[0]` (input) moved it **7% → 30%**,
matching the best baseline. The proposal named this ("readout first… extend to hidden
layers"); M4 gave the concrete reason.

**Why anchoring:** unregularized continual adaptation *over-adapts and collapses* — the
online-MLP recovered to 0.8 then fell to 0 with catastrophic forgetting (7% base
retention). Anchoring (`W += anchor·(W0 − W)` per step) pulls weights back toward the
pretrained anchor, resolving the stability-plasticity dilemma: R-STDP kept 43% base
retention (highest) *and* 30% shifted success. This is the literature's "controlled
forgetting" mitigation, in a local-plasticity-friendly form (a per-synapse decay toward a
stored value — hardware-cheap, unlike replay).

**Implementation:** `SNNController` now takes `plastic_layers` (per-layer weight/learner/
anchor) and `anchor`; per-layer saturation bounds from each layer's own weight scale.
FPGA note: multi-layer + a decay-to-anchor term are still local per-synapse ops — the
crossbar datapath is unchanged (one extra register for W0 per synapse).

**Status:** positive **signal** at n=30 (CIs overlap); M5's ≥10-seed GPU runs confirm
significance and can tune anchor/η.

---

## D10. Extra M4b baseline: TM-NORM (reward-free threshold calibration)  (RESOLVED — add as 6th controller)

**Question:** R-STDP's recovery under sensor dropout could in principle be mostly
*statistical renormalization* to the corrupted input's new scale, rather than genuine
reward-driven relearning. Is there a cheap way to rule this out?

**Found:** Zhao et al. 2025 ([snn_learning.md](snn_learning.md), arXiv:2505.05375) —
**Threshold Modulation (TM-NORM)**: an online test-time adaptation method for SNNs that
recalibrates each neuron's firing threshold from an EMA of its own membrane-potential
mean/variance (batch-norm-on-the-membrane-potential, re-parameterized into the threshold).
**No backprop, no reward signal, no weight update at all** — pure unsupervised statistical
calibration, trivial to bolt onto the existing ALIF cells.

**Decision:** add **TM-NORM as a 6th controller** in M4b/M5. If it recovers close to
R-STDP's 30% under the same sensor-dropout shift, that undercuts the "reward-modulated
plasticity" claim (H1) — the recovery would be attributable to renormalization, not
learning. If it recovers substantially less, that's direct evidence the reward-driven part
of R-STDP is doing real work. Either outcome strengthens the thesis's honesty; this is a
cheap, sharp stress test (no training loop needed, just EMA stat tracking at inference
time), so there is no reason not to run it before claiming H1.

**Also adopted (M6):** the same paper's energy-accounting method — per-operation counts
(MACs/ACs/MULs) × published per-op energy costs (Horowitz 2014, 45nm: 0.9pJ/AC, 3.7pJ/MUL,
4.6pJ/MAC) — replaces a bare firing-rate percentage with an actual µJ-per-decision estimate
for the H2 energy claim.

---

## D11. Spiking locomotion actor: rate readout vs population coding (RESOLVED — PopSAN population coding)

**Context (L-track):** training a *spiking* Go2 locomotion policy with RL (Isaac Lab / rsl_rl
PPO) so plasticity can reach the gait (M4c ice ceiling). First cut (`spiking_actor.py`, L2)
used a plain firing-rate readout over the last hidden layer.

**Lit review (2026-07-21) — thorough read of the SOTA:**
- **PopSAN** (Tang et al. 2020, arXiv:2010.09635; code combra-lab/pop-spiking-deep-rl): the
  canonical spiking actor for continuous-control DRL. **Population coding** on both ends —
  each observation dim is encoded by a population of neurons with *learnable Gaussian
  receptive fields* (μ,σ trainable), and each action dim is decoded from its own output
  population by *learnable weights* over the T-step firing rate. Hybrid: **spiking actor +
  deep-MLP critic**; integrates with DDPG/TD3/SAC and **PPO** (actor predicts the action mean,
  trained by the clipped surrogate loss). Matches deep actors on MuJoCo control; 140× less
  energy on Loihi vs Jetson TX2.
- **Key ablation (decisive for us):** PopSAN's own "RateSAN" baseline — single neuron per dim,
  rate-coded, i.e. **exactly our L2 v1** — *failed to match* the deep actor even at T=25
  (5× more timesteps), due to limited single-neuron representation capacity. Learnable
  population encoders measurably increased separation between observation encodings.
- **Fully Spiking NN for Legged Robots** (Wang/Wu et al. 2023, arXiv:2310.05022): applies
  PopSAN to A1 quadruped / Cassie / MIT Humanoid in Isaac Gym (RMA+AMP), current-based LIF,
  surrogate-grad; SNN ≈ ANN (humanoid sometimes better) across pyramids/stairs/stones. The
  closest prior work to the L-track.
- **MDC-SAN** (AAAI 2022): population coding + 2nd-order dynamic neurons; beats the deep actor.
- **ILC-SAN** (Chen et al. 2024): first *fully* spiking (membrane-voltage action decode +
  intralayer output connections) to match mainstream deep RL.

**Decision:** rebuild the actor as **PopSAN-style population coding** (L2b) — learnable Gaussian
input populations + population output decoder + current-based LIF, T=5, deep-MLP critic. Keep
the L2 rate-readout net as the "why population coding was needed" ablation (mirrors our nav-layer
strong-and-fair discipline, D7). MDC-SAN / ILC-SAN logged as SOTA upgrades (L6) if PopSAN
underperforms the MLP baseline (reward 36.25). Plasticity sites for the later R-STDP gait-recovery
experiment (L4) = the population/hidden weight matrices.

**Why not jump straight to ILC-SAN (the most SOTA):** PopSAN has public code, an explicit PPO
integration, and *legged-robot validation*; it is the lowest-risk path to a working spiking
locomotion policy on our exact stack. ILC-SAN's fully-spiking decode is a bigger, less-validated
build — defer to L6.

---

## D12. L3 spiking-policy tuning: reference-grounded fixes tried, none broke the ceiling (OPEN — documented negative result)

**Context:** L3's first full run (2048 envs, 1500 iters) trained clean but hit reward
**0.68** vs the MLP baseline's **36.25** (vel-err 1.38 vs 0.16 m/s). Diffed the actor
line-by-line against the actual PopSAN PPO reference source (Tang et al. 2020 code,
`popsan_drl/popsan_ppo/popsan.py`) to find principled, grounded fixes rather than guess.

**Fixes found and tried, each isolated in its own 400-iter run (2048 envs) for a fair
apples-to-apples comparison — the MLP baseline reaches reward 33.45 by iteration 400
(94% of its final 36.25), so 400 iters is enough to see if a fix is working:**

| config | reward @ iter 400 |
|---|---|
| v1: no fixes (obs_normalization=False) | 0.68 (@1500 iters — worse budget) |
| **v1.5: `obs_normalization=True` + T 5→8 only** | **2.9 — best result found** |
| v2: + encoder σ→0.387 (was ~0.667, spacing-derived) + actor-grad×0.1 | 2.1–2.3 |
| v3: + rectangular surrogate (replacing fast-sigmoid) + actor-grad×0.1 | 1.9 |
| v4: rectangular surrogate + actor-grad×1.0 (no LR cut) | 1.8–2.7 |
| v5: rectangular surrogate + actor-grad×1.0 + tanh-bounded decoder | 0.82 — worse |

**Each fix was independently well-grounded** (not guessed): σ and the actor/critic LR
gap are literal reference hyperparameters (`std=sqrt(0.15)`, `actor_lr=1e-5` vs
`critic_lr=1e-4`, a 10x gap emulated here via a backward-hook gradient scale since
rsl_rl/Isaac Lab PPO has one shared LR). The rectangular surrogate replaces a fast-
sigmoid(slope=25) shown to be ~36x weaker than PopSAN's window at a typical
|v-threshold|=0.2 — a real vanishing-gradient risk through a 3-layer x T=8 unrolled
graph. The tanh bound matches the reference decoder's `output_activation=nn.Tanh`
("Squashed Gaussian ... Spike Actor") exactly.

**Result: none of them helped; some made it slightly worse.** This rules out σ,
actor-LR, surrogate shape, and decoder bounding as *the* dominant bottleneck (each
alone or combined). It does NOT rule out: (a) a real implementation bug not yet found
despite a careful line-by-line diff, (b) a genuine mismatch between PopSAN's validated
scale (small classic-control Gym tasks: obs≤111 dims, act≤8 dims) and Go2 locomotion's
harder regime (obs=48, act=12, contact-rich dynamics, reward shaping tuned for the MLP),
or (c) a training-length gap larger than 400 iters can reveal — untested: the single
best config (v1.5) has never been run to the full 1500 to see if it's slower-but-still-
climbing rather than truly plateaued.

**Decision:** keep this a documented, honest open finding rather than a silent stall.
**Attempted the full-1500-iter diagnostic; hit a practical wall, not a scientific
answer.** The run (v6, the v4-equivalent config) measured ~59s/iter after iteration 75
— at that rate 1500 iters is ~24h, not the ~75min the earlier 400-iter runs suggested.
GPU was healthy (85% util, 54°C, no memory pressure, no competing processes) — the
likely explanation is that iteration-rate was never precisely measured across this
session's many overlapping background checks (rough elapsed-time sampling, not a timed
benchmark), so the 400-iter runs' true wall-clock cost may have been underestimated
throughout. Killed the run rather than block on an unbounded wait.
**Follow-up (overnight, full 1500 iters, properly timed): DEFINITIVE — structural, not
training-length.** rsl_rl's own `Iteration time` field (ground truth) showed steady-state
~2.9-3.1s/iter (the earlier "~59s/iter" alarm was this session's own imprecise elapsed-
time polling, not a real slowdown). Full trajectory: iter 400 reward 1.6, iter
600-1400 oscillating 1.2-3.6 with **no upward trend**, final (iter 1499) 3.03,
last-100-iter mean **2.48** (stdev 0.61). **The policy converges (stops improving) by
~iteration 400 and then sits flat for 1100 more iterations.** This rules out "just
needs more time" conclusively — 1100 extra iterations bought zero net progress.

**Hypothesis CONFIRMED — real breakthrough.** The Go2 cfg's PPO uses an **adaptive
KL-based LR schedule** (`schedule="adaptive"`, `desired_kl=0.01`); a population-coded,
quantized firing-rate policy output plausibly makes KL-divergence estimates noisier/
larger-looking than a smooth MLP's, so the schedule crushes the effective LR early and
never recovers (adaptive KL schedules ratchet down fast, back up slowly) — exactly the
observed "climbs, then flatlines forever" shape. **v8: same best config +
`algorithm.schedule="fixed"`, `learning_rate=1e-3` (constant, no adaptive shrink), 800
iters: reward climbed 4.8→8.7 and was STILL rising near the end — no flatline** (vs.
every adaptive-schedule run capping at ~2-3 by iter 400 and never moving again). **Best
result yet, ~3x the adaptive-schedule ceiling.** `SPIKING_FIXED_LR=1` env var toggles
this in `scripts/wsl_isaac_go2_spiking.sh` / `make_isaac_train_spiking.py`.

**Full 1500-iter run (v9, true apples-to-apples vs. the MLP's own training length):**
climbs steadily to iter ~600-800 (reward 7.8-8.1), then **settles into a stable plateau**
for the remaining 700+ iterations — last-100-iter mean **8.08** (max 9.31, min 6.51,
stdev 0.58), vel-err mean 1.46 m/s (still far from the MLP's 0.16). So: real, substantial,
reproducible improvement (2.5→8, ~3x) — genuinely fixed the earlier hard-flatline
pathology — but a new, higher plateau, not full convergence to MLP-level performance.

**Read:** the *shape* changed from "flatline, zero variance" (adaptive schedule) to
"oscillating plateau, real variance" (fixed schedule, stdev 0.58, swinging 6.5-9.3) —
consistent with instability/noise at this fixed LR (1e-3, unchanged from the MLP's own
tuned value) rather than a hard capacity ceiling. **v10 (gentler fixed LR=3e-4, full 1500 iters): WRONG DIRECTION.** Last-100-iter mean
**5.87** (stdev 0.74, still-rising trajectory through iter 1200 before settling) — worse
than v9's lr=1e-3 result (8.08, stdev 0.58, saturated by iter ~700), though its
velocity-tracking error was slightly better (1.05 vs 1.46 m/s — reward and vel-error
aren't perfectly aligned here, other reward terms likely diverge). **Conclusion: the
original lr=1e-3 (== the MLP's own tuned value) is the right magnitude; instability
wasn't the limiting factor.** Best known config stays: `schedule="fixed"`, `lr=1e-3`.

**v11 (best config extended to 3000 iters, 2x the MLP's own budget): DEFINITIVE — reward
~8 is a hard structural ceiling, not a training-length artifact.** Four measurement
windows spanning iter 600-3000 are statistically indistinguishable: iter 600-800 mean
7.92 (stdev 0.62), iter 1400-1600 mean 8.10 (stdev 0.56), iter 2200-2400 mean 8.09
(stdev 0.55), last-100 (iter 2900-3000) mean 7.68 (stdev 0.61). Vel-err unchanged at
1.447 m/s. **Doubling the training budget bought zero net progress a second time, now
at 2x scale — this rules out "needs more time" conclusively.**

**Overnight investigation, final summary (D12 closed as a completed empirical study):**
1. Root cause of the ORIGINAL flatline (reward ~2.5, D12 first entries): PPO's adaptive
   KL-based LR schedule crushing the spiking actor's effective learning rate early
   (confirmed — fixing the schedule alone gave the entire ~3x gain below).
2. Fix: `schedule="fixed"`, `learning_rate=1e-3` (== the MLP's own tuned value; a
   gentler 3e-4 is worse, v10) — **real, reproducible, now 2x-training-length-verified
   ceiling at reward ~8** (vs. the MLP's 36.25; vel-err 1.45 vs 0.16 m/s).
3. Remaining gap (8 vs 36) is a genuine architectural/capacity limit of this exact
   PopSAN configuration (pop sizes 10/10, hidden (256,256), T=8) on Go2's much harder
   regime (48 obs, 12 continuous joints, contact-rich dynamics) than PopSAN's validated
   scale (small classic-control Gym tasks, obs≤111, act≤8). Candidate next levers,
   untested tonight, need a fresh session to prioritize: larger `in_pop`/`out_pop` (e.g.
   20), larger hidden layers, more spiking timesteps T, or accepting this policy
   (reward 8, stable, not falling) as a "walks passably, not competitively" substrate —
   which may be SUFFICIENT for L4's actual question (does R-STDP gait-adaptation recover
   RELATIVE performance under a terrain shift), since L4 cares about relative recovery
   under shift, not matching the MLP's absolute walking quality.

**Architectural-lever sweep (v12-v14, 400-iter runs, same metric — last-50-iter mean
within the first 400 iters — for a fair apples-to-apples comparison against the
baseline's 6.06±0.61):**

| lever | config | mean reward | stdev |
|---|---|---|---|
| baseline | pop=10/10, hidden 128×3, T=8 | 6.06 | 0.61 |
| v12: larger populations | pop=20/20 | 6.05 | 0.62 — **no effect** |
| v13: bigger hidden layers | hidden=256×3 | **6.63** | **0.50** — modest gain, more stable |
| v14: more timesteps | T=16 | 5.95 | 0.69 — **no effect**, 2x compute cost |

Doubling population size or spiking timesteps bought nothing — rules those out as the
binding constraint. Bigger hidden layers gave a small, real-looking edge (both higher
mean and lower variance, i.e. more consistent, not just a lucky draw) — worth extending
to a longer run to see if a genuinely higher-capacity network compounds its advantage
over time, unlike the original config (proven flat even at 2x the MLP's training
length, v11). Testing hidden=256x3 extended to 1500 iters next.

**v15 (hidden=256x3, full 1500 iters) result: same ceiling, different path.** Unlike the
original config's hard early flatline (v9/v11: saturates by iter ~700, dead flat for
2000+ more iterations), this one climbs slowly and is STILL RISING at iter 1400 (peak
8.99) before a slight dip — last-100 mean **7.62**. Windowed means: iter300-400=6.40,
iter600-800=6.90, iter1100-1300=7.77 — a real, if slow, upward drift the whole run,
unlike the baseline's dead-flat plateau. Vel-err unchanged (1.404, matches every prior
run's ~1.4-1.46). **Read: bigger hidden layers change the LEARNING DYNAMICS (slower,
not-yet-saturated) but land in essentially the same ~7.6-8.0 final range at this budget
— not a clear win, though the still-rising trajectory at iter 1400 leaves open whether
more iterations than 1500 would separate it from the baseline's proven-flat ceiling.**

---

## D13. L4 first cut: R-STDP transferred from the nav layer does not recover locomotion under ice (OPEN)

**Setup:** the v9 spiking locomotion policy (checkpoint model_1499.pt) deployed inside
Isaac Lab's `Isaac-Velocity-Flat-Unitree-Go2-Play-v0` (clean, no obs-corruption noise,
no random pushes). Terrain friction combine_mode="multiply" (terrain x robot's fixed
0.8/0.6 foot friction) -- default terrain friction 1.0 gives effective ~0.8/0.6; icy
shift found by the same headroom-screening method as M4c: friction=0.05/0.15 floored
the policy (catastrophic near-instant fall, ~85% of episodes, no room to recover);
friction=0.4 gives a genuine, bimodal ~25% fall rate with real headroom -- when it
doesn't fall the episode looks like baseline, when it does it's within ~15-19 steps.

**R-STDP setup:** `PopSpikingRSTDPController` (input+readout plastic layers per D9,
TD-error third factor per D8, anchor=0.005, eta=0.05) -- the exact hyperparameters
that worked for the nav-layer M4 pilot, transferred directly, no locomotion-specific
tuning.

**Result (n=20 baseline, n=20 frozen-shift, n=30 rstdp-shift, n=20 retention):**

| phase | mean return | fall rate |
|---|---|---|
| baseline (normal friction) | 13.37 | 0/20 |
| frozen, icy (friction=0.4) | 12.62 | 5/20 (25%) |
| R-STDP under shift, 1st half | 1.22 | 7/15 (47%) |
| R-STDP under shift, 2nd half | -2.06 | 8/15 (53%) |
| retention (adapted weights, normal friction) | 10.32 | 0/20 |

**R-STDP does not recover -- it makes things worse under the shift, and trends further
down over the block rather than up.** But it does NOT catastrophically forget the base
task (retention ≈77% of baseline mean return, 0% fall rate) -- the anchor mechanism is
doing its stability job even though the adaptation itself isn't productive here.

**This mirrors the nav layer's own history, not a dead end:** M4's first R-STDP attempt
(readout-only) also stalled badly (7%) before input+readout+TD-error+anchor tuning
(D8/D9) reached 30% -- a multi-iteration process. L4 is at that same "iteration 1" stage,
not a proven failure. Most likely lever: **eta is probably too large for this
architecture** -- the locomotion actor's plastic layers are much larger (fc[0] is
480->128 = 61k weights vs the nav layer's smaller layers) and population coding is a
more delicately-tuned structure (L3's own tuning journey showed it's sensitive to
encoding scale, decoder bounds, surrogate shape under ordinary gradient descent) --
Hebbian-correlation updates at the same eta may simply perturb it faster than gradient
descent would. **⚠ L4 is ILL-POSED on the current base policy (2026-07-22, see D12 correction +
`archive/L4_gait_check/`):** the L3 spiking policy doesn't actually walk — it belly-flops
into a stationary crouch. You cannot study "R-STDP recovers a gait under a terrain shift"
when there is no gait to begin with, so the negative L4 results below (and the eta sweep)
are **uninformative** — R-STDP was being asked to recover something that was never there.
L4 needs a genuinely-walking spiking base policy first (a reward-shaping/exploration
problem on the L3 side, per the D12 correction). Keeping the pipeline + results recorded
since the pipeline itself is sound and reusable once a walking policy exists.

**✅ WALKING ACHIEVED (2026-07-22 autonomous session) via distillation, not reward-shaping.**
The L3-crouch/L4-standstill saga resolved: (1) anti-crouch reward shaping (D14 above)
fixed the belly-flop (base 0.11→0.24m) but the spiking policy then found a *stand-still*
local optimum — 2 reward variants (incl. 4x velocity reward + entropy) both refused to
walk under a forced forward command. Conclusion: the spiking policy won't DISCOVER walking
via PPO exploration regardless of reward. (2) Fell back to the M3 recipe — distill the
walking MLP teacher: verified MLP walks (forced vx=0.5 → 0.501, err 0.026), collected 128k
(obs,act) pairs, BC-trained the PopSAN actor (val-MSE 0.025). **The distilled spiking
policy WALKS: 5/5 episodes x 1000 steps, 0 falls, return 26-41 (matches the MLP), upright
stepping gait confirmed by MuJoCo-replay video.** (3) PPO fine-tuning from the BC init
*degraded* the gait (vel-err 0.036→0.29) and was abandoned — distillation alone wins.
Limitation: falls under a sustained-constant command (~5s, covariate shift; the MLP walks
at ~0.18m height too, so the distilled net's 0.19m is faithful, not a crouch). Full record
in `archive/L4_gait_check/`. Lesson reinforced: verify walking by trajectory+video, never
by reward alone.

**DAgger made it robust to sustained commands (2026-07-22):** the BC net fell ~5s into a held-constant command (covariate shift). Rolled the student under held forward commands, labeled its drift-states with the teacher (153.6k pairs), aggregated + retrained warm-started. Now under sustained vx=0.5: 3/3 x 1000 steps, 0 falls, vx err 0.028 (matches MLP), ~9.3m forward (`dagger_walk_forward.gif`). The M2 nav-BC->DAgger recipe transferred cleanly to spiking locomotion.

**Tested: smaller eta (0.005, a 10x cut) -- WRONG DIRECTION, rules out "eta too large."**
Fall rate got WORSE, not better: 19/30 (63%) vs eta=0.05's 15/30 (50%). Since both a 10x
larger and the original eta show similar-or-worse degradation, the step-size magnitude
is not the primary lever -- something more structural is at play. **Untested next
(higher-value now that eta is ruled out): fewer plastic layers** -- readout-only (D9's
nav-layer lesson: readout-only *stalls* without helping, doesn't actively *hurt* --
if that's also true here, it would isolate INPUT-layer plasticity specifically as the
destabilizing factor, since perturbing the population encoder's stimulation pathway
might break the delicately-tuned population code in ways gradient descent never would).
Also untested: a much stronger anchor (current 0.005 may be too weak to counteract
Hebbian drift at either eta tried), or gating (`gate_threshold`, suppress updates when
the TD-error is small, currently disabled/0.0).

---

**⚠ MAJOR CORRECTION (2026-07-22, `archive/L4_gait_check/`): reward ~8 is a NON-WALKING
crouch local optimum, NOT a capacity ceiling.** We finally dumped a real rollout
trajectory and looked at it (figure + MuJoCo replay GIF) instead of only tracking reward.
The policy **collapses its base from 0.40 m to ~0.11 m in ~0.5 s and belly-flops** —
legs splayed, body on the floor — then stays there, travelling 0.49 m in 20 s with mean
|vx| ≈ 0.02 m/s and never tracking the velocity command. Reward ~8-13 comes from not
triggering base-contact termination while collecting alive/orientation reward. So the
whole D12 "hard architectural ceiling / capacity" framing below is **wrong**: this is an
**optimization/exploration failure** (the spiking net gets stuck in a degenerate
optimum), not a representational-capacity limit — the MLP walks under the identical reward
(36, vel-err 0.16). The ~1.4 m/s velocity-tracking error that reappeared across *every*
D12 variant was the tell we didn't heed: none of them ever walked. The architectural-lever
sweep (populations/hidden/T all landing at ~8) is consistent — they all fall into the same
crouch optimum. Next levers are reward-shaping / exploration (penalize low height, NoisySAN,
curriculum), NOT bigger networks. Lesson (again, cf. the D3 obstacles-invisible render bug):
**watch the robot, don't just trust the scalar** — a reward number hid a non-walking policy
for the entire L3 investigation.

**Consolidated read across the whole overnight+continuation investigation:** reward ≈8
(vel-err ≈1.4 m/s) is a robust number that reappears across the default architecture (v9,
2x-budget-verified flat), larger populations (v12), more timesteps (v14), and bigger
hidden layers (v15, different path, same destination). This consistency across 4
architectural variants is itself meaningful evidence that ~8 reflects something more
fundamental about this PopSAN-on-Go2 setup than any single hyperparameter — likely the
population/spiking encoding's fit to a 48-dim continuous-control observation space,
rather than raw network capacity. Next real lever (untested, bigger undertaking, not a
simple param sweep): a structurally different decode (ILC-SAN's fully-spiking
membrane-voltage output + intralayer connections, or MDC-SAN's 2nd-order dynamic
neurons — both already flagged as L6 options) — or accept ~8 and proceed to L4, since
R-STDP gait-recovery only needs a policy that walks *some* way, not competitively.

---

## D15. L5 energy (H2): the distilled spiking locomotion policy needs sparsity + minimal T to beat the MLP — and only marginally (RESOLVED — walking + energy-positive at T=5)

**Question:** does the spiking *locomotion* controller (the L4 distilled PopSAN walker)
actually cost less energy than the MLP baseline? The nav-layer ALIF-SNN fired at ~1.4% and
was dramatically cheaper; H2 assumed locomotion inherits this. Method: SynOps vs MACs on
real walking obs, 45 nm (Horowitz 2014), `scripts/l5_energy_analysis.py`. Every policy is
re-verified to still walk by trajectory (path length + vx tracking + base height), never
by reward.

**Finding — the naive distilled walker is MORE expensive than the MLP:**
| policy | firing | T | val-MSE | energy | vs MLP | walks? |
|---|---|---|---|---|---|---|
| dense (DAgger walker) | 62.7% | 8 | 0.025 | 328.9 nJ | **0.57× (costlier)** | ✅ |
| + firing penalty λ=0.05 | 44.2% | 8 | 0.029 | 266.9 nJ | **0.70× (costlier)** | ✅ |
| + penalty, **T=5** λ=0.03 | 47.7% | 5 | 0.038 | 178.1 nJ | **1.04× cheaper** | ✅ path 18.3 m |
| + penalty, T=4 | 44.7% | 4 | 0.049 | 136.8 nJ | 1.36× cheaper | ❌ belly-flop |
| MLP baseline | — | — | — | 186.1 nJ | 1.0× | ✅ |

Two reasons it starts costlier: (1) **behavior cloning produces dense firing** (62.7%, vs
nav's 1.4%) — BC matches actions, nothing rewards sparsity; (2) **population coding + T**:
the encoder inflates obs 48→480 so `fc[0]` is 480×128, and every SynOp is paid T times.

**Resolution — two levers, in order:** (a) added a differentiable mean-spike-activity term
to the distillation loss (`PopSpikingActorNet.forward_with_activity` + `--firing-penalty`)
→ halves firing (62.7%→44%) with negligible gait loss, but only reaches 0.70× (T=8 overhead
dominates). (b) **T is the dominant energy knob but trades against gait fidelity** — T=4 is
1.36× cheaper but breaks walking (val-MSE 0.049 → belly-flop). **T=5 is the minimum T that
preserves the gait → 1.04× cheaper while still walking.** So H2 holds for locomotion, but
only after sparsity regularization AND T minimization, and only by a thin margin.

**Interpretation (the honest, thesis-grade point):** locomotion does NOT get the nav
layer's order-of-magnitude energy win. Continuous precise motor control resists the
sparse-firing regime (needs denser firing + several timesteps for action fidelity); coarse
discrete navigation is naturally sparse. **This is the same message as D13** (R-STDP
recovers navigation but destabilizes locomotion): the locomotion layer is a harder, more
delicate substrate for neuromorphic methods than navigation — a nuanced, layer-specific
contribution rather than a blanket "SNNs win everywhere." Caveat: 45 nm digital SynOps is a
conservative proxy; on neuromorphic hardware (Loihi, PopSAN's ~140× vs GPU) the advantage
is far larger — 1.04× is a digital-ASIC floor, not the neuromorphic ceiling. Full record +
videos: `archive/L5_energy/`.

**⚠ CORRECTION (2026-07-23, user feedback: "it walks too crouched i dont like it"):**
the walker above was faithful to its teacher but the *teacher itself* only walked at
0.184 m (Go2's natural standing height ≈0.30 m) — the flat-velocity reward never demanded
a taller stance, and distillation can't exceed what it's shown. **Fixed at the source:**
retrained the MLP teacher with a base-height reward (target 0.30 m; the MLP has no PPO
discovery problem, so it simply learns taller — unlike the spiking net's D14 stand-still
trap). New teacher verified by trajectory: 0.279–0.305 m, vx tracks 0.5 (err ≤0.07), 0
falls. Re-ran collect→distill→DAgger→firing-penalty→T-sweep on top of it — **every energy
number is unchanged**: T=5 is still the minimum T that preserves the gait, still **1.04×
cheaper than the MLP (178.5 nJ)**, now at the correct standing height (0.302 m, vx 0.507,
3/3×1000 steps no falls, verified both by trajectory and by eye in the rendered video). One
new wrinkle: the first T=5-upright retrain was fragile (2/3 held-command episodes eventually
fell despite walking cleanly for hundreds of steps first) even though its offline val-MSE
(0.050) looked similar to the working T=4 crouched failure — a gentler retrain (lower
firing-penalty, more epochs) fixed it to 3/3 robust despite nearly-identical val-MSE,
reinforcing **offline loss does not predict closed-loop robustness**. Details:
`archive/L5_energy/README.md` "UPDATE (2026-07-23)".

**⚠ SECOND CORRECTION (2026-07-23, user feedback watching the video: the rear-right leg
looked like it was dragging):** quantified with a new per-leg amplitude/jitter check
(`scripts/l4_leg_amplitude.py`) — confirmed the MLP teacher itself had the asymmetry (RR
hip swing 0.123 vs 0.25–0.32 for the other three legs), same root-cause pattern as the
crouch. Fixed via Isaac Lab's built-in `feet_slide` reward term (penalizes foot velocity
during ground contact). First attempt at weight −1.0 was a clean failure mode worth
recording: the policy froze rigidly to avoid the penalty entirely (all joint amplitudes
collapsed to ~0.01, vx→0) — corrected to −0.1 (the value Isaac Lab's own humanoid configs
use). Re-ran the full pipeline; **every energy number holds**: T=5 still ~1.03× cheaper
(181.0 nJ) at the now taller AND less-asymmetric gait (0.338 m, vx 0.523, 3/3×1000 steps
0 falls, path 20.5 m — the best walking metrics of any checkpoint in this project). RR's
whole-leg drag is resolved (hip went from smallest to largest amplitude of the four legs);
its thigh joint alone retains a partial residual asymmetry. Details:
`archive/L4_gait_check/README.md` "DRAGGING REAR LEG FIX", `archive/L5_energy/README.md`
"UPDATE 2".

## D16. M5 full comparison: M4's R-STDP recovery signal does not replicate at n=10 seeds (honest negative, sensor-dropout shift)

**Question:** M4's pilot (single model-seed=7, one env-seed block of 30 episodes)
reported R-STDP recovering to 30% success under sensor dropout, beating frozen
SNN's 17% — flagged explicitly as unconfirmed ("CIs overlap at n=30... a
positive *signal*, not a proven effect"). M5's job: run it properly across
independent seeds and find out if the signal is real.

**Method:** rebuilt `scripts/m5_full_comparison.py` (originally an autonomous-
session draft with two real bugs — see `archive/M5_full_comparison/README.md`
for the full account) to run all 6 controllers × 10 seeds × 30 episodes,
Holm-corrected pairwise significance vs R-STDP. Two bugs fixed first: (1) the
shift config left `sensor_dropout_start` unset (random dead-beam location
every episode instead of M4's fixed, learnable corruption — silently defeats
R-STDP's whole premise); (2) the TM-NORM baseline (D10) had an unbounded
threshold-drift feedback loop (v_th going negative flips the ALIF reset's sign,
causing runaway divergence) giving an implausible 0.0%±0.0% across all seeds.

**Finding — R-STDP does not beat frozen SNN, checked six independent ways:**
after fixing both bugs, R-STDP (8.7%±3.7%) numerically trails frozen SNN
(11.0%±2.6%, p=0.14) at dropout=0.30. Investigated every plausible confound
before accepting this: (a) a coarse eta/anchor sweep found an apparent winner
(16.0%) that **regressed to noise (10.3%, p=0.63) when validated at full
rigor** — a reminder that small-n screens produce false positives; (b) a
severity screen found dropout=0.30 (M4's own single-seed pick) is **already
floored** (75–85% collision, near-zero success from 0.25 up) — the real
headroom zone is 0.15–0.20, mirroring M4c's own ice-severity lesson; (c) re-run
at the corrected dropout=0.20 (real headroom, base 42.7%→~20%): **R-STDP
(19.3%) still ties frozen SNN (21.3%), p=0.70**; (d) a plasticity-scope ×
third-factor ablation (all-layers vs input+readout, TD-critic vs RPE) at the
corrected severity: **every variant (19.3–21.7%) ties frozen SNN**. Diagnostics
(per-seed + termination-reason breakdown): no seed outliers (R-STDP's 10 seeds
sit in a tight band each time); falls are rare everywhere (1–6%, the
locomotion policy never breaks down mid-episode); the dominant failure is
**collision**, elevated for the whole SNN family (79–94%) vs the MLPs (~60%)
regardless of plasticity setting.

**Conclusion:** this is a genuine, thoroughly-triangulated negative result, not
a bug or a mistuned hyperparameter. M4's 30% was very likely a single-seed
outlier — precisely the risk M4's own writeup flagged. **The part of H1 that
holds up in every single test: R-STDP reliably, significantly beats pure
Hebbian STDP** (e.g. 19.3% vs 8.7% at dropout=0.20) — reward modulation is
doing real work, just not enough to beat a frozen network on this particular
(sensor-level) shift. Contrast with M4c (terrain/sand, a dynamics-level shift):
R-STDP closes the *full* gap to the MLP there. **H1 is shift-dependent, not
universally true or false** — a nuanced, defensible thesis finding, consistent
with the L-track's own nav-vs-locomotion contrast (D13/D15): reward-modulated
plasticity helps where the fix is a coarse re-calibration the controller's
own interface can express (velocity commands under a terrain fault, or here,
apparently not a sensor-level remap through this specific SNN's population-
coded action interface). Full record, all six checks' data, and reproduce
commands: `archive/M5_full_comparison/README.md`.

## D17. M6 energy + M4b extensions: H2 confirmed on navigation; the robustness belongs to ALIF, not R-STDP; and online plasticity is the thing the FPGA must fix

**Question:** (a) does the nav-layer SNN actually save energy (H2), and does it degrade
gracefully under noise (H4)? (b) M5 left two escape hatches — maybe R-STDP helps at some
untested *severity*, or maybe the result is specific to the one dead-beam *mask* M4/M5
always used. (c) The M4b plan's item 7 was a pre-registered ablation with a stated
decision rule that had never been run.

**Method:** ported L5's SynOps/Horowitz-45nm model to the nav layer with firing rates
measured on real closed-loop rollouts, and extended it three ways the literature model
omits (`src/nmc/eval/energy.py`): neuron-state updates, the spike encoder, and **the cost
of the online learning rule itself** — the last reported both as the dense reference
implementation runs and as an **event-driven lower bound**. Plus four M4b screens
(`scripts/m4b_extras.py`). Full records: `archive/M6_energy/`, `archive/M4b_extras/`.

**Finding 1 — H2 holds on navigation, and it is a real contrast with locomotion.**
Frozen SNN is **10.4× cheaper** than the frozen MLP on the literature-standard SynOps
view (125.5 vs 1304.8 nJ/decision) and **2.4×** on full accounting (549.7 vs 1321.7 nJ).
L5 got only 1.04× for locomotion after sparsity regularization and minimal T. Same
method, same 45 nm constants, order-of-magnitude different answer — the cleanest
quantitative statement yet of the layer-specific thesis message (cf. D13/D15).

**Finding 2 — at 1.5% firing the neurons cost 3.4× the synapses** (420 vs 126 nJ of the
549.7 nJ total). The per-tick membrane/adaptive-threshold arithmetic is paid by every
neuron whether or not it spikes, so beyond a certain sparsity the SynOps metric stops
describing where the energy goes. Actionable: the remaining lever is T (or event-driven
neuron updates), not further sparsification — which is also what L5 concluded about T
from the opposite direction.

**Finding 3 (the honest one) — per-decision efficiency does not survive conversion to
per-task efficiency.** Energy per *successful navigation*: frozen SNN 2276 µJ vs frozen
MLP 2280 µJ — a dead heat, the 2.4× per-decision advantage being exactly cancelled by
the SNN's lower success rate and longer paths. Efficiency quoted per inference is not a
deployment claim. This metric should be reported alongside per-decision energy anywhere
the thesis makes an efficiency argument.

**Finding 4 — online plasticity is expensive, and that is precisely the FPGA's job.**
R-STDP's dense eligibility update costs **15 538 nJ/decision** — 28× its own inference,
and 12× the MLP's entire forward pass. The **event-driven/crossbar formulation is 20.8×
cheaper (747 nJ)**, which brings an *online-learning* controller to parity with a frozen
MLP forward pass, and is **10.2× cheaper than backprop learning** (online-MLP: 7587 nJ).
M7/M8 now has a quantified target instead of a hand-wave, and H3 gains a second, sharper
framing: the crossbar's win is on the *update*, not the forward pass.

**Finding 5 — the pre-registered neuron ablation reassigns the credit.** The M4b-7
decision rule was explicit: a large frozen-LIF/frozen-ALIF gap "would mean ALIF itself is
doing some of the work." Result (10 seeds × 10 eps, dropout 0.20): **frozen LIF 7.0%,
frozen ALIF 17.0% (+10 pts, p=0.051), R-STDP-ALIF 12.0% (−5 pts vs frozen ALIF,
p=0.302).** So the shift-robustness M4 read as R-STDP recovery is **a property of the
ALIF neuron model** — adopted back in D2 as an unrelated SOTA upgrade and never
separately credited — while the plasticity adds nothing on top. This also **disagrees
with Zhao et al. 2025 Table VII**, which reports adaptive-threshold neurons alone *not*
rescuing OOD performance and which is what motivated running the ablation; the
disagreement should be stated in the write-up, not smoothed over.

**Finding 6 — M5's two escape hatches are closed, with one softening.** No severity in
0.10–0.30 shows a significant R-STDP advantage over frozen SNN (deltas +6/−6/−8/−0 pts,
all p>0.38). No dead-beam mask reaches significance either; but **pooled over four masks
R-STDP is 24.0% vs frozen 22.0%**, and M4/M5's start=8 mask turns out to be one of the
less favourable ones — so the honest estimate of the true effect is **~0, not negative**.
R-STDP also shows larger across-mask variance (15.9% vs 9.2%): plasticity adds spread.

**Finding 7 — H4 is REFUTED: the MLP degrades more gracefully than the SNN.** Within
σ ≤ 0.2 nothing separates (that range is simply too mild). Extending to σ=0.8 makes the
ordering clear and it is the *opposite* of the hypothesis: the frozen **MLP is essentially
noise-immune** (42% → 38%, slope −1.3 pts/0.1σ) while the frozen **SNN loses more than
half** (≈45% → 18%, −1.8); online-MLP collapses only at the extreme (N50 = 0.685) and
TM-NORM dies outright (0%). Mechanism, and it is coherent rather than a fluke: the SNN
does not read the LiDAR, it **Poisson-samples** it over T=20 ticks, so additive input
noise *compounds with* encoding noise instead of being averaged away — the MLP consumes
the analog value directly and never pays that. The standard neuromorphic
"temporal filtering averages noise out" claim does not hold at T=20. Levers if the thesis
wants to recover H4: raise T (energy cost, per Finding 2) or drop stochastic rate coding
for a latency/learned encoder — which **D4 already flags as the SOTA alternative to
revisit**, and this is the concrete evidence that it should be.

**How to apply:** the honest nav-layer scorecard the write-up should present is
**mixed and specific, not a blanket verdict**: energy — WIN (10.4x, and a real contrast
with locomotion's 1.04x); shift-robustness from ALIF neuron dynamics — WIN (+10 pts,
Finding 5); noise robustness — LOSS (Finding 7); online-plasticity recovery — NULL
(M5/D16 and Finding 6). Lead the energy argument with the nav-vs-loco contrast, always
pair per-decision energy with energy-per-success (Finding 3), and use the
dense-vs-event-driven plasticity gap (Finding 4) as the quantified motivation for the
FPGA chapter. Credit ALIF, not R-STDP, for shift-robustness. Two concrete follow-ups this
opens: **D4 (drop stochastic rate coding for a latency/learned encoder)** now has direct
evidence behind it, and **D1 (e-prop)** remains the scoped upgrade path for the plasticity
null. See D16 above for the companion M5 entry.

## D18. M4c's terrain result does not replicate either — R-STDP now has zero surviving significant wins

**Question:** M5 (D16) found M4's sensor-dropout "R-STDP recovery" win (30% vs
17%) did not replicate at real multi-seed rigor. M4c's terrain result (R-STDP
"closes the gap" on sand, "beats both" on ice) was the thesis's one remaining
positive H1 evidence -- but it rested on the exact same kind of thin sample
(n=15, one seed) that the dropout claim had before M5 broke it. Overdue check:
does the terrain result survive the same treatment?

**Method:** `scripts/m4c_rigorous.py` reproduces `render_terrain_videos.py`'s
exact protocol (frozen MLP/SNN: no adaptation; R-STDP: 15-episode silent
warm-up then measure) across 10 seeds x 30 episodes instead of M4c's original
sample, with Holm-corrected Welch t-tests against R-STDP, matching M5's
standard exactly.

**Finding -- it does not replicate:**

| Terrain | Frozen MLP | Frozen SNN | R-STDP SNN | R-STDP vs Frozen SNN |
|---|---|---|---|---|
| Sand (mu=1.6) | 53.3%+/-10.9% | 42.3%+/-11.7% | 46.7%+/-10.5% | +4.3 pts, p=0.421 |
| Ice (mu=0.28) | 44.7%+/-10.7% | 29.0%+/-10.8% | 34.0%+/-13.3% | +5.0 pts, p=0.393 |

Both of M4c's headline claims fail: R-STDP is not "closing the gap" to the MLP
on sand (it is, if anything, 6.7 pts *below* the MLP, also not significant),
and it does not "beat both" on ice (its lead over frozen SNN is real numerically
but nowhere near significant, p=0.39). Frozen SNN's own number moved
substantially between the two runs too (27%->42.3% sand, 27%->29.0% ice) --
confirming the n=15 sample was noisy for the frozen baseline as much as for
R-STDP, i.e. this is the same statistical-illusion mechanism as D16, not a
new or different failure mode.

**Correction, resolved the same night:** the sand condition above was first tested at mu=1.6, which turned out to be the value M4c's OWN earlier screening had REJECTED as unusable ("sand *helps* it to 55%") -- the actual headline claim ("closes the gap to the MLP") was measured at the RETUNED mu=1.20. Re-run at the correct mu=1.20 (10 seeds x 30 eps, Holm-corrected): **Frozen MLP 55.0%+/-10.2%, Frozen SNN 42.7%+/-8.7%, R-STDP SNN 40.3%+/-11.1%.** R-STDP vs Frozen SNN: -2.3 pts, p=0.626 (not significant, and numerically WORSE). R-STDP vs Frozen MLP: -14.7 pts, **p=0.019 (SIGNIFICANTLY worse)** -- the exact opposite of the original "closes the gap to the MLP" claim. This is now the strongest single piece of evidence in the whole project that a small-sample R-STDP "win" was pure noise: not just "not significant", but significantly reversed once measured at the condition the claim actually rested on. The ice result (mu=0.28) was correct as originally tested and stands as reported above.

**Also discovered in the process (visual audit, before this rigor check ran):**
the site's `snn_rstdp_sand.gif`, captioned as a recovery demo, was actually a
recorded failure -- `render_terrain_videos.py` records episode 0 unconditionally
with no success check (unlike `eval_mlp_go2.py`'s own `eval_frozen`, which only
keeps a GIF `if reached`). Fixed by finding a verified success; see
`archive/M4b_terrain_walk_compare/README.md`. 4 of 5 attempts along the way
were dynamic-obstacle collisions, and a separate audit
(`scripts/m5_collision_kind_audit.py`) confirmed **70-73% of ALL collisions
across every controller, on the standard procedural map, are with a moving
obstacle** -- a confound unrelated to whichever shift is nominally being
studied. This directly motivated `scripts/custom_map.py`, a hand-designed
deterministic map for cleaner future tests.

**Thesis-level consequence:** at full statistical rigor, R-STDP currently has
**no surviving significant recovery advantage on any shift type tested** --
sensor dropout at two severities (D16/M5), terrain sand, terrain ice (this
entry). This is a materially different position than the roadmap held as of
D16/D17: H1 is not merely "shift-dependent" (real on terrain, absent on
sensor-level faults) -- so far it has not been demonstrated anywhere at
rigor. This does not mean H1 is false; it means the two pieces of evidence
that seemed to support it were both single-seed artifacts, caught by doing
exactly what M5/M4c's own writeups said should be done next. Two open paths
remain, both in progress the same night this was found: (a) the three
previously-untested shift types (`sensor_bias`, `sensor_range`, `goal_drift`
-- see `archive/shift_taxonomy/`), since the shift-dependence hypothesis may
still hold for a shift class not yet tried; (b) e-prop (D1), a structurally
different three-factor rule, tested under identical conditions
(`archive/D1_eprop/`) to separate "the rule" from "the problem" as the cause
of R-STDP's null results so far.

---

## D19. e-prop (D1) tested where R-STDP failed — the null result generalizes beyond R-STDP

**Question (from D1/D18):** is R-STDP's lack of any surviving significant recovery
advantage (sensor dropout, terrain sand, terrain ice, custom map) about **the rule**
(a purely Hebbian trace + one global reward scalar) or **the problem** (this shift,
this network, this closed-loop-RL interface)? e-prop (Bellec et al. 2020) is a
structurally different three-factor rule — its eligibility trace comes from the
neuron's own membrane/adaptation dynamics (an online BPTT approximation, using the
*same* FastSigmoid surrogate already used for M3's offline pretraining) and its third
factor is a per-neuron symmetric-feedback signal, not one broadcast scalar. Implemented
from scratch in `src/nmc/plasticity/eprop.py` / `src/nmc/controllers/eprop_snn.py`,
independently NumPy-parity-checked against the real PyTorch network (20/20 actions
matched, max diff 0.0 at eta=0).

**Method:** an eta guardrail sweep (base distribution, no shift) found eta=0.005 is the
largest learning rate that stays within 10 pts of the frozen baseline with finite
weights (`archive/D1_eprop/guardrail.md`). At that eta, e-prop was then tested under
the exact sensor-dropout protocol R-STDP was tested under in M5 (10 seeds x 30 eps,
both severities) — with Frozen SNN and R-STDP SNN **re-run fresh in the same pass**
(not compared against old summary numbers) so the comparison is a real paired Welch
t-test on matched raw per-seed data, Holm-Bonferroni corrected across all 4
comparisons (`scripts/eprop_rigor2.py`, `archive/D1_eprop/rigor2.md`,
`raw_success_rates.csv`). A first pass (`eprop_rigor.py`) had skipped both of those
corrections and produced an enticing but unreliable raw p=0.019 — flagged and
corrected before being trusted, the same discipline D16/D18 already established.

**Result:**

| dropout | e-prop vs | delta | Holm-corrected p | significant? |
|---|---|---|---|---|
| 0.30 | Frozen SNN | +0.3 pts | 0.902 | no |
| 0.30 | R-STDP SNN | +2.7 pts | 0.711 | no |
| 0.20 | Frozen SNN | +7.3 pts | 0.231 | no |
| 0.20 | R-STDP SNN | +9.3 pts | 0.078 | no |

**Decision: e-prop does not show a Holm-corrected significant advantage over frozen
SNN either.** The null result generalizes beyond R-STDP specifically. Combined with
D17's neuron-model ablation (frozen ALIF alone gains +10 pts that no plasticity rule
tested has improved on), the most defensible reading is that **the neuron model, not
online synaptic plasticity, is what's doing the shift-robustness work** in this
setup — a genuine, honestly-negative result about online RL-adapted local learning
rules at this network scale, obtained by testing a second, structurally different
rule rather than concluding from R-STDP alone. Full write-up: `archive/D1_eprop/README.md`.

---

_Update this log whenever a new SOTA option is identified. Every "we chose the simpler
thing" must have an entry saying why and when to revisit._
