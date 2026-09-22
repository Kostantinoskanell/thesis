# Handoff brief — Neuromorphic Control for Robotics (ECE Patras thesis)

Written 2026-09-04 to hand this project off to another coding agent (or a fresh
session of this one). **Source of truth is the repo itself** — this file is a
compressed map of it, not a replacement for reading `ROADMAP.md`.

## 1. What this thesis is

Undergraduate ECE thesis: a hardware/software co-design comparing a spiking neural
network (SNN) with **reward-modulated STDP (R-STDP)** against DNN/RL baselines, on a
Unitree Go2 quadruped, under a **mid-episode distribution shift** (sensor dropout /
terrain change). An FPGA co-processor structurally analogous to a memristive crossbar
implements the synaptic weight update. See `docs/references/README.md` (proposal) and
`ROADMAP.md` (full plan + locked decisions + status table).

## 2. Locked scoping decisions (do not relitigate without the user)

- **R-STDP is the primary plasticity rule** the headline hypothesis (H1) rests on.
  Pure Hebbian STDP is kept only as an ablation.
- **Full FPGA co-design is in scope**, but software science is front-loaded — no HDL
  work starts until the R-STDP result exists in simulation (already true — see below).
- **Robot/sim: Unitree Go2, MuJoCo, full rigid-body dynamics everywhere** (not a
  kinematic toy model). Two-layer architecture: an SNN **navigator** outputs velocity
  commands `[vx, vy, omega]`; a separately-trained **RL locomotion policy** (MuJoCo
  Playground / Isaac Lab rsl_rl PPO) walks the robot tracking those commands.
- **H3 (FPGA) is framed as "find the crossover network size vs CPU/GPU,"** not "beat
  the GPU at all sizes" — losing at small sizes is a valid, expected finding.

## 3. Where things stand (as of last commit, 2026-07-24 — check `git log` for anything newer)

Two tracks:

**M-track (navigation layer, PyBullet/MuJoCo, `nmc` conda env, Windows or WSL CPU):**
- M0–M4c all **done**. Headline: M4 (go/no-go pilot) got a **GO signal** — R-STDP
  recovers a policy corrupted by sensor dropout 7%→30% success while retaining 43% of
  base-task performance (best of any controller); online-MLP recovers early then
  catastrophically forgets. M4c extended this to terrain (ice/sand) with the same
  conclusion once shift severity was properly tuned.
- **Not statistically rigorous yet** — these are n=15–30 pilots with overlapping CIs.
- **➡ NEXT CONCRETE STEP: M5** — full comparison of all 6 controllers × all metrics,
  **≥10 seeds with 95% CIs**, Holm/Bonferroni-corrected pairwise significance tests.
  This is what turns "GO signal" into a defensible thesis result. **Unstarted.**
- After M5: **M6** — energy sweep (SynOps/µJ per decision, method already proven in
  L5 below, just needs porting to the nav-layer controllers) + noise-degradation curves.

**L-track (locomotion layer, Isaac Lab rsl_rl PPO, GPU-required, WSL2 env `nmc-snn`):**
- **Declared complete** (L0–L5 done, L6 deferred as optional/future work). Opened
  because M4c showed nav-layer R-STDP can't fix *locomotion* faults (e.g. ice slipping)
  since that's below the navigator's interface — this track pushes plasticity down
  into the gait itself.
- Built a **PopSAN-style population-coded spiking actor** (Tang 2020) trained via PPO
  in Isaac Lab; hit a real architectural-looking ceiling (reward~8) that turned out to
  be a **belly-flop local optimum**, not a capacity limit (caught by rendering video,
  not trusting the reward scalar — reinforced our verification protocol, see §5).
  Fixed via **distillation from a walking MLP teacher + DAgger** → first working
  spiking Go2 locomotion policy found in the lit search, robust to sustained commands.
  Then fixed two more issues purely by *watching the video* the user flagged: too
  crouched (retrained teacher with base-height reward) and a dragging rear leg
  (`feet_slide` reward term, weight -0.1 — Isaac Lab's own humanoid-config value after
  -1.0 caused the policy to freeze rigid).
- **R-STDP result on locomotion: negative** — it *destabilizes* the gait under an icy
  shift rather than recovering it (fall rate rises 25%→53%), while retaining ~77% of
  base performance (anchoring works, adaptation doesn't). This contrasts cleanly with
  the nav-layer's positive R-STDP result.
- **Energy (H2) result:** naive distilled walker fires densely (62.7%) and is actually
  *costlier* than the MLP (0.57×). Fixed with a firing-rate penalty during distillation
  + **T=5 timesteps** (the minimum that preserves the gait) → **1.04× cheaper than the
  MLP, and still walks** — a real but thin margin (nav layer's SNN win was much larger,
  1.4% firing vs 62%+). Honest 45nm-digital floor; Loihi/neuromorphic hardware would do
  far better (PopSAN cites ~140× vs GPU).
- **Overall L-track thesis message:** neuromorphic online-learning + sparse-coding
  efficiency fit **coarse discrete navigation** naturally, but are a **marginal,
  delicate fit for continuous fine motor control**. Motivates e-prop as scoped future
  work for the locomotion layer (not attempted — deferred, see D1 in
  `docs/references/sota_decisions.md`).

**FPGA track (M7–M8): not started.** Waits on M5/M6 per the software-first plan.
The R-STDP golden reference (M0) and the SynOps/energy method (L5, portable to M6)
already exist as the two prerequisites it needs.

## 4. Repo map — read these, in this order, before touching anything

1. `ROADMAP.md` — the actual plan, locked decisions, milestone table with full status
   prose per milestone (much more detail than this brief's summary above).
2. `docs/references/sota_decisions.md` — dated decision log (`D1`, `D2`, ... `D15`):
   every SOTA-vs-conventional choice, why, and what would make us revisit it. **Read
   this before picking any method** — many "obvious" choices were already tried and
   rejected here (e.g. plain firing-rate spiking actor superseded by PopSAN population
   coding, D11).
3. `docs/references/` — the rest: `neuromorphic_robotics.md`, `snn_learning.md`,
   `locomotion.md`, `fpga_acceleration.md`, `memristors_hardware.md`,
   `compute_environments.md`, `study_plan.md` — curated, topic-grouped citations.
4. `docs/debug-log/` — dated war-stories (symptom → cause → fix → lesson). Check
   before re-debugging something (e.g. Isaac Lab WSL 8GB bring-up, quadruped gait
   collapse, geomgroup-4 invisible obstacles).
5. `archive/<milestone>/README.md` — for any milestone you touch, its archive folder
   has the full write-up, figures, and GIFs of that milestone's journey.
6. `src/nmc/` layout: `envs/` (PyBullet + MuJoCo Go2 nav envs), `controllers/` (MLP,
   SNN), `plasticity/stdp.py` (STDP/R-STDP golden reference), `encoding/` (spike
   encoders), `eval/metrics.py` (SynOps, recovery time, CIs), `locomotion/` (PopSAN
   actor, R-STDP controller for locomotion), `rl/envs/go2/` (Isaac Lab / Playground Go2
   joystick env), `platform/go2_rl_walker.py` (the Windows-side locomotion driver the
   SNN navigator commands).

## 5. Working conventions the user expects (from prior sessions)

- **Always pick the state-of-the-art method over the conventional/easy one**, even if
  harder to implement. If SOTA conflicts with a locked decision, surface the trade-off
  explicitly rather than silently picking one (e.g. e-prop is more SOTA than R-STDP for
  online RSNN learning, but R-STDP maps better to the memristor-crossbar hardware
  story — this exact tension is already recorded, don't re-decide it alone).
- **Verify by video/trajectory, never by reward scalar alone.** The L-track's biggest
  bug (a belly-flop policy with a plausible-looking reward) was only caught by
  rendering an episode. For every simulation milestone: render at least one episode
  GIF + a diagnostic plot, saved to `archive/<milestone>/`, in addition to numeric
  checks.
- **Offline loss does not predict closed-loop robustness** — always validate with a
  real rollout (multiple episodes, sustained commands), not just validation MSE/loss.
- **Shift severity must leave real headroom.** Several milestones hit "shift is either
  trivially easy or an instant catastrophic failure" and had to screen for a severity
  that produces a real, partial success rate before drawing conclusions.
- **Document proactively, not on request:** add to `docs/references/` when a new
  source comes up, add a dated `docs/debug-log/` entry for any non-trivial bug (symptom
  → cause → fix → lesson), and write an `archive/<milestone>/README.md` when a
  milestone closes.
- **Every controller comparison needs ≥10 seeds with 95% CIs** — single-run
  differences are not defensible at a thesis defense.
- Git: commit freely as work lands; only push at milestone boundaries (ask before
  push, or match whatever cadence the user states in a session).
- Setup: `nmc` conda env (Python 3.11) for PyBullet/MuJoCo/Windows-side work (see
  `README.md` §Setup); a separate WSL2 environment (referred to as `nmc-snn` /
  `IsaacLab` in debug-log) with GPU access for Isaac Lab PPO training, 32GB
  RAM/16GB VRAM nominally required but confirmed to run on an 8GB laptop after plumbing
  fixes (see `docs/debug-log/2026-07-21_isaac-lab-wsl-8gb-bringup.md`).

## 6. Immediate next task

Run **M5**: the full 6-controller × ≥10-seed × all-metrics comparison on the nav-layer
Go2 env, with Holm/Bonferroni-corrected pairwise significance tests on recovery time
and success rate, reusing the metric suite already built for M2/M3/M4
(`src/nmc/eval/metrics.py`). Then **M6** (port the L5 SynOps energy method to the
nav-layer controllers + add noise-degradation curves). Full detail and rationale for
this ordering is in `ROADMAP.md`'s "▶ Continuation plan" section.
