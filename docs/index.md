---
layout: default
title: Home
---

# Memristor-Inspired Neuromorphic Control for Robotics

Undergraduate thesis work (ECE, University of Patras). A hardware-software
co-design comparing a spiking neural network (SNN) with reward-modulated STDP
(R-STDP) against DNN/RL baselines, on a real Unitree Go2 quadruped simulated
with full rigid-body dynamics (MuJoCo / Isaac Lab), under a mid-episode
distribution shift, plus a planned FPGA co-processor structurally analogous
to a memristive crossbar.

Full proposal: [`proposal/proposal.tex`](https://github.com/Kostantinoskanell/thesis/blob/main/proposal/proposal.tex) &middot;
Full milestone plan: [`ROADMAP.md`](https://github.com/Kostantinoskanell/thesis/blob/main/ROADMAP.md) &middot;
Every design decision and why: [`sota_decisions.md`](https://github.com/Kostantinoskanell/thesis/blob/main/docs/references/sota_decisions.md)

## System architecture

<img src="assets/img/architecture.svg" alt="Two-layer control architecture">

Two-layer control, closed loop, under a shift: a navigator decides *where to
go*, a locomotion policy walks the robot there. R-STDP is tested at both
insertion points (M-track and L-track) against the same frozen baseline.

## Core mechanisms

<img src="assets/img/concepts.svg" alt="LIF neuron, STDP, R-STDP, memristor analogy">

| Mechanism | One line |
|---|---|
| LIF / ALIF neuron | Leaky integrate-and-fire; ALIF adds an adaptive threshold that rises per spike and decays, giving longer temporal memory. |
| STDP | Hebbian: `+A_+ exp(-dt/tau)` if pre fires before post (LTP), `-A_- exp(dt/tau)` if post before pre (LTD). No notion of task success. |
| R-STDP | STDP accumulates into an eligibility trace `e_ij`; a reward/TD-error `r(t)` gates consolidation: `dw = eta * r(t) * e_ij`. |
| Memristor analogy | Conductance `g` in `[g_min, g_max]` &equiv; synaptic weight; the FPGA reproduces this *structure* (state retention, saturation, locality), not the device physics. |

## Results at a glance

| Track | Question | Metric | Result |
|---|---|---|---|
| M (navigation) | Energy vs MLP | SynOps / full accounting | 10.4x / 2.4x cheaper |
| M (navigation) | Shift-robustness source | frozen-LIF vs frozen-ALIF | +10 pts (p=0.05) &mdash; it's the neuron model |
| M (navigation) | R-STDP recovery vs frozen SNN | sensor dropout, 10 seeds x 30 eps | no significant effect (6 checks) |
| M (navigation) | R-STDP recovery vs frozen SNN | terrain (sand/ice) | full recovery to MLP level |
| M (navigation) | Noise robustness (H4) | success at sigma=0.8 | MLP 38% vs SNN 18% &mdash; refuted |
| L (locomotion) | Energy vs MLP | best config (T=5, sparsity-regularized) | 1.04x cheaper |
| L (locomotion) | R-STDP on the gait | icy terrain | destabilizes it (honest negative) |
| L (locomotion) | Walking policy | spiking-network-controlled Go2 | first working one found in lit. search |
| D (foundation) | Locomotion policy | trained from scratch, PPO | bit-exact sim-to-Windows parity |

---

## D &mdash; dynamics foundation

Go2 stands and walks under full physics before any plasticity science runs;
no pretrained policy existed, so one was trained from scratch and exported to
a bit-exact Windows runtime.

<div class="gif-row">
<figure><img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/P1_go2_mujoco/stand.gif" alt="Go2 standing under PD control"><figcaption class="cap">D1: standing, PD control</figcaption></figure>
<figure><img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/D3a_policy_export_windows/walk_windows.gif" alt="Go2 walking under the exported policy"><figcaption class="cap">D2/D3a: walking, exported policy</figcaption></figure>
<figure><img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/D3a_policy_export_windows/fig_windows_tracking.png" alt="Velocity tracking, Windows export"><figcaption class="cap">velocity-tracking parity, train vs. deploy</figcaption></figure>
</div>

| Milestone | Result |
|---|---|
| D1 | height holds ~0.26 m under PD, no collapse |
| D2 | PPO-trained, cmd 1.0 &rarr; actual 0.84 &plusmn; 0.07 m/s |
| D3a | NumPy-vs-JAX parity, max err 2.7e-7 |
| D3 | dynamic nav env: LiDAR raycast + obstacles + shift, ~20x realtime |

Details: [`D2_go2_rl_go2model`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/D2_go2_rl_go2model), [`D3a_policy_export_windows`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/D3a_policy_export_windows), [`D3_go2_nav_env`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/D3_go2_nav_env)

---

## M &mdash; navigation layer

An SNN/MLP decides *where to go* from LiDAR; the D-track policy walks the
robot there. Question: does releasing the SNN to R-STDP recover faster than a
frozen network after a shift?

<div class="gif-row">
<figure><img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/D3_go2_nav_env/nav_episode.gif" alt="Go2 navigating with a mid-episode shift"><figcaption class="cap">the task: goal + obstacles + mid-episode shift</figcaption></figure>
<figure><img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/M1b_expert/episode.gif" alt="Scripted A* expert episode"><figcaption class="cap">privileged A* teacher (imitation-data source)</figcaption></figure>
<figure><img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/M4b_terrain_walk_compare/snn_rstdp_sand.gif" alt="R-STDP recovering on sand"><figcaption class="cap">R-STDP on sand: closes the gap to the MLP</figcaption></figure>
<figure><img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/M4b_terrain_walk_compare/snn_rstdp_ice_mu028_warm15.gif" alt="R-STDP recovering on ice"><figcaption class="cap">R-STDP on ice (mu=0.28): 40% vs frozen 27%</figcaption></figure>
</div>

**Core comparison, sensor dropout (10 seeds x 30 eps each, 95% CI, Holm-corrected):**

| Controller | Success @ dropout=0.30 | Success @ dropout=0.20 (corrected) | Recovery time (eps) |
|---|---|---|---|
| Frozen MLP | 30.3% &plusmn; 7.7% | 34.0% &plusmn; 9.5% | 4.2 / 6.4 |
| Online MLP | 21.0% &plusmn; 12.5% | 24.0% &plusmn; 13.4% | 9.2 / 6.3 |
| Frozen SNN | 11.0% &plusmn; 2.6% | 21.3% &plusmn; 7.2% | 20.3 / 7.7 |
| Pure-STDP SNN | 5.0% &plusmn; 3.4% | 8.7% &plusmn; 4.5% | 28.9 / 20.9 |
| **R-STDP SNN** | 8.7% &plusmn; 3.7% | 19.3% &plusmn; 5.3% | 18.4 / 11.8 |
| TM-NORM SNN | 1.3% &plusmn; 2.2% | 1.7% &plusmn; 2.2% | 30.0 / 30.0 |

R-STDP vs frozen SNN: not significant at either severity (p=0.14, p=0.70).
0.30 was M4's original pick and turned out already floored &mdash; see the
severity sweep below.

**Why: the neuron-model ablation (pre-registered, dropout=0.20, 10 seeds x 10 eps):**

| Variant | Success | vs. previous |
|---|---|---|
| Frozen LIF | 7.0% &plusmn; 5.9% | &mdash; |
| Frozen ALIF | 17.0% &plusmn; 9.0% | +10.0 pts, p=0.051 (neuron model alone) |
| R-STDP + ALIF | 12.0% &plusmn; 5.6% | -5.0 pts, p=0.30 (plasticity adds nothing) |

**Severity sweep, all six controllers (5 seeds x 10 eps):**

| Controller | drop=0.10 | drop=0.15 | drop=0.20 | drop=0.30 |
|---|---|---|---|---|
| Frozen MLP | 50.0% | 34.0% | 22.0% | 22.0% |
| Online MLP | 44.0% | 32.0% | 16.0% | 30.0% |
| Frozen SNN | 42.0% | 30.0% | 20.0% | 12.0% |
| Pure-STDP SNN | 10.0% | 8.0% | 4.0% | 2.0% |
| R-STDP SNN | 48.0% | 24.0% | 12.0% | 12.0% |
| TM-NORM SNN | 2.0% | 0.0% | 2.0% | 2.0% |

**Terrain shift (frozen MLP / frozen SNN / R-STDP SNN):**

| Terrain | Frozen MLP | Frozen SNN | R-STDP SNN |
|---|---|---|---|
| Sand (mu=1.20) | 47% | 27% | **47%** (full recovery) |
| Ice (mu=0.28) | 33% | 27% | **40%** (beats both) |

**Energy per decision, 45 nm Horowitz model (nJ):**

| Controller | Inference | Neurons | Encoder | Learning | Total | vs. MLP |
|---|---|---|---|---|---|---|
| Frozen MLP | 1321.7 | &mdash; | &mdash; | &mdash; | 1321.7 | 1.00x |
| Frozen SNN | 125.6 | 420.0 | 4.2 | &mdash; | 549.7 | 2.40x cheaper |
| R-STDP SNN (dense) | 125.1 | 420.0 | 4.2 | 15538.5 | 16087.7 | 0.08x (12x costlier) |
| R-STDP SNN (event-driven, FPGA target) | 125.1 | 420.0 | 4.2 | 747.3 | 1296.5 | 1.02x cheaper |
| Online MLP (backprop) | 1321.7 | &mdash; | &mdash; | 7586.7 | 8908.4 | 0.15x costlier |

Literature-standard view (SynOps only): frozen SNN is **10.4x cheaper**. Per
*successful* navigation, the win disappears: frozen SNN 2276 &mu;J vs. frozen
MLP 2280 &mu;J &mdash; a dead heat.

**Noise robustness (H4), success rate vs. LiDAR noise sigma:**

| Controller | 0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.8 |
|---|---|---|---|---|---|---|
| Frozen MLP | 42% | 52% | 34% | 42% | 34% | **38%** |
| Frozen SNN | 40% | 50% | 40% | 28% | 18% | **18%** |
| R-STDP SNN | 42% | 48% | 30% | 28% | 36% | 34% |

H4 hypothesized the SNN degrades more gracefully; the opposite happened &mdash;
rate coding *samples* the sensor, so noise compounds with sampling noise.

<div class="gif-row">
<figure><img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/M4b_extras/fig_severity_sweep.png" alt="severity sweep figure"></figure>
<figure><img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/M6_energy/fig_m6_energy.png" alt="energy breakdown figure"></figure>
<figure><img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/M6_energy/fig_h4_noise_high.png" alt="noise sweep figure"></figure>
<figure><img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/M4b_extras/fig_trajectory_overlay.png" alt="trajectory overlay figure"></figure>
<figure><img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/M4_pilot_go2/fig_pilot_sensor.png" alt="M4 pilot recovery figure"></figure>
<figure><img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/M0_scaffold/fig_stdp_kernel.png" alt="STDP kernel golden reference"></figure>
</div>

Full write-ups: [`M5_full_comparison`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/M5_full_comparison), [`M4b_extras`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/M4b_extras), [`M6_energy`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/M6_energy), [`M4b_terrain_walk_compare`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/M4b_terrain_walk_compare)

---

## L &mdash; locomotion layer

The M-track found R-STDP can't fix a *body/physics* fault (icy terrain) from
the navigation layer &mdash; it's below the navigator's interface. This track
puts the spiking network in charge of the gait itself.

<div class="gif-row">
<figure><img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/L4_gait_check/dagger_walk_forward_v3.gif" alt="Spiking Go2 walking, robust"><figcaption class="cap">final spiking walker, DAgger-robustified</figcaption></figure>
<figure><img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/L5_energy/sparse_t5_v3_walk.gif" alt="Sparse T5 spiking walker"><figcaption class="cap">sparsified, T=5, energy-positive</figcaption></figure>
<figure><img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/L4_gait_check/fig_distilled_gait.png" alt="gait diagnostic figure"></figure>
</div>

| Stage | Result |
|---|---|
| Direct PPO training | collapsed into a belly-flop local optimum (looked like a capacity ceiling until rendered) |
| Distillation + DAgger | walking, robust to sustained commands (final GIF above) |
| Energy (dense, T=8) | 328.9 nJ, **0.57x (costlier)** than the MLP (186.1 nJ) |
| + firing-rate penalty | 266.9 nJ, 0.70x (still costlier) |
| + penalty, T=5 (best) | 178.1-181.0 nJ, **1.04x cheaper**, still walks |
| T=4 | 136.8 nJ, 1.36x cheaper, but **breaks walking** |
| R-STDP on the gait (icy shift) | fall rate rises 25% &rarr; 53% &mdash; **destabilizes**, does not recover |

Full write-ups: [`L4_gait_check`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/L4_gait_check), [`L5_energy`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/L5_energy)

---

## Repo layout

```
src/nmc/
  envs/go2_nav_env.py     MuJoCo Go2 nav env: LiDAR raycast, obstacles, mid-episode shift
  platform/go2_rl_walker.py   the trained locomotion policy every controller drives
  controllers/            frozen/online MLP, LIF/ALIF-SNN (+R-STDP), TM-NORM baselines
  locomotion/             PopSAN spiking actor + R-STDP for the gait itself (L-track)
  plasticity/stdp.py      online STDP / R-STDP (golden reference for the FPGA)
  eval/{metrics,energy}.py    SynOps/CI metrics + the 45nm energy model (M6/L5)
fpga/                     HDL + PyNQ host (not yet started, see ROADMAP M7/M8)
docs/references/          curated citations + the full dated SOTA-decision log
docs/debug-log/           dated war-stories: symptom -> cause -> fix -> lesson
archive/<milestone>/      figures, GIFs, and full write-ups per milestone
```

Every non-trivial bug: [`docs/debug-log`](https://github.com/Kostantinoskanell/thesis/tree/main/docs/debug-log/).
Every design decision: [`docs/references/sota_decisions.md`](https://github.com/Kostantinoskanell/thesis/blob/main/docs/references/sota_decisions.md).

## Setup

```powershell
# nav-layer / M-track (Windows, CPU): conda env `nmc`, Python 3.11
conda create -n nmc python=3.11 -y
conda install -n nmc -c conda-forge pybullet numpy matplotlib -y
conda run -n nmc python -m pip install snntorch stable-baselines3 gymnasium mujoco
```

Locomotion-track (L) training runs in WSL2 + Isaac Lab (GPU) &mdash; see
[`docs/debug-log/2026-07-21_isaac-lab-wsl-8gb-bringup.md`](https://github.com/Kostantinoskanell/thesis/blob/main/docs/debug-log/2026-07-21_isaac-lab-wsl-8gb-bringup.md).

## Status

M0-M6 (navigation) and L0-L5 (locomotion) are done. Next: M7/M8, the FPGA
co-processor. Full detail: [`ROADMAP.md`](https://github.com/Kostantinoskanell/thesis/blob/main/ROADMAP.md).

---

*Front page for [github.com/Kostantinoskanell/thesis](https://github.com/Kostantinoskanell/thesis).*
