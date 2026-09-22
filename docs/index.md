---
layout: default
title: Home
---

# Memristor-Inspired Neuromorphic Control for Robotics

Undergraduate thesis work (ECE, University of Patras). A hardware–software
co-design comparing a **spiking neural network (SNN)** with **reward-modulated
STDP (R-STDP)** against DNN/RL baselines, on a real **Unitree Go2** quadruped
simulated with full rigid-body dynamics (MuJoCo / Isaac Lab), under a
mid-episode distribution shift — plus an FPGA co-processor structurally
analogous to a memristive crossbar for the synaptic weight update.

Full proposal: [`proposal/proposal.tex`](https://github.com/Kostantinoskanell/thesis/blob/main/proposal/proposal.tex). Full milestone
plan, locked decisions, and every result in detail:
[`ROADMAP.md`](https://github.com/Kostantinoskanell/thesis/blob/main/ROADMAP.md). Every SOTA-vs-conventional design choice and why:
[`references/sota_decisions.md`](https://github.com/Kostantinoskanell/thesis/blob/main/docs/references/sota_decisions.md).

## At a glance

Two parallel studies ask the same question — *does letting a spiking network
keep learning online (R-STDP) help it recover from a shift faster than a
frozen network?* — at two different levels of the control stack, plus the
foundation both stand on:

| | Question | Headline result |
|---|---|---|
| 🧭 **Navigation** ([M-track](#m--navigation-layer-the-core-study)) | Can the SNN *deciding where to go* recover from a broken sensor / changed terrain? | Energy: **10.4× cheaper** than the MLP. Robustness: real, but it's the **neuron model** doing it, not R-STDP. Recovery via R-STDP itself: **no significant effect**, confirmed 6 independent ways. |
| 🦿 **Locomotion** ([L-track](#l--locomotion-layer-pushing-plasticity-into-the-gait)) | Can the SNN *walking the robot* recover from icy terrain? | First working **spiking-network-controlled Go2 walk** found in the literature search, energy-positive. R-STDP on the gait itself **destabilizes** it — an honest negative. |
| 🏗️ **Foundation** ([D-track](#d--dynamics-foundation-getting-the-real-robot-to-walk)) | Get the real Go2 walking in full-physics simulation at all | Trained from scratch (no pretrained policy existed), exported to a bit-exact Windows runtime. |

Every result below is backed by seeded, multi-run statistics (Holm-corrected
significance tests, 95% CIs) — not single lucky runs. Two of the most
interesting findings in this project are honest **negative** results that
survived heavy scrutiny, not the ones that confirmed the original hypothesis.

---

## D — dynamics foundation (getting the real robot to walk)

Before any plasticity science could run, the Go2 had to actually stand and
walk under full physics — no pretrained policy for this robot existed, so it
was trained from scratch (PPO, MuJoCo Playground), then exported to a
pure-NumPy runtime with bit-exact train↔deploy parity.

<img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/D3a_policy_export_windows/walk_windows.gif" width="420" alt="Go2 walking under the trained RL policy">

*Trained locomotion policy tracking a velocity command — the walking
"actuator" every controller below drives.*

Details: [`archive/D2_go2_rl_go2model/`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/D2_go2_rl_go2model), [`archive/D3a_policy_export_windows/`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/D3a_policy_export_windows), [`archive/D3_go2_nav_env/`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/D3_go2_nav_env).

---

## M — navigation layer (the core study)

An SNN decides *where to go* (avoid obstacles, reach a goal) from LiDAR;
a trained locomotion policy (above) actually walks the robot there. The
question: when a distribution shift hits (dead LiDAR beams, changed terrain),
does releasing the SNN to online R-STDP help it recover faster than a frozen
network?

<img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/D3_go2_nav_env/nav_episode.gif" width="420" alt="Go2 navigating around obstacles with a mid-episode shift">

*The navigation task: reach the goal, avoid obstacles, survive a mid-episode
distribution shift.*

**What we found, after six independent checks (not one lucky run):**
- **Energy — a real win.** The frozen SNN is **10.4× cheaper** than the MLP on
  the standard SynOps metric, 2.4× on full honest accounting (counting neuron
  updates and the spike encoder, which the literature usually skips).
- **Shift-robustness — real, but wrongly attributed.** A pre-registered
  ablation found the robustness M4's pilot credited to R-STDP is actually a
  property of the **ALIF neuron model** (frozen-LIF 7% → frozen-ALIF 17%,
  p=0.05) — adding R-STDP on top of ALIF adds *nothing* (p=0.30).
- **R-STDP recovery itself — no effect.** M4's single-seed pilot showed a
  promising 30% vs 17% signal; at proper multi-seed rigor (10 seeds × 30 eps,
  Holm-corrected) that signal **did not replicate** — checked across two shift
  severities, a hyperparameter sweep, two plasticity scopes, and two
  third-factor signals, all in agreement.
- **Noise robustness — refuted.** The standard "SNNs degrade more gracefully"
  claim does not hold here: the MLP is essentially noise-immune while the SNN
  loses more than half its success rate, because rate-coding *samples* the
  sensor instead of reading it directly.
- On **terrain** shifts specifically, R-STDP *does* close the full gap to the
  MLP — so the effect is shift-dependent, not universally absent.

Full write-ups: [`archive/M5_full_comparison/`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/M5_full_comparison) (the
core statistical result), [`archive/M4b_extras/`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/M4b_extras) (the
neuron-model ablation), [`archive/M6_energy/`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/M6_energy) (energy +
noise), [`archive/M4b_terrain_walk_compare/`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/M4b_terrain_walk_compare)
(the terrain result). Decision log: [D16–D17](https://github.com/Kostantinoskanell/thesis/blob/main/docs/references/sota_decisions.md).

---

## L — locomotion layer (pushing plasticity into the gait)

The M-track found that R-STDP at the *navigation* level can't fix a fault
that's really a *body/physics* problem (e.g. slipping on ice) — it's below
the navigator's interface. This track pushes the spiking network and R-STDP
**into the gait itself**.

<img src="https://raw.githubusercontent.com/Kostantinoskanell/thesis/main/archive/L4_gait_check/dagger_walk_forward_v3.gif" width="420" alt="Spiking neural network walking the Go2">

*A spiking neural network is the walking controller here — not just the
navigator. First working spiking-Go2 locomotion policy found in the
literature search.*

**What we found:**
- Training a spiking walking policy directly via PPO **collapsed into a
  belly-flop local optimum** that looked like a capacity ceiling until the
  rollout was actually rendered and watched (reward alone hid it completely).
- Fixed via **distillation** from a walking MLP teacher + **DAgger** for
  robustness to sustained commands — the walking GIF above is the result,
  fully closing the loop after two more rounds of "watch the video, fix what's
  actually wrong" (it was walking too crouched; a rear leg was dragging).
- **Energy: positive, but only barely** (1.04× cheaper than the MLP, after
  sparsity regularization and minimizing timesteps) — a sharp contrast with
  the navigation layer's 10.4×, and a clean illustration that sparse spike
  coding suits coarse navigation far better than continuous motor control.
- **R-STDP on the gait: destabilizes it.** Releasing the walking policy to
  R-STDP under an icy-terrain shift made the gait progressively collapse
  rather than recover — the opposite of the (partial) nav-layer effect.

Full write-ups: [`archive/L4_gait_check/`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/L4_gait_check), [`archive/L5_energy/`](https://github.com/Kostantinoskanell/thesis/tree/main/archive/L5_energy).

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
fpga/                     HDL + PyNQ host (not yet started — see ROADMAP M7/M8)
docs/references/          curated citations + the full dated SOTA-decision log
docs/debug-log/           dated war-stories: symptom -> cause -> fix -> lesson
archive/<milestone>/      figures, GIFs, and full write-ups per milestone
```

**Heavily documented on purpose**: every non-trivial bug is in
[`docs/debug-log/`](https://github.com/Kostantinoskanell/thesis/tree/main/docs/debug-log/), every design decision (and every SOTA
option considered and rejected) is in
[`docs/references/sota_decisions.md`](https://github.com/Kostantinoskanell/thesis/blob/main/docs/references/sota_decisions.md), and
every milestone has a full write-up in `archive/`.

## Status

M0–M6 (navigation) and L0–L5 (locomotion) are done — see the table above and
[`ROADMAP.md`](https://github.com/Kostantinoskanell/thesis/blob/main/ROADMAP.md) for full detail. **Next: M7/M8, the FPGA
co-processor.**

---

*This page is the GitHub Pages front page for the project repository:*
[*github.com/Kostantinoskanell/thesis*](https://github.com/Kostantinoskanell/thesis)
