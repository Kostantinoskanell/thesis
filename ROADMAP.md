# Thesis Roadmap — Neuromorphic Control for Robotics

Locked scoping decisions (2026-07-17):

- **Plasticity:** R-STDP is the **primary** rule the headline hypothesis rests on.
  Pure STDP is kept as an *ablation*, not the main plan. (Reason: pure STDP is
  Hebbian and task-agnostic — it cannot support a "faster recovery" claim.
  R-STDP adds one eligibility trace + a scalar reward broadcast, which the
  software and FPGA are already structured for.)
- **FPGA:** Full hardware/software co-design is in scope from the start (student
  has HDL experience). But the **software science is front-loaded** so no HDL
  time is spent until the R-STDP recovery result exists in simulation.
- **Resources:** GPU available; FPGA board obtainable; HDL experience present.

## De-risking order (do NOT start with HDL)

1. **Env + shift protocol** — `src/nmc/envs/nav_env.py` (fill the `NotImplementedError`
   PyBullet stubs).
2. **MLP baselines** — frozen + online-TD (`controllers/mlp.py`), imitation
   pretraining from a scripted expert.
3. **★ The pilot (highest priority):** pretrain the SNN with surrogate gradients,
   then measure what pure STDP vs R-STDP actually do to the pretrained policy
   under the shift. **This one experiment decides whether the thesis works —
   before any FPGA effort.**
4. Only if the pilot shows real R-STDP recovery → build the FPGA accelerator.

## Milestones

| # | Milestone | Exit criterion | Status |
|---|-----------|----------------|--------|
| M0 | Repo scaffold + plasticity golden reference | STDP/R-STDP unit tests green | ✅ done |
| M1 | PyBullet env runs an episode with mid-episode shift | random policy completes a 60 s episode, shift fires at t=30 s | ✅ done |
| M1b | **Privileged A\* teacher** + imitation dataset | ~100% static avoidance; 13k+ clean (obs,action) demo steps logged | ✅ done |
| M2 | MLP frozen + online baselines trained | frozen MLP (imitation on M1b) reaches goal pre-shift; online MLP updates from TD target | ☐ |
| M3 | SNN pretrained (surrogate grad) matches MLP pre-shift | SNN success rate ≈ MLP within a few % pre-shift | ☐ |
| M4 | **Pilot (go/no-go gate):** R-STDP vs pure STDP vs frozen-SNN recovery | R-STDP recovery time < frozen SNN; result plotted | ☐ |
| M5 | Full comparison: all 5 controllers × all metrics, **≥10 seeds w/ 95% CIs** | metrics table + error bars reproduced from real runs; sig. test on recovery time | ☐ |
| M6 | Robustness + energy sweeps (H2, H4) | SynOps + noise-degradation curves (per-seed CIs) | ☐ |
| M7 | FPGA STDP datapath, LUT validated vs golden ref (<2% err) | Verilog Δw matches `kernel_reference` in sim (cosim) | ☐ |
| M8 | Hardware-in-loop + network-size speedup sweep (H3) | kernel & end-to-end speedup vs CPU/GPU, crossover point found | ☐ |
| M9 | Writing, figures, defense prep | thesis draft complete | ☐ |

**Every run is seeded and config-driven** (`configs/`), and every controller is
evaluated over **≥10 random seeds** so comparisons carry confidence intervals — a
single-run recovery-time difference is not defensible at a defense.

**Every simulation milestone produces a visual** — at minimum one rendered episode
GIF (`scripts/render_episode.py`) and a diagnostic plot — saved to `archive/`, both to
document the journey and to *verify behaviour by eye* (watching the robot catches bugs
metrics hide).

## Platform track (Go2 quadruped) — PARALLEL, decoupled from the science

The SNN's `[vx, vy, omega]` interface is unchanged; a locomotion layer walks the robot.
Runs in parallel with M2–M8 (which use the fast kinematic model) — for realistic visuals
and a final integrated/hardware demo. See [docs/references/locomotion.md](docs/references/locomotion.md).

| # | Milestone | Exit criterion | Status |
|---|-----------|----------------|--------|
| P1 | Quadruped (A1 URDF) walks in PyBullet from velocity commands | ✅ done — CPG trot tracks [vx,ω], stays upright; GIF + gait-diagnostics graph in `archive/P1_quadruped/` (convex-MPC is the tracking-fidelity upgrade) | ✅ |
| P2 | Swap in Go2 URDF; nav env drives the walking robot's base | SNN action → velocity → gait; obstacle-avoidance episode renders with legs | ☐ |
| P3 | Real-Go2 deploy interface (unitree_sdk2 sport mode) | SNN velocity commands drive the physical Go2 (hardware demo) | ☐ |

**Sequencing:** science stays on the critical path (M2 next). P1 can start whenever a
compelling walking visual is wanted; it does not gate the plasticity results.

## Open technical risks (revisit each milestone)

- **STDP destroys the pretrained policy.** The "release to plasticity" step may
  drift weights away from the good solution. Mitigation: small η, R-STDP gating,
  apply plasticity to the readout layer first (see `SNNController`).
- **Rate-coded LiDAR lacks temporal structure** STDP can exploit. Check in M4.
- **Plasticity-layer scope.** Readout-layer plasticity is enough for the R-STDP
  *policy* claim (H1), but the pure-STDP *representation-level* ablation claim needs
  plasticity on the **sensory/input layer** too. Extend `SNNController` to target a
  chosen layer before M4 if the ablation is to be measured properly.
- **FPGA may lose to GPU** at these tiny network sizes. That is a *valid finding*;
  H3 is framed as "find the crossover point," not "beat the GPU."
- **Python 3.13 incompatibility** with torch/pybullet → we use a **conda env `nmc`
  (Python 3.11)**; PyBullet installed from conda-forge (no MSVC needed).

## Procurement / lead-time actions

- **Order the FPGA board now** even though HDL work starts ~week 18. It only needs to
  *arrive* by Phase 4; ordering early costs nothing and removes shipping/customs risk
  from the critical path. Recommended: Digilent **Arty Z7-20** (Zynq-7020, PyNQ-ready).
- GPU already available → no cloud spend needed yet. Physical robot/LiDAR: **defer**
  (simulation-only covers M1–M8; buy only if a physical demo is added).
