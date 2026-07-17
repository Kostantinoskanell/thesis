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

## ⚙️ ENGINE PIVOT (2026-07-17): MuJoCo + Go2, full dynamics EVERYWHERE

Decided with the user: retire the kinematic unicycle; the **entire** thesis runs on the
**official MuJoCo Go2 model with real rigid-body dynamics** (user has a Go2 in the lab).
Two-layer: a **pretrained RL locomotion policy** walks the Go2 tracking `[vx, vy, omega]`;
the SNN navigator commands velocities on top. This makes the dynamics foundation a
**prerequisite** for the science (M2–M8 now run on the dynamic Go2), not a parallel
decoupled track. PyBullet kinematic env (M1/M1b) retained only as a fast prototype/fallback.
Engine verified: MuJoCo 3.10 loads+renders the Go2 on Windows; torque actuators (PD to
stand, policy to walk). See [docs/references/locomotion.md](docs/references/locomotion.md).

### Dynamics foundation (MuJoCo Go2) — now on the critical path, BEFORE M2
| # | Milestone | Exit criterion | Status |
|---|-----------|----------------|--------|
| D1 | Go2 stands in MuJoCo under PD (full dynamics) | height holds ~0.26 m, no collapse; render + stability plot | ✅ done (`archive/P1_go2_mujoco/`) |
| D2 | Go2 walks via a **trained RL velocity policy** | tracks [vx,ω] stably; walking GIF + tracking graph | 🔟 in progress — CPG interim works (`archive/D2_go2_walk/`); building the RL policy: **train in WSL2 (JAX/MJX MuJoCo Playground) on the RTX 4060, export MLP, deploy in the Windows MuJoCo loop** (GPU passthrough verified). No pretrained artifact existed, so we train our own |
| D3 | MuJoCo nav env (Go2 + LiDAR raycast + obstacles + mid-episode shift + reward) | dynamic replacement for `nav_env`; A* teacher re-run → demos re-collected on dynamics | ☐ |

Then M2–M8 (below) run on the D3 dynamic env. Real-Go2 deploy (SDK sport mode) is a
final integrated/hardware demo after the science.

**Superseded:** the PyBullet A1 CPG walk (old "P1", `archive/P1_quadruped/`) — kept as a
reference artifact; MuJoCo Go2 + RL policy replaces it.

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
