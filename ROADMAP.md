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
| M2 | MLP frozen + online baselines trained | frozen MLP (imitation on M1b) reaches goal pre-shift; online MLP updates from TD target | ✅ done on the **dynamic Go2 env**, SOTA-upgraded (see [D7](docs/references/sota_decisions.md)). BC failed closed-loop (covariate shift → DAgger). **Frozen MLP (512×512, 5 seeds × 30 eps): 37% success, Wilson-95% [29,45]; SPL 0.346** (bar 30%; privileged teacher 62%). Online baseline upgraded to **eligibility-trace TD(λ)** — the fair R-STDP analogue. Metric suite (SPL/CIs) reusable for M5. SNN (M3) trains on `data/imitation_go2_dagger.npz` (61k steps) (`archive/M2_mlp_baselines_go2/`) |
| M3 | SNN pretrained (surrogate grad) matches MLP pre-shift | SNN success rate ≈ MLP within a few % pre-shift | ✅ done — **ALIF** LIF-SNN (D2 SOTA upgrade), surrogate-grad BPTT on the DAgger data, population-vote decode. Frozen closed-loop (3 seeds × 30 eps): **41% success, Wilson-95% [32,51]**, SPL 0.318 — **matches/beats the MLP's 37%** despite lower *offline* accuracy (offline≠closed-loop, again). Fires sparsely (**1.4% firing rate**, H2 preview). This is also the frozen-SNN ablation (controller 3). e-prop deferred to M5 (D1) (`archive/M3_snn_pretrain_go2/`) |
| M4 | **Pilot (go/no-go gate):** R-STDP vs pure STDP vs frozen-SNN recovery | R-STDP recovery time < frozen SNN; result plotted | ✅ **GO signal** (n=30, CIs overlap → M5 confirms). Needed a clean shift (**sensor dropout**, not obstacle density — the latter needs no re-learning), a **TD-error** third factor (D8), and **input-layer plasticity + anchoring** (D9): R-STDP **7%→30%**, the only controller to both adapt (30%, tied-best) and retain the base task (43%, highest); online-MLP recovers then catastrophically forgets (7%). (`archive/M4_pilot_go2/`) |
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
| D2 | Quadruped walks via a **trained RL velocity policy** | tracks [vx,ω] stably; walking GIF + tracking graph | ✅ done, **on the actual Go2 model** — Playground has no built-in Go2, so we ported the Go1 joystick env to Go2 ourselves: `src/nmc/rl/envs/go2/` (Menagerie's `go2_mjx.xml` + the sensors Playground's joystick task needs but Menagerie doesn't ship — `local_linvel`, `upvector`, per-foot `global_linvel` — registered into `mujoco_playground._src.locomotion` at import, no site-packages edits). PPO trained in MuJoCo Playground on the RTX 4060 (WSL2), 200M steps (~2.8 h — ~6× slower per-step than the Go1 run since Go2's XML doesn't disable body self-collision like Go1's feetonly variant), final reward 23.98; **cmd 1.0→actual 0.84±0.07 m/s, height 0.27–0.30 m, upright throughout** (`archive/D2_go2_rl_go2model/`). Tracking is a bit looser than the earlier Go1-model result (0.95 m/s) — plausibly the extra self-collision contacts eating into the reward-shaping margin; revisit reward/collision tuning if D3 needs tighter tracking. The original Go1-model run is superseded (`archive/D2_go2_rl/`, kept as reference) — train/deploy now agree on the same robot. |
| D3a | Policy exported to the Windows loop, train↔deploy parity proven | NumPy fwd pass matches brax bit-near-exactly; walks under scripted `[vx,vy,ω]` on Windows | ✅ done — parity max err 2.7e-7 over 64 vectors; 4-phase demo (fwd/turn/fast/stop) tracks all phases, upright throughout (`archive/D3a_policy_export_windows/`). `Go2RLWalker` (`src/nmc/platform/go2_rl_walker.py`) is the locomotion layer the SNN drives. NB: its `obs()` must stay in lockstep with `nmc/rl/envs/go2/joystick.py::_get_obs` |
| D3 | MuJoCo nav env (Go2 + LiDAR raycast + obstacles + mid-episode shift + reward) | dynamic replacement for `nav_env`; A* teacher re-run → demos re-collected on dynamics | ✅ done — `src/nmc/envs/go2_nav_env.py`: same obs/action/reward/privileged contract as the kinematic env; shift via pre-allocated parked mocap obstacles; group-filtered `mj_ray` LiDAR; fall detection; **~20× realtime**. Teacher works after a real bug-fix (A* post-shift freeze → progressive inflation fallback, see debug-log). Demos on dynamics: **62% success, 15,131 clean steps** → `data/imitation_go2.npz` (`archive/D3_go2_nav_env/`) |

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
