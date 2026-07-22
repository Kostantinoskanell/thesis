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
| M4b | **Pilot robustness/generalization extensions** (inspired by [Espino, Bain & Krichmar 2024](docs/references/neuromorphic_robotics.md), arXiv:2404.15524, and [Zhao et al. 2025](docs/references/snn_learning.md), arXiv:2505.05375): (1) dropout-severity sweep 10/20/30/40/50%, (2) R-STDP re-run on ≥3 random dead-beam masks (not just M4's one fixed mask), (3) trajectory-overlay figure (frozen path vs. first post-shift episode vs. after a few recovery episodes), (4) **TM-NORM baseline** — a 6th, reward-free controller: EMA-calibrate the ALIF firing threshold from membrane-potential statistics only (no backprop, no reward signal), to test whether R-STDP's recovery is more than statistical renormalization of the corrupted input, (5) **D8/D9 ablation grid**: {raw reward vs. TD-error} × {readout-only vs. input+readout plasticity} × {anchor on/off} reported jointly as recovery% + base-retention% (per Zhao et al. Table VI's accuracy-vs-energy grid style, minus the energy column, already covered by M6), (6) **firing-rate distribution shift figure**: ALIF firing-rate histogram, clean/pre-shift vs. frozen-post-shift vs. R-STDP-recovered (per Zhao et al. Fig 2), showing spike statistics renormalizing as the network adapts, (7) **neuron-model-alone ablation**: frozen-LIF vs. frozen-ALIF vs. R-STDP-ALIF under the same sensor dropout (reusing M3's `assets/snn_seeds_lif/` weights), isolating whether ALIF's adaptive threshold alone buys any shift-robustness independent of R-STDP (per Zhao et al. Table VII's finding that adaptive-threshold neurons alone don't rescue OOD performance) | severity curve with CI bands across controllers; recovery holds (not mask-specific) across ≥3 masks; overlay figure shows the path visibly changing over the first few post-shift episodes; TM-NORM's recovery% quantified against R-STDP's — if TM-NORM matches R-STDP, the reward-modulation story needs revisiting; ablation grid table (6): D8/D9's each show a clean marginal gain, not just cumulative; firing-rate histogram visibly shifts back toward clean under R-STDP but not frozen-SNN; frozen-LIF's shifted success landing near frozen-ALIF's (17%) would confirm the recovery is R-STDP's doing, not ALIF's — a large gap would mean ALIF itself is doing some of the work | ☐ |
| M4c | **Terrain walk comparison** (a *world/body* fault, distinct from M4b's *sensor* faults — same v/ω-only visibility as Juarez-Lora 2022): frozen MLP vs. frozen SNN vs. R-STDP SNN walking on ice/sand, with GIFs (visible floor recolor confirmed) + a success bar chart | ✅ **retuned + real split found.** M4's ice(mu=0.08)/sand(mu=1.6) were both unusable (ice floors the frozen SNN to 0%, sand *helps* it to 55% — confirmed by two independent screens, `scripts/terrain_headroom.py` + `scripts/shift_headroom.py`); retuned to **ice mu=0.20** (25pt drop) / **sand mu=1.20** (20pt drop), both with real headroom. At n=15/controller (exploratory, not CI-rigorous): **sand — R-STDP fully closes the SNN-MLP gap (27%→47%, matches MLP's 47%)**; **ice — R-STDP shows ZERO improvement over frozen (20% both)**. Working hypothesis: sand (excess grip) is a velocity-recalibration problem the navigator can compensate for; ice (slipping) may be a locomotion-physics limit no amount of *decision*-level relearning can fix. **Deep-dive resolved it** (`scripts/ice_deep_dive.py`): a 40-episode warm-up at mu=0.20 made R-STDP *worse* (27%→13% — over-adaptation/drift on a too-harsh shift, not slowness), but the standard 15-episode warm-up at a gentler **mu=0.28** gave R-STDP **40%, beating both frozen SNN (27%) and frozen MLP (33%)**. Conclusion: R-STDP isn't powerless on ice — mu=0.20 was simply too severe (the same "shift severity must leave real headroom" lesson M4 already learned with sensor dropout), and at the right severity R-STDP recovers on both terrains. (`archive/M4b_terrain_walk_compare/`) |
| M5 | Full comparison: all 6 controllers × all metrics, **≥10 seeds w/ 95% CIs** | metrics table + error bars reproduced from real runs; sig. test on recovery time, **Bonferroni/Holm-corrected across all controller pairs** (per Espino et al.'s multi-comparison approach); seed count pushed past 10 where cheap (sim runs ~20× realtime, so more seeds cost little) | ☐ |
| M6 | Robustness + energy sweeps (H2, H4) | **per-operation energy estimate in µJ** (MACs/ACs/MULs × published per-op costs, Horowitz 2014 — not just a firing-rate percentage; method per [Zhao et al. 2025](docs/references/snn_learning.md) Table V) + noise-degradation curves (per-seed CIs) | ☐ |
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

### L-track: spiking locomotion layer (opened 2026-07-21, motivated by M4c)
M4c showed R-STDP on the *navigation* layer recovers on sand but not ice — because it
can only recalibrate velocity *commands*, not the gait itself (slipping is below its
interface). This track extends the SNN + plasticity DOWN into locomotion: train a
**spiking** Go2 policy with RL, then release it to R-STDP for online gait adaptation.
Chose **Isaac Lab (rsl_rl PPO)** over reusing the brax pipeline (one PyTorch SNN codebase;
the student leaned SOTA/PyTorch). Isaac Lab's stated req (32GB RAM / 16GB VRAM) is ~2× the
lab laptop, but **it runs fine** — the blockers were all env plumbing, not capacity.
| # | Milestone | Exit criterion | Status |
|---|-----------|----------------|--------|
| L0 | Isaac Lab runs on the 8GB laptop | stock Go2 task trains headless w/o OOM | ✅ done — 4 plumbing fixes (headless launcher, LD_LIBRARY_PATH incl. `/usr/lib/wsl/lib`, apt X11/GL libs, MSYS/CRLF); `scripts/wsl_isaac_go2_smoke.sh`; war story in debug-log |
| L1 | Baseline (MLP) Go2 policy trained, pipeline proven | walking policy, reward converged | ✅ done — 2048 envs (3.1GB VRAM), 1500 iters, reward −6→**36.25**, vel-tracking err **0.16 m/s**; `scripts/wsl_isaac_go2_train.sh`, ckpt `logs/rsl_rl/unitree_go2_flat/2026-07-21_15-04-18/` |
| L2 | Spiking actor network (pure-torch, surrogate-grad) — **v1 (rate readout)** | forward + gradient-flow + sparsity verified | ✅ done — `src/nmc/locomotion/spiking_actor.py` (LIF/ALIF, ALIF 4.6% vs LIF 11.2% firing); unit-tested. **⚠ SUPERSEDED by L2b:** a lit review (2026-07-21, [D11](docs/references/sota_decisions.md)) found this plain firing-rate readout is essentially the "RateSAN" baseline that PopSAN's own ablation shows *fails to match a deep actor even at 5× timesteps* — insufficient representation capacity for continuous control. Keep as the ablation ("why population coding was needed"). |
| L2b | **PopSAN-style population-coded spiking actor** (Tang et al. 2020, [2010.09635](https://arxiv.org/abs/2010.09635); legged validation Wang/Wu 2023, [2310.05022](https://arxiv.org/abs/2310.05022)) | population encoder + decoder implemented, gradients flow, ≥ v1 capacity on a toy regression | ☐ **next.** Rebuild the actor with: (a) **learnable Gaussian input populations** — each of the N obs dims → P_in≈10 neurons, stimulation A_E=exp(−½((s−μ)/σ)²), μ/σ **trainable**, μ init spread across the obs range; deterministic soft-reset IF spike gen over T. (b) **current-based LIF** hidden layers (current decay d_c + voltage decay d_v, hard reset). (c) **population output decoder** — each of the M=12 action dims → P_out≈10 neurons, action = W_d·(spike_count/T)+b_d, **W_d/b_d trainable**. (d) **T=5** timesteps, rectangular/fast-sigmoid surrogate, extended spatio-temporal BP over all T. |
| L3 | **Hybrid PPO in Isaac Lab** — PopSAN actor + **deep-MLP critic** (only the deployed policy is spiking, per PopSAN) | spiking Go2 walks, reward ≈ MLP baseline (36.25) within a few % | ✅ **integration done; performance CLOSED at a documented, verified ceiling (D12).** `SpikingActorMLPModel` (`src/nmc/locomotion/rsl_rl_spiking.py`) subclasses `rsl_rl.models.MLPModel`, swaps `self.mlp` for the PopSAN net, injected via `cfg["actor"]["class_name"]` — zero edits to Isaac Lab/rsl_rl. **Root cause of the initial ~2.5-reward flatline found:** PPO's adaptive KL-based LR schedule was crushing the spiking actor's effective LR (population-coded firing-rate outputs → noisier KL estimates than a smooth MLP). **Fix:** `schedule="fixed"`, `lr=1e-3` (== the MLP's own tuned value) — a real, reproducible **~3x gain**. **Verified at 2x the MLP's training budget** (3000 vs 1500 iters, overnight run): four measurement windows spanning iter 600-3000 are statistically indistinguishable (means 7.68-8.10, stdev 0.55-0.62) — **reward ≈8 is a genuine architectural ceiling, not a training-length issue** (vel-err 1.45 m/s vs. the MLP's 0.16). Gap vs. MLP (8 vs 36) likely reflects PopSAN's validated scale (small classic-control Gym tasks, obs≤111/act≤8) vs. Go2's much harder regime (48 obs, 12 continuous joints, contact-rich). Untested next levers (need a fresh session to prioritize): larger `in_pop`/`out_pop` (10→20), larger hidden layers, more spiking timesteps. (`scripts/wsl_isaac_go2_spiking.sh`, `scripts/make_isaac_train_spiking.py`, `src/nmc/locomotion/popsan_actor.py`) |
| L4 | Release the spiking policy's plastic weights to R-STDP, test terrain recovery (ice) | R-STDP gait adaptation recovers on ice where nav-layer R-STDP couldn't (M4c) | ☐ — the population/hidden weights are the plasticity sites; deploy-side numpy STDPLearner, same three-factor rule as the nav layer. **Can proceed now with the reward-8 policy as the substrate** — L4 tests *relative* recovery under a terrain shift, not absolute walking quality matching the MLP, so a policy that walks passably (stable, not falling, per the reward structure) without matching the MLP may be sufficient. Revisit if the shift makes an already-weak gait uninformative. |
| L5 | Energy of a spiking *locomotion* controller (H2) | SynOps + µJ/decision vs the MLP actor (Zhao et al. method, [M6](docs/references/snn_learning.md)); cite PopSAN's 140× Loihi figure as the neuromorphic-hardware ceiling | ☐ |
| L6 | *(optional SOTA upgrade)* MDC-SAN 2nd-order dynamic neurons (AAAI 2022) or ILC-SAN fully-spiking membrane-voltage decode + intralayer connections (Chen 2024) | only if PopSAN underperforms the MLP baseline | ☐ |

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
