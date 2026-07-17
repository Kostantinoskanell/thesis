# D2 (interim) — Go2 walks in MuJoCo via CPG, full dynamics

_Archived 2026-07-17 · INTERIM locomotion layer — see decision note below_

- CPG trot joint targets tracked by the Go2's PD controller (torque actuators),
  driven by the `[vx, omega]` interface, full rigid-body dynamics.
- **Works but low quality:** the Go2 stays upright (height 0.24–0.29 m, no fall)
  and moves, BUT `fig_walk_tracking.png` shows heavy **yaw wobble** (actual ω swings
  ±1.5 rad/s even when commanded straight) and speed under-tracks (~0.2 vs 0.6 m/s).
  It veers rather than walking cleanly.
- `walk.gif` — the dynamic Go2 trotting (visibly wobbly).
- `fig_walk_tracking.png` — stability (good) + command tracking (poor).

## Why this is only interim
The CPG is not a good Go2 locomotion controller. A proper controller is needed for
clean velocity tracking. Options under decision (ROADMAP D2):
- **Convex MPC in MuJoCo** (Windows-native, model-based, robust) — e.g. go2-convex-mpc.
- **RL policy** (the SOTA pick) — requires a Linux/GPU training pipeline (WSL2 +
  Isaac/Playground), then deploy the exported policy in our MuJoCo loop.
The plug-and-play pretrained RL route is unavailable on Windows/classic-MuJoCo (see
docs/debug-log/2026-07-17_go2-locomotion-no-pretrained.md).
