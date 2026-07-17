# P1 — quadruped (A1) walks in PyBullet from velocity commands

_Archived 2026-07-17 · Platform track (decoupled from the plasticity science)_

- CPG (central-pattern-generator) trot gait on the built-in A1 URDF (Go2's
  predecessor). CPG is itself a neuromorphic locomotion primitive, so the
  low-level layer stays on-theme.
- Driven by the same `[vx, omega]` interface the SNN navigator emits.
- Verified: walks forward ~0.3 m/s and turns left (+108° for omega>0), **stays
  upright** (base height 0.27–0.31 m, never near the 0.2 m fall line).
- `walk.gif` — 3D render of the A1 trotting forward then turning.
- `fig_gait_diagnostics.png` — base height (stability) + speed/ω command tracking.

Known limitation (expected for open-loop CPG): actual speed ≈ 60% of commanded and
ω is noisy. Convex-MPC or a learned RL policy would tighten tracking — see
docs/references/locomotion.md. Next: P2 swaps in the Go2 URDF and drives the nav
env's obstacle-avoidance episode with the walking robot.
