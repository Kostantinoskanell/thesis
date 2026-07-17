# D1 — Unitree Go2 stands in MuJoCo under full dynamics

_Archived 2026-07-17 · engine pivot: MuJoCo + real Go2, dynamics everywhere_

- Official MuJoCo Menagerie `unitree_go2` model, full rigid-body contact dynamics.
- The Go2 uses **torque actuators**, so standing needs an active **PD controller**
  (naive pose-holding collapses — see debug-log). Walking needs the pretrained RL
  policy (D2, next).
- Verified: base height holds 0.255–0.270 m over 6 s — stable stance, no collapse.
- `stand.png` — photoreal render of the real Go2 standing.
- `stand.gif` — 6 s of stable stance.
- `fig_stand_stability.png` — base height vs time (stability proof).

This is the foundation for the dynamics migration: D2 = walk via RL velocity policy,
D3 = MuJoCo navigation env, then the plasticity science (M2–M8) runs on the dynamic Go2.
