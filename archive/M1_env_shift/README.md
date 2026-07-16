# M1 — PyBullet nav env with mid-episode distribution shift

_Archived 2026-07-17_

- Differential-drive (kinematic unicycle) robot, 32-beam 360° LiDAR, 10x10 m arena.
- Smoke test PASS: 60 s episode, shift fires at exactly t=30 s, static 8->16, dynamic 3->6.
- `fig_arena_shift.png` — top-down view before (Phase 1) and after (Phase 2) the shift, with LiDAR beams; the obstacle-density doubling is the adaptivity test.
