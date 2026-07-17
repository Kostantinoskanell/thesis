# Debug Log

One dated entry per serious problem: **symptom → cause → fix → lesson**. These are the
"war stories" — invaluable for the thesis's methodology/limitations section and for not
re-solving the same problem twice.

| Date | Issue | File |
|------|-------|------|
| 2026-07-17 | PyBullet won't install on Python 3.13 (Windows) | [pybullet-python313](2026-07-17_pybullet-python313.md) |
| 2026-07-17 | M1 smoke test infinite loop (episode-clock reset) | [smoke-test-infinite-loop](2026-07-17_smoke-test-infinite-loop.md) |
| 2026-07-17 | Navigation expert: 5 failed attempts → A* + defensive braking | [expert-tuning-collisions](2026-07-17_expert-tuning-collisions.md) |
| 2026-07-17 | `conda run python -c` fails on multi-line scripts | [conda-run-inline-newlines](2026-07-17_conda-run-inline-newlines.md) |
| 2026-07-17 | Quadruped CPG trot collapses (metric said PASS, GIF said fell) | [quadruped-gait-collapse](2026-07-17_quadruped-gait-collapse.md) |
| 2026-07-17 | MuJoCo Go2 collapses holding home pose (torque vs position actuators) | [go2-torque-actuators](2026-07-17_go2-torque-actuators.md) |

Naming convention: `YYYY-MM-DD_short-slug.md`.
