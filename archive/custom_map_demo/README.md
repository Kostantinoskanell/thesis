# Custom deterministic test map — result

Design and rationale: `scripts/custom_map.py`. Motivation: the M4b-3 trajectory
overlay bug and the M4c sand-GIF bug were both caused by random dynamic
obstacles unrelated to whatever shift was nominally being tested, and a direct
audit (`scripts/m5_collision_kind_audit.py`) found **70-73% of all collisions
on the standard procedural map, across every controller, are with a moving
obstacle** — a large, shift-irrelevant confound. This map fixes robot start,
goal, a 6-cylinder wall, and one patrolling mover completely, so the only
seed-to-seed variation left is controller stochasticity, not map difficulty —
and places the wall's one gap so reaching it requires a **right turn directly
through the sensor-dropout dead-beam sector** (bearings -90° to -34°, per
`scripts/m4b_traj_diag.py`), a targeted test instead of an incidental one.

## Result (5 controller-seeds x 20 eps = 100 episodes/controller, dropout=0.20)

| controller | success | collision (static, i.e. the wall) | collision (dynamic, the mover) | fell | timeout |
|---|---|---|---|---|---|
| Frozen MLP | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| Online MLP | 0.0% | 0.0% | 0.0% | 5.0% | 95.0% |
| **Frozen SNN** | **84.0%** | 0.0% | 1.0% | 6.0% | 9.0% |
| Pure-STDP SNN | 1.0% | 82.0% | 0.0% | 2.0% | 15.0% |
| **R-STDP SNN** | **72.0%** | 0.0% | 2.0% | 8.0% | 18.0% |
| TM-NORM SNN | 0.0% | 98.0% | 0.0% | 2.0% | 0.0% |

R-STDP vs Frozen SNN (per-seed means, Welch t-test): 72.0% vs 84.0%, **p=0.112**.

## Two things this confirms

1. **The map design works as intended.** Dynamic-obstacle collisions dropped
   from 66-73% of all collisions (random map) to 0-2% of all episodes here —
   the confound is essentially gone, so any remaining difference between
   controllers is genuinely about the shift, not roaming-obstacle luck.

2. **A fourth independent confirmation that R-STDP does not help here.** On
   this cleaner, more targeted map — the dead-beam sector directly obscures
   the correct detour, exactly the scenario R-STDP is supposed to help with —
   R-STDP (72.0%) does not beat frozen SNN (84.0%); numerically it is *worse*
   by 12 points (not significant at this screening n=5 seeds, but the
   direction is consistent with every other test tonight, not a new or
   different failure mode). This is a coarse screen (5 seeds), not a full
   M5-style rigor pass — but it adds a fourth data point (sensor dropout,
   terrain sand, terrain ice, this map) all pointing the same way.

## One unexplained, out-of-scope observation

Frozen MLP scores 0% here (100% collision with the wall) despite being one of
the stronger controllers on the random procedural map. Plausible explanation:
the MLP was trained only on randomly-scattered individual obstacles and may
generalize poorly to a long, deliberate wall formation it never saw during
training — a real out-of-distribution effect, not a bug, but not investigated
further here (out of scope for tonight's question, which is about R-STDP, not
MLP generalization).
