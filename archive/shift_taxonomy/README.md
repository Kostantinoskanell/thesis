# Shift taxonomy: the three previously-untested shift types

## Question

Three shift types were implemented in `Go2NavConfig` from early on but never
actually tested end-to-end: `sensor_bias` (LiDAR readings offset by a fixed
bias), `sensor_range` (LiDAR max range reduced), `goal_drift` (goal position
biased/rotated). Every prior R-STDP result (M5's sensor dropout, M4c's
terrain sand/ice, the custom map) tested only two shift *classes* — sensor
corruption and terrain/physics faults. This was the last open question from
that line of work: does "R-STDP doesn't help" hold across a genuinely wider
taxonomy of shifts, or were the tested shifts unrepresentative?

## Method

Same two-stage protocol as M5, applied fresh to each shift type:

1. **Severity screen** (5 seeds x 15 eps, 5 severity points): find a severity
   with real headroom (frozen SNN success > 5%, not floored) — the exact
   lesson M4/M5 already learned the hard way (a floored or trivial severity
   makes any comparison uninformative).
2. **Full rigor sweep** (10 seeds x 30 eps, Holm-corrected vs R-STDP) at the
   auto-picked severity, across all 6 controllers.

`scripts/shift_taxonomy.py`. This run also survived an actual machine reboot
mid-way (`sensor_range` had to be redone from scratch after the interrupted
first attempt lost no valid data, since `run_severity`/`run_rigor` only write
their output files at the end of each phase) — resumed via
`scripts/shift_taxonomy_resume.py`, which reuses the same functions for just
the two shift types that hadn't completed yet.

## Results

| shift type | severity (auto-picked) | Frozen SNN | R-STDP SNN | R-STDP vs Frozen SNN (Holm p) | significant? |
|---|---|---|---|---|---|
| `sensor_bias` | 3.0 (lidar_bias_m) | 17.7%+/-5.2% | 21.3%+/-5.8% | 0.175 | no |
| `sensor_range` | 2.0 (sensor_range_m) | 32.3%+/-10.5% | 25.0%+/-8.6% | 0.124 | no |
| `goal_drift` | 0.9 (goal_drift_rad) | 37.0%+/-10.4% | 38.3%+/-11.0% | 1.000 | no |

Full per-controller tables and severity screens: `severity_<shift>.md` /
`rigor_<shift>.md` for each of the three shift types.

## Cross-shift summary

| shift type | R-STDP beats frozen SNN? |
|---|---|
| sensor dropout (M5) | NO |
| terrain sand (M4c) | NO (significantly *worse* than MLP) |
| terrain ice (M4c) | NO |
| sensor_bias | NO |
| sensor_range | NO |
| goal_drift | NO |

**R-STDP shows no significant recovery advantage over a frozen, non-plastic
network on any of six shift types tested**, spanning three genuinely
different shift classes (sensor corruption, terrain/physics, goal
perturbation). This was the last remaining gap in the shift taxonomy: with
it closed, "R-STDP doesn't help here" is no longer a claim about one or two
shift types, or one plasticity rule (e-prop, D19, tested independently and
also failed to show a significant advantage) — it now holds across the full
breadth of shifts this project's environment supports. Combined with D17's
neuron-model ablation (frozen ALIF alone: +10 pts that no plasticity rule
tested has topped), the most defensible overall reading is that **the
neuron model, not online synaptic plasticity, is what provides shift
robustness in this setup** — a genuine, thoroughly-checked negative result
for online RL-adapted local learning rules at this network scale, not a
single-shift artifact or an implementation gap in one rule.
