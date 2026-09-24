# Shift taxonomy: rigor stage for `goal_drift` @ severity=0.9

10 seeds x 30 eps, Holm-corrected significance vs R-STDP SNN.

| controller | success (mean+/-SD) | recovery (eps) | p vs R-STDP (success) | p vs R-STDP (recovery) |
|---|---|---|---|---|
| Frozen MLP | 38.3%+/-9.8% | 0.9 | 1.000 | 0.293 |
| Online MLP | 34.7%+/-12.5% | 3.2 | 1.000 | 1.000 |
| Frozen SNN | 37.0%+/-10.4% | 3.2 | 1.000 | 1.000 |
| Pure-STDP SNN | 13.7%+/-5.7% | 12.3 | 0.000 * | 0.150 |
| R-STDP SNN | 38.3%+/-11.0% | 3.2 | - | - |
| TM-NORM SNN | 1.3%+/-1.6% | 30.0 | 0.000 * | 0.000 * |

**R-STDP significantly beats frozen SNN on `goal_drift`? NO**