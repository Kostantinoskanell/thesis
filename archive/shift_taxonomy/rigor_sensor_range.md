# Shift taxonomy: rigor stage for `sensor_range` @ severity=2.0

10 seeds x 30 eps, Holm-corrected significance vs R-STDP SNN.

| controller | success (mean+/-SD) | recovery (eps) | p vs R-STDP (success) | p vs R-STDP (recovery) |
|---|---|---|---|---|
| Frozen MLP | 45.0%+/-10.1% | 1.3 | 0.001 * | 0.446 |
| Online MLP | 43.7%+/-10.0% | 2.3 | 0.002 * | 0.794 |
| Frozen SNN | 32.3%+/-10.5% | 3.9 | 0.124 | 0.970 |
| Pure-STDP SNN | 13.7%+/-5.7% | 19.9 | 0.009 * | 0.004 * |
| R-STDP SNN | 25.0%+/-8.6% | 4.0 | - | - |
| TM-NORM SNN | 0.0%+/-0.0% | 30.0 | 0.000 * | 0.000 * |

**R-STDP significantly beats frozen SNN on `sensor_range`? NO**