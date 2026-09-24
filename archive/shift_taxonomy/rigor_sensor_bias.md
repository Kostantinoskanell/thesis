# Shift taxonomy: rigor stage for `sensor_bias` @ severity=3.0

10 seeds x 30 eps, Holm-corrected significance vs R-STDP SNN.

| controller | success (mean+/-SD) | recovery (eps) | p vs R-STDP (success) | p vs R-STDP (recovery) |
|---|---|---|---|---|
| Frozen MLP | 1.7%+/-2.2% | 30.0 | 0.000 * | 0.000 * |
| Online MLP | 6.0%+/-3.9% | 26.7 | 0.000 * | 0.000 * |
| Frozen SNN | 17.7%+/-5.2% | 14.6 | 0.175 | 0.034 * |
| Pure-STDP SNN | 4.0%+/-4.2% | 28.6 | 0.000 * | 0.000 * |
| R-STDP SNN | 21.3%+/-5.8% | 6.1 | - | - |
| TM-NORM SNN | 10.3%+/-5.3% | 23.1 | 0.001 * | 0.001 * |

**R-STDP significantly beats frozen SNN on `sensor_bias`? NO**