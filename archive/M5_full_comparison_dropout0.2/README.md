# M5 Full Comparison Results

| Controller | Success Rate (Mean ± SD) | Mean Recovery Time (Eps) | p-value (Success vs R-STDP) | p-value (Recovery vs R-STDP) |
|---|---|---|---|---|
| Frozen MLP | 34.0% ± 9.5% | 6.4 | 3.642e-03 * | 3.716e-01 |
| Online MLP | 24.0% ± 13.4% | 6.3 | 7.024e-01 | 3.716e-01 |
| Frozen SNN | 21.3% ± 7.2% | 7.7 | 7.024e-01 | 3.716e-01 |
| Pure-STDP SNN | 8.7% ± 4.5% | 20.9 | 9.965e-04 * | 1.731e-01 |
| R-STDP SNN | 19.3% ± 5.3% | 11.8 | - | - |
| TM-NORM SNN | 1.7% ± 2.2% | 30.0 | 4.361e-06 * | 3.377e-05 * |
