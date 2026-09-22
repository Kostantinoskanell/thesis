# M4b-1 — dropout-severity sweep, all six controllers

5 seeds x 10 eps per point, 95% CI (t) across seeds.

| controller | drop=0.1 | drop=0.15 | drop=0.2 | drop=0.3 |
|---|---|---|---|---|
| Frozen MLP | 50.0%±19.6% | 34.0%±14.2% | 22.0%±13.6% | 22.0%±20.4% |
| Online MLP | 44.0%±14.2% | 32.0%±25.4% | 16.0%±20.8% | 30.0%±12.4% |
| Frozen SNN | 42.0%±16.2% | 30.0%±15.2% | 20.0%±19.6% | 12.0%±10.4% |
| Pure-STDP SNN | 10.0%±12.4% | 8.0%±10.4% | 4.0%±6.8% | 2.0%±5.6% |
| R-STDP SNN | 48.0%±23.9% | 24.0%±14.2% | 12.0%±13.6% | 12.0%±13.6% |
| TM-NORM SNN | 2.0%±5.6% | 0.0%±0.0% | 2.0%±5.6% | 2.0%±5.6% |

## R-STDP vs frozen SNN at each severity (Welch t-test, per-seed)

| severity | frozen SNN | R-STDP | delta | p |
|---|---|---|---|---|
| 0.1 | 42.0% | 48.0% | +6.0 pts | 0.582 |
| 0.15 | 30.0% | 24.0% | -6.0 pts | 0.446 |
| 0.2 | 20.0% | 12.0% | -8.0 pts | 0.383 |
| 0.3 | 12.0% | 12.0% | -0.0 pts | 1.000 |

**Any severity where R-STDP significantly beats frozen SNN? NO**