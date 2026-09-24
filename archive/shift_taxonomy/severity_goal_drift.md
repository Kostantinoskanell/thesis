# Shift taxonomy: severity screen for `goal_drift`

5 seeds x 15 eps per point, 95% CI (t) across seeds.

| controller | sev=0.15 | sev=0.3 | sev=0.6 | sev=0.9 | sev=1.2 |
|---|---|---|---|---|---|
| Frozen MLP | 37.3%+/-20.8% | 37.3%+/-12.6% | 45.3%+/-12.3% | 38.7%+/-13.6% | 24.0%+/-9.4% |
| Online MLP | 42.7%+/-16.1% | 48.0%+/-12.3% | 40.0%+/-10.1% | 29.3%+/-12.6% | 30.7%+/-20.8% |
| Frozen SNN | 38.7%+/-20.6% | 38.7%+/-17.0% | 41.3%+/-6.9% | 30.7%+/-12.6% | 32.0%+/-13.6% |
| Pure-STDP SNN | 13.3%+/-8.3% | 14.7%+/-6.9% | 12.0%+/-6.9% | 8.0%+/-9.1% | 13.3%+/-5.9% |
| R-STDP SNN | 45.3%+/-17.0% | 44.0%+/-15.0% | 44.0%+/-7.4% | 33.3%+/-16.6% | 26.7%+/-16.6% |
| TM-NORM SNN | 2.7%+/-4.5% | 0.0%+/-0.0% | 2.7%+/-4.5% | 4.0%+/-4.5% | 6.7%+/-5.9% |

**Auto-picked severity for the rigor stage: 0.9** (largest drop from mildest tested that keeps frozen SNN > 5%)