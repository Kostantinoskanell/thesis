# Shift taxonomy: severity screen for `sensor_range`

5 seeds x 15 eps per point, 95% CI (t) across seeds.

| controller | sev=1.5 | sev=2.0 | sev=3.0 | sev=4.0 | sev=6.0 |
|---|---|---|---|---|---|
| Frozen MLP | 60.0%+/-15.5% | 36.0%+/-11.1% | 34.7%+/-17.0% | 41.3%+/-23.7% | 45.3%+/-19.8% |
| Online MLP | 52.0%+/-15.9% | 33.3%+/-11.7% | 36.0%+/-17.2% | 30.7%+/-21.6% | 45.3%+/-15.9% |
| Frozen SNN | 28.0%+/-23.7% | 26.7%+/-28.7% | 33.3%+/-25.5% | 42.7%+/-25.2% | 46.7%+/-21.1% |
| Pure-STDP SNN | 6.7%+/-0.0% | 6.7%+/-8.3% | 13.3%+/-11.7% | 13.3%+/-15.5% | 10.7%+/-12.6% |
| R-STDP SNN | 26.7%+/-13.1% | 34.7%+/-25.1% | 36.0%+/-26.6% | 41.3%+/-6.9% | 42.7%+/-9.4% |
| TM-NORM SNN | 0.0%+/-0.0% | 0.0%+/-0.0% | 0.0%+/-0.0% | 0.0%+/-0.0% | 4.0%+/-4.5% |

**Auto-picked severity for the rigor stage: 2.0** (largest drop from mildest tested that keeps frozen SNN > 5%)