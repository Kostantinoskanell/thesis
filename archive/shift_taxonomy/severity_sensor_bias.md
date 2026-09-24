# Shift taxonomy: severity screen for `sensor_bias`

5 seeds x 15 eps per point, 95% CI (t) across seeds.

| controller | sev=0.5 | sev=1.0 | sev=1.5 | sev=2.0 | sev=3.0 |
|---|---|---|---|---|---|
| Frozen MLP | 29.3%+/-19.9% | 25.3%+/-13.6% | 13.3%+/-8.3% | 10.7%+/-12.6% | 2.7%+/-7.4% |
| Online MLP | 29.3%+/-17.2% | 24.0%+/-17.2% | 12.0%+/-13.6% | 9.3%+/-13.8% | 4.0%+/-4.5% |
| Frozen SNN | 29.3%+/-19.1% | 25.3%+/-24.4% | 21.3%+/-15.9% | 17.3%+/-19.1% | 12.0%+/-10.8% |
| Pure-STDP SNN | 14.7%+/-9.1% | 5.3%+/-3.7% | 4.0%+/-7.4% | 1.3%+/-3.7% | 1.3%+/-3.7% |
| R-STDP SNN | 33.3%+/-19.4% | 18.7%+/-23.0% | 17.3%+/-19.1% | 12.0%+/-14.8% | 18.7%+/-12.3% |
| TM-NORM SNN | 16.0%+/-16.1% | 16.0%+/-16.1% | 12.0%+/-12.3% | 9.3%+/-9.4% | 8.0%+/-6.9% |

**Auto-picked severity for the rigor stage: 3.0** (largest drop from mildest tested that keeps frozen SNN > 5%)