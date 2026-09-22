# M6 / H4 — graceful degradation under sensor noise

Base distribution (no dropout shift), additive Gaussian LiDAR noise, 5 seeds x 10 eps per point. CI = 95% (t, across seeds).

| controller | σ=0.3 | σ=0.5 | σ=0.8 | slope (pts per 0.1σ) | N50 |
|---|---|---|---|---|---|
| Frozen MLP | 42.0%±16.2% | 34.0%±18.8% | 38.0%±23.9% | -0.6 | never |
| Online MLP | 40.0%±24.8% | 36.0%±38.9% | 10.0%±15.2% | -6.2 | 0.685 |
| Frozen SNN | 28.0%±16.2% | 18.0%±18.4% | 18.0%±18.4% | -1.8 | never |
| Pure-STDP SNN | 16.0%±18.8% | 8.0%±10.4% | 6.0%±6.8% | -1.9 | 0.500 |
| R-STDP SNN | 28.0%±13.6% | 36.0%±22.6% | 34.0%±14.2% | 1.1 | never |
| TM-NORM SNN | 14.0%±11.1% | 4.0%±6.8% | 0.0%±0.0% | -2.7 | 0.440 |