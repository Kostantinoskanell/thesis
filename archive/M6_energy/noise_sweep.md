# M6 / H4 — graceful degradation under sensor noise

Base distribution (no dropout shift), additive Gaussian LiDAR noise, 5 seeds x 10 eps per point. CI = 95% (t, across seeds).

| controller | σ=0.0 | σ=0.025 | σ=0.05 | σ=0.1 | σ=0.2 | slope (pts per 0.1σ) | N50 |
|---|---|---|---|---|---|---|---|
| Frozen MLP | 42.0%±10.4% | 42.0%±10.4% | 50.0%±19.6% | 52.0%±10.4% | 34.0%±14.2% | -3.8 | never |
| Online MLP | 48.0%±16.2% | 42.0%±10.4% | 48.0%±5.6% | 42.0%±10.4% | 36.0%±28.6% | -5.4 | never |
| Frozen SNN | 40.0%±23.2% | 52.0%±25.4% | 42.0%±26.9% | 50.0%±29.1% | 40.0%±12.4% | -1.6 | never |
| Pure-STDP SNN | 14.0%±6.8% | 16.0%±11.1% | 16.0%±22.6% | 8.0%±10.4% | 4.0%±11.1% | -6.2 | 0.125 |
| R-STDP SNN | 42.0%±10.4% | 44.0%±20.8% | 54.0%±14.2% | 48.0%±10.4% | 30.0%±8.8% | -7.0 | never |
| TM-NORM SNN | 2.0%±5.6% | 16.0%±14.2% | 20.0%±12.4% | 20.0%±17.6% | 14.0%±18.8% | 3.2 | never |