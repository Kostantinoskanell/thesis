# M4b-2 — is the result mask-specific?

dropout=0.20, dead-beam block starting at four different positions (M4/M5 always used start=8). 5 seeds x 10 eps.

| dead-beam start | frozen SNN | R-STDP | delta | p |
|---|---|---|---|---|
| 0 | 32.0%±18.4% | 46.0%±18.8% | +14.0 pts | 0.178 |
| 8 (M4/M5) | 20.0%±19.6% | 12.0%±13.6% | -8.0 pts | 0.383 |
| 16 | 8.0%±10.4% | 6.0%±6.8% | -2.0 pts | 0.668 |
| 24 | 28.0%±22.2% | 32.0%±10.4% | +4.0 pts | 0.667 |

Pooled over masks: frozen 22.0%, R-STDP 24.0%. Across-mask spread (sd of per-mask means): frozen 9.2%, R-STDP 15.9%.