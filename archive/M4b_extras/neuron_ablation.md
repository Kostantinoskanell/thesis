# M4b-7 — neuron model alone vs plasticity

dropout=0.20, 10 seeds x 10 eps. Isolates what the ALIF adaptive threshold buys from what R-STDP buys.

| variant | success | 95% CI |
|---|---|---|
| frozen LIF | 7.0% | ±5.9% |
| frozen ALIF | 17.0% | ±9.0% |
| R-STDP ALIF | 12.0% | ±5.6% |

- ALIF vs LIF (neuron model alone): +10.0 pts, p=0.051
- R-STDP-ALIF vs frozen ALIF (plasticity alone): -5.0 pts, p=0.302