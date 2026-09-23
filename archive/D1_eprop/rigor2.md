# e-prop rigor test (CORRECTED): does e-prop recover where R-STDP did not?

eta=0.005, anchor=0.005, plastic_layers=(0,-1) -- identical protocol and seeds (6000-6009, seed*100+ep resets) for all three controllers, run together in this pass so the comparison is a real paired Welch t-test on matched raw data, not an approximation against old summary statistics. Holm-Bonferroni corrected across all 4 comparisons below.


## dropout = 0.3

| controller | success (mean+/-SD) |
|---|---|
| Frozen SNN | 11.0%+/-2.7% |
| R-STDP SNN | 8.7%+/-3.9% |
| **e-prop SNN** | 11.3%+/-7.9% |

## dropout = 0.2

| controller | success (mean+/-SD) |
|---|---|
| Frozen SNN | 21.3%+/-7.6% |
| R-STDP SNN | 19.3%+/-5.6% |
| **e-prop SNN** | 28.7%+/-9.7% |

## Holm-Bonferroni corrected comparisons (e-prop vs baseline)

| dropout | vs | delta | raw p | Holm p | significant? |
|---|---|---|---|---|---|
| 0.3 | Frozen SNN | +0.3% | 0.902 | 0.902 | no |
| 0.3 | R-STDP SNN | +2.7% | 0.355 | 0.711 | no |
| 0.2 | Frozen SNN | +7.3% | 0.077 | 0.231 | no |
| 0.2 | R-STDP SNN | +9.3% | 0.019 | 0.078 | no |


**Verdict: e-prop does NOT show a Holm-corrected significant advantage over frozen SNN -- the null result generalizes beyond R-STDP specifically**