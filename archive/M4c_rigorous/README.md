# M4c stress-test: terrain result at M5 rigor (10 seeds x 30 eps, Holm-corrected)

Reproduces render_terrain_videos.py's protocol (R-STDP: 15-ep silent warm-up then 30-ep measured block) across 10 seeds instead of M4c's single-ish sample.


## sand (mu=1.6)

| controller | success (mean +/- SD) | M4c's original (n=15) |
|---|---|---|
| Frozen MLP | 53.3% +/- 10.9% | 47% |
| Frozen SNN | 42.3% +/- 11.7% | 27% |
| R-STDP SNN | 46.7% +/- 10.5% | 47% |

**R-STDP vs the others (Welch t-test, Holm-corrected):**

| comparison | delta | p (raw) | p (Holm) | significant? |
|---|---|---|---|---|
| R-STDP vs Frozen SNN | +4.3 pts | - | 0.421 | no |
| R-STDP vs Frozen MLP | -6.7 pts | - | 0.406 | no |

## ice_mu028 (mu=0.28)

| controller | success (mean +/- SD) | M4c's original (n=15) |
|---|---|---|
| Frozen MLP | 44.7% +/- 10.7% | 33% |
| Frozen SNN | 29.0% +/- 10.8% | 27% |
| R-STDP SNN | 34.0% +/- 13.3% | 40% |

**R-STDP vs the others (Welch t-test, Holm-corrected):**

| comparison | delta | p (raw) | p (Holm) | significant? |
|---|---|---|---|---|
| R-STDP vs Frozen SNN | +5.0 pts | - | 0.393 | no |
| R-STDP vs Frozen MLP | -10.7 pts | - | 0.156 | no |