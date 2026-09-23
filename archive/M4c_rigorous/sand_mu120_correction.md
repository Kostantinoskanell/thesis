# M4c sand correction: re-run at the ACTUAL claimed mu=1.20

The first m4c_rigorous.py pass used mu=1.6 (render_terrain_videos.py's bare CLI default) -- but that is the value M4c's OWN screening rejected as "unusable" (sand *helps* success to 55%). The real headline claim ('closes the gap to the MLP') was measured at the RETUNED mu=1.20. This corrects that.

| controller | success (mean+/-SD) | M4c's original claim (n=15, mu=1.20) |
|---|---|---|
| Frozen MLP | 55.0%+/-10.2% | 47% |
| Frozen SNN | 42.7%+/-8.7% | 27% |
| R-STDP SNN | 40.3%+/-11.1% | 47% |

**R-STDP vs the others (Holm-corrected):**

| comparison | delta | p (Holm) | significant? |
|---|---|---|---|
| R-STDP vs Frozen SNN | -2.3 pts | 0.626 | no |
| R-STDP vs Frozen MLP | -14.7 pts | 0.019 | YES |