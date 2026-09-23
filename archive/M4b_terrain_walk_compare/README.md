# M4c terrain walk comparison — GIF audit and fix (2026-09-22)

Full numeric write-up lives in the M4c row of [`ROADMAP.md`](../../ROADMAP.md). This
file documents a bug found and fixed during a visual audit of the GIFs in this folder.

## Bug found: the "R-STDP recovers on sand" GIF actually showed a failure

`scripts/render_terrain_videos.py` records **episode 0 unconditionally**
(`record_frames=(ep == 0)`) with no check on whether that episode actually
succeeded — unlike `eval_mlp_go2.py`'s own `eval_frozen()`, which only keeps a
GIF `if reached`. `snn_rstdp_sand.gif`, captioned on the project site as
demonstrating R-STDP "closing the gap to the MLP" on sand, was actually episode
0 for that config — and episode 0 was a **collision with a dynamic (moving)
obstacle**, not a success. The GIF genuinely did stop mid-episode and loop back
abruptly, which is exactly why it looked "erratic" on inspection.

Diagnosed with `scripts/m4c_gif_audit.py` (reruns the exact seed/warm-up
`render_terrain_videos.py` used and reports the true outcome):

| GIF | terrain | reached | fell | collision | notes |
|---|---|---|---|---|---|
| `mlp_ice.gif` / `snn_frozen_ice.gif` | ice, mu=0.08 (default) | False | **True** | — | consistent with mu=0.08 being "too harsh" (M4c already knew this — see ROADMAP) |
| `snn_rstdp_ice.gif` | ice, mu=0.08 | False | **True** | — | same |
| `snn_rstdp_sand.gif` (OLD) | sand, mu=1.6 | **False** | False | **True (dynamic)** | **wrong — captioned as a recovery demo** |
| `snn_rstdp_ice_mu028_warm15.gif` | ice, mu=0.28 | **True** | False | False | genuine success, verified correct |

## Fix

`scripts/m4c_fix_sand_gif.py` reruns the same warm-up, then scans subsequent
episodes (same protocol) until it finds a **verified** success
(`reached and not fell and not collision`) and saves that instead. Took 6 tries:

```
seed=8000: collision (dynamic)
seed=8001: timeout
seed=8002: collision (dynamic)
seed=8003: collision (dynamic)
seed=8004: collision (dynamic)
seed=8005: REACHED — used for the new snn_rstdp_sand.gif
```

**4 of 5 failed attempts were dynamic-obstacle collisions** — independent
evidence (alongside the M4b-3 trajectory-overlay bug fix) that random moving
obstacles are a significant source of episode-outcome variance in this
environment, unrelated to the terrain/sensor shift actually being studied. See
`archive/M4b_extras/README.md` for the related finding on the sensor-dropout
side, and `scripts/m5_collision_kind_audit.py` for a direct measurement of how
much of the overall failure rate this accounts for.

`snn_rstdp_ice_mu028_warm15.gif` (the other terrain GIF featured on the site)
was independently verified genuine — no fix needed there.

## Answered: the terrain result does NOT survive rigor either (2026-09-23)

M4c's original terrain numbers (sand 47%/47%, ice 40% vs 27%/33%) were computed
at n=15, one seed — the same kind of thin sample that turned out not to
replicate for the sensor-dropout shift (M5). `scripts/m4c_rigorous.py` reran
both terrain conditions at M5's exact standard (10 seeds x 30 eps, Holm-corrected,
R-STDP's own 15-episode warm-up protocol preserved):

| Terrain | Frozen MLP | Frozen SNN | R-STDP SNN | R-STDP vs Frozen SNN |
|---|---|---|---|---|
| Sand (mu=1.6) | 53.3%+/-10.9% | 42.3%+/-11.7% | 46.7%+/-10.5% | +4.3 pts, **p=0.421 (not significant)** |
| Ice (mu=0.28) | 44.7%+/-10.7% | 29.0%+/-10.8% | 34.0%+/-13.3% | +5.0 pts, **p=0.393 (not significant)** |

**Both of M4c's headline claims ("sand: fully closes the gap to the MLP",
"ice: beats both") do not replicate.** R-STDP is numerically a few points above
frozen SNN on both terrains, but nowhere close to significant at n=10 seeds --
the same magnitude of effect (and the same fate) as M4's original sensor-dropout
"win". Frozen SNN's own number moved substantially too (27%->42.3% sand,
27%->29.0% ice), confirming M4c's n=15 single-ish sample was noisy on BOTH
sides, not just R-STDP's.

**Correction, resolved:** the sand row above used mu=1.6 -- the severity M4c's own screening rejected as unusable, not the retuned mu=1.20 the actual headline claim was measured at. Re-run at the correct mu=1.20 (`archive/M4c_rigorous/sand_mu120_correction.md`):

| controller | success (mean+/-SD) |
|---|---|
| Frozen MLP | 55.0%+/-10.2% |
| Frozen SNN | 42.7%+/-8.7% |
| R-STDP SNN | 40.3%+/-11.1% |

R-STDP vs Frozen SNN: -2.3 pts, p=0.626 (not significant, numerically worse). R-STDP vs Frozen MLP: **-14.7 pts, p=0.019 -- SIGNIFICANTLY worse**, the exact opposite of "closes the gap to the MLP". The ice result (mu=0.28) was correct as originally tested.

**Thesis-level consequence: at full statistical rigor, R-STDP currently has NO
surviving significant recovery advantage on any shift type tested so far**
(sensor dropout at two severities: null (M5); terrain sand: null; terrain ice:
null). The remaining open questions are (a) the three shift types
implemented-but-never-tested before tonight (sensor_bias, sensor_range,
goal_drift -- `archive/shift_taxonomy/`), and (b) whether a structurally
different learning rule (e-prop -- `archive/D1_eprop/`) succeeds where R-STDP
did not. See [D18](../../docs/references/sota_decisions.md) for the full
writeup. Full data: `archive/M4c_rigorous/README.md`.
