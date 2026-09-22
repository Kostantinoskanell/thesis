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

## Open question this raises

M4c's original terrain numbers (sand 47%/47%, ice 40% vs 27%/33%) were computed
at n=15, one seed — the same kind of thin sample that turned out not to
replicate for the sensor-dropout shift (M5). Given episode 0 alone failed for
sand, and 4/5 of a small follow-up sample also failed, it is worth confirming
the aggregate terrain numbers hold up at real multi-seed rigor before treating
them as the thesis's settled positive result. See `scripts/m4c_rigorous.py` /
`archive/M4c_rigorous/README.md` for that check.
