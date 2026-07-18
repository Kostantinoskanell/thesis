# M4 pilot — the go/no-go gate: R-STDP recovery under distribution shift

_2026-07-18 · outcome: **GO signal** (positive, mechanism understood; multi-seed
confirmation deferred to M5). This was the pivotal experiment; it took real iteration._

## The question
Does releasing the M3-pretrained SNN to online R-STDP let it **recover** after a
mid-mission distribution shift, better than a frozen policy — without forgetting the
base task?

## The journey (each step diagnosed, not guessed — see STATE.md history + debug notes)
1. **Shift type matters most.** Obstacle-density doubling (our first shift) does *not*
   require re-learning — the correct policy is unchanged, just stressed — so no
   controller could show "recovery" (everyone floored). Switched to **sensor dropout**
   (a block of LiDAR beams fail → the pretrained input→action mapping is genuinely
   *wrong*): frozen SNN drops 44%→~20% with real headroom. (Terrain ice/sand also built;
   ice too harsh, sand too mild for the friction-randomized walker — kept for later.)
2. **Third factor matters.** Raw reward is tiny + always-positive → R-STDP just drifts
   (destroys the policy). Fixed with **RPE** then a **TD-error via a linear critic**
   (dopamine = TD error) — a proper zero-mean learning signal. (D8.)
3. **Plasticity SITE matters (the key fix).** Readout-only R-STDP *cannot* re-map a
   corrupted **input** — it stalled at 7% while the full-backprop online-MLP recovered
   early (evidence the input must adapt). Extending R-STDP to the **input layer** +
   **weight anchoring** (elastic pull to pretrained, to stop over-adaptation collapse)
   moved R-STDP **7% → 30%**. (D9.)

## Result — sensor dropout (30% of beams fail), 4 controllers × 30 eps
| controller | shifted success | base retention | over the block |
|---|---|---|---|
| **R-STDP SNN** (input+readout, TD, anchor) | **30%** [17,48] | **43%** (highest) | recovers, mild late drift |
| frozen MLP | 30% [17,48] | 37% | — (no adaptation) |
| online-MLP (TD-λ) | 23% [12,41] | **7%** (forgot base) | 0.8 peak → **collapse to 0** |
| frozen SNN | 17% [7,34] | 30% | — (no adaptation) |

**Headline:** R-STDP is the **only** controller that both *adapts* (30%, tied-best) **and**
*retains* the base task (43%, highest). The online-MLP recovers transiently then suffers
**catastrophic forgetting**; R-STDP's anchoring prevents that. This is the stability-
plasticity dilemma resolved in R-STDP's favor — the thesis's H1 story, on a clean shift.

## Honest limitations (→ M5 confirms)
- **CIs overlap at n=30** (R-STDP [17,48] vs frozen SNN [7,34]): a positive *signal*, not
  a proven effect. Point estimates and the adapt+retain combination consistently favor
  R-STDP. **M5 runs ≥10 seeds on the GPU env to establish significance.**
- R-STDP still drifts down in the final third (anchor helped, didn't fully stabilize);
  M5 can tune anchor/η, or add early-stopping.
- Single seed-model, one shift instance (fixed dead-beam block).

## Reproduce
```
# headroom (which shift gives room to recover):  scratch headroom.py -> sensor 0.30 ~20%
python scripts/pilot_m4.py --mode compare --shift-type sensor --dropout 0.30 \
    --reward-mode td --eta 0.05 --plastic-layers input+readout --anchor 0.005 --episodes 30
```
Figure: `fig_pilot_sensor.png` (recovery curves + adapt-vs-retain). The obstacle/dense
runs (`fig_pilot_obstacles.png`, earlier `fig_pilot_recovery.png`) are the negative
controls that motivated the sensor shift.
