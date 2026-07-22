# L4 R-STDP terrain recovery — does plasticity recover the spiking gait on ice? (No.)

_2026-07-22 · run on the robust DAgger walking policy (the first valid substrate)._

## Question
The L-track's whole motivation (M4c): nav-layer R-STDP can't fix slipping on ice because
it never touches the legs. So release the *locomotion* spiking policy itself to R-STDP and
test recovery on ice — the locomotion analogue of the nav-layer H1 recovery claim.

## Setup
Substrate: the DAgger-distilled spiking walker (walks robustly, tracks velocity). Shift:
low terrain friction (ice). Command: sustained forward vx=0.3. R-STDP:
`PopSpikingRSTDPController`, TD-error third factor, elastic anchor.

## Finding: R-STDP does NOT recover — it destabilizes the gait
First, the walker is quite ice-robust on its own (frozen, friction 0.06, 6 eps):
**0/6 falls** (all 1000 steps), mean return ~7.6, still tracks vx (0.30→0.26, slip). So ice
degrades performance modestly but doesn't break it.

Releasing to R-STDP (friction 0.06, continual over the episode block) made it **worse**,
consistently across settings:
| config | outcome |
|---|---|
| frozen (no plasticity) | 0/6 falls, mean return ~7.6 |
| R-STDP eta=0.01, readout-only, anchor 0.01 | falls (len 73, 852), returns negative |
| R-STDP eta=0.003, readout-only, anchor 0.02 (gentlest) | eps 1-2 hold, then **eps 3-6 all collapse (len=85)** — progressive destabilization |

Even the gentlest R-STDP (near-frozen eta + strong anchor) drifts the weights until the
gait collapses and stays collapsed. Input+readout plasticity would be worse; gentler eta
approaches "no adaptation" (= frozen). So this is not a tuning miss — it is the behavior.

## Interpretation — a real contrast between the two layers
R-STDP **works on the navigation layer** (M4: recovers a sensor-dropout shift 7%→30%) but
**fails on the locomotion layer**. Why the difference is the interesting part:
- **Navigation** = a coarse, *discrete* decision (4 actions, population-vote) — robust to
  weight perturbation, and the shift (sensor dropout) needs a clean input→action *remap*
  that Hebbian correlation can find.
- **Locomotion** = *continuous*, precise, delicately-tuned joint control — small Hebbian
  weight changes disrupt the finely-balanced gait faster than they usefully adapt it, and
  the ice shift needs subtle *motor recalibration* that correlation-based updates can't
  cleanly produce; the noise accumulates into collapse.

This delineates **where reward-modulated plasticity helps vs doesn't** on the same robot
with the same rule — arguably a more honest and nuanced contribution than "R-STDP works
everywhere." It also motivates the D1 upgrade path (e-prop, a per-neuron learning signal)
for fine motor control, where a global-scalar R-STDP is too crude.

## Honest status
Negative result, cleanly reproduced (2 eta, +anchor). Not a bug — the pipeline is unit-
tested and the nav-layer R-STDP works with the same machinery. Verified by episode-length
collapse (falls at len=85), not reward alone.
