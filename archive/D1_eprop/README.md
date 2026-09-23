# D1: e-prop -- does a structurally different plasticity rule recover where R-STDP failed?

## Question

R-STDP never produced a significant recovery advantage across six independent
tests (sensor dropout at two severities, terrain sand, terrain ice, a custom
obstacle map, a neuron/plasticity-scope ablation -- see `M5_full_comparison`,
`M4b_extras`, `M4c_rigorous`, `custom_map_demo`). Is that null result about
**R-STDP specifically** (a purely Hebbian trace, one global reward scalar), or
about **the shift itself** being unrecoverable by any local online learning
rule at this network scale? e-prop (Bellec et al. 2020) is a structurally
different three-factor rule -- its eligibility trace is derived from the
neuron's own membrane/adaptation dynamics (an online BPTT approximation) and
its third factor is a per-neuron symmetric-feedback signal rather than one
global scalar. See `src/nmc/plasticity/eprop.py` for the implementation and
scope note, and [`docs/concepts.md`](https://github.com/Kostantinoskanell/thesis/blob/main/docs/concepts.md) for the plain-language explanation.

## Stage 1: eta guardrail (`guardrail.md`)

Swept eta in `[0.0005, ..., 0.2]` on the base (no-shift) distribution to find
the largest learning rate that doesn't destabilize the pretrained network.
Everything above eta=0.005 degraded (30% down to 2% vs. a 42% frozen
baseline). **Recommended: eta=0.005.**

## Stage 2: first rigor pass -- methodologically flawed, corrected (`rigor.md` -> `rigor2.md`)

The first rigor script (`eprop_rigor.py`) ran e-prop fresh (10 seeds x 30 eps,
both dropout severities) but compared it against **old M5 aggregate
(mean, SD) numbers** via a normal approximation, and never applied the
Holm-Bonferroni correction it imported despite testing 4 comparisons. That
run showed an enticing raw p=0.019 (e-prop vs R-STDP at dropout=0.20) --
exactly the kind of thin, uncorrected signal that turned out not to replicate
for the M4c terrain claim and the M5 pilot's hyperparameter "improvement".

**Correction (`eprop_rigor2.py`):** `m5_full_comparison.py`'s controller
factories and env-reset scheme (`seed*100+ep`, seeds 6000-6009) are directly
importable, so Frozen SNN and R-STDP SNN were re-run **fresh, in the same
pass, with the identical seeds** as e-prop -- giving genuinely matched raw
per-seed data (saved to `raw_success_rates.csv`, so this never has to be
re-derived from a summary approximation again) for a real paired Welch
t-test, Holm-corrected across all 4 comparisons.

## Result (10 seeds x 30 eps, matched raw data, Holm-corrected)

| dropout | controller | success (mean+/-SD) |
|---|---|---|
| 0.30 | Frozen SNN | 11.0%+/-2.7% |
| 0.30 | R-STDP SNN | 8.7%+/-3.9% |
| 0.30 | **e-prop SNN** | 11.3%+/-7.9% |
| 0.20 | Frozen SNN | 21.3%+/-7.6% |
| 0.20 | R-STDP SNN | 19.3%+/-5.6% |
| 0.20 | **e-prop SNN** | 28.7%+/-9.7% |

| dropout | e-prop vs | delta | raw p | **Holm p** | significant? |
|---|---|---|---|---|---|
| 0.3 | Frozen SNN | +0.3% | 0.902 | 0.902 | no |
| 0.3 | R-STDP SNN | +2.7% | 0.355 | 0.711 | no |
| 0.2 | Frozen SNN | +7.3% | 0.077 | 0.231 | no |
| 0.2 | R-STDP SNN | +9.3% | 0.019 | **0.078** | no |

**Verdict: e-prop does not show a Holm-corrected significant advantage over
frozen SNN.** The numerically largest gap (e-prop vs R-STDP at dropout=0.20,
+9.3 pts) looks promising raw but does not survive correction for the 4
comparisons tested.

## What this means for the thesis

This is the answer to the open D1 question, and it's a stronger, more
defensible finding than a shaky positive would have been: **the null result
generalizes beyond R-STDP.** Two structurally different three-factor
plasticity rules (a purely Hebbian trace with a global scalar, and a
neuron-dynamics-derived trace with per-neuron symmetric feedback) both fail
to produce a significant recovery advantage over a frozen, non-plastic
network on this shift. Combined with the M4b-7 neuron-model ablation
(frozen ALIF alone gains +10 pts that plasticity does not improve on), the
overall picture is that **the neuron model, not online synaptic plasticity,
is doing the shift-robustness work** in this setup -- a real, honestly
negative result about online RL-adapted local learning rules at this network
scale, not a implementation failure of any one rule.

Raw data: `raw_success_rates.csv` (controller, dropout, seed, success_rate --
60 rows, the matched data behind the table above). Full guardrail sweep:
`guardrail.md`. Superseded first-pass (kept for the audit trail of the
methodology fix): `rigor.md`.
