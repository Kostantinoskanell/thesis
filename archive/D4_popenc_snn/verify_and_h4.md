# D4 PopEnc-SNN: baseline parity + H4 re-test

## Baseline parity (base distribution, 3 seeds x 20 eps)

**Correction:** the first run of this check had a bug (`_parity_task` accepted
a `seed` argument but never used it -- `env.reset(seed=4000+ep)` was a
hardcoded constant, and this controller is fully deterministic, so all 3
"seeds" silently ran the identical 20 episodes and trivially agreed: reported
as 50.0% +/- 0.0%, which was actually one 20-episode run, not 60 independent
ones. Fixed in `scripts/popenc_verify_and_h4.py` (seed now feeds
`env.reset()`) and re-run standalone via `scripts/popenc_parity_fix.py`.

PopEnc-SNN (corrected): 46.7% +/- 7.2% (per-seed: 45.0%, 45.0%, 50.0%), mean firing rate 0.5%

Reference: original rate/TTFS SNN (M3) 41% [32,51] Wilson-95; MLP 37% [29,45].

**Parity gate (PASS):** reasonable enough to proceed to H4 -- the correction
did not change the gate outcome, only the (previously fake) precision.

## H4 re-test (base distribution, additive LiDAR noise, 3 seeds x 15 eps)

| controller | s=0.0 | s=0.1 | s=0.2 | s=0.3 | s=0.5 | s=0.8 | slope (pts/0.1s) |
|---|---|---|---|---|---|---|---|
| Frozen MLP | 46.7% | 53.3% | 35.6% | 42.2% | 33.3% | 33.3% | -2.1 |
| Frozen SNN | 37.8% | 42.2% | 40.0% | 26.7% | 20.0% | 13.3% | -3.8 |
| PopEnc SNN | 46.7% | 44.4% | 55.6% | 46.7% | 24.4% | 20.0% | -4.1 |