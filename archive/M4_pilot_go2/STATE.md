# M4 pilot — IN PROGRESS (inconclusive), resume here

_Paused 2026-07-18. M4 is NOT done — the go/no-go is still open._

## Where we are
The pilot machinery works and R-STDP is wired end-to-end (readout-layer plasticity via
the `STDPLearner` golden reference, driven by `SNNController.learn` / `SNNNavController`).
Three third-factor modes exist and are selectable: `raw` | `rpe` | `td` (default `td`).
A neuromodulatory `gate_threshold` and a linear-critic TD-error path are implemented.

## Results so far (both inconclusive — do NOT quote as a verdict)
- **Guardrail (no-shift, base dist):** raw reward *destroys* the policy at every eta
  (55%->15%); RPE preserves better only at very low eta. Root cause diagnosed: the env
  reward is tiny + almost always positive (~+0.004/step), so raw/EMA-baseline gives
  R-STDP no usable signal. -> motivated the TD-error third factor (D8).
- **TD-error diagnostic:** delta is now properly zero-mean (std 0.039, collision spikes
  ~-1), critic stable. The signal is healthy.
- **Compare run 1 (mid-episode shift @30s):** shift usually fired AFTER the goal was
  reached (goals reachable in ~20-28s) -> not a real recovery test. R-STDP drifted on the
  effectively-base distribution (30% vs frozen 40%).
- **Compare run 2 (persistent dense 16+6, TD-error):** everyone floored (R-STDP 17%,
  frozen 20%, online-MLP 7%) -> no headroom to show recovery. Inconclusive.

**Net:** no clean recovery test yet — one regime too easy (shift missed), the other too
hard (floor). Readout-only R-STDP shows no benefit so far.

## Decision being held (user: "continue tomorrow")
Pick the next step (see the AskUserQuestion options):
1. **Moderate regime first (recommended, cheap):** re-run compare at a shift where the
   pretrained policy keeps ~30-40% competence (real headroom), still readout-only. First
   clean recovery test. Likely config: ~11-12 static + 4 dynamic, or shift fired early
   (~t=5s) with a farther goal. Tune so frozen-SNN lands ~30-40%, not floor/ceiling.
2. **Hidden-layer plasticity + moderate regime:** implement R-STDP beyond the readout
   (proposal's named next lever; readout-only can't re-represent a harder distribution),
   for the strongest GO attempt. Bigger change to `SNNController` (currently readout-only).
3. **Accept inconclusive/NO-GO, document + move on** (valid finding per proposal).

## Key code (all committed as M4 WIP)
- `src/nmc/controllers/snn.py` — `SNNController` reward_mode raw/rpe/td, gate_threshold,
  linear critic; `SNNNavController` passes obs/next_obs through; spike-stats + gate diag.
- `scripts/pilot_m4.py` — `--mode guardrail|compare`, `--reward-mode`, `--eta`, `--gate`;
  compare uses the dense regime + retention (stability) test.
- Collision-view render: `go2_rl_walker.render(show_collision=True)` +
  `render_go2_nav.py --show-collision` -> `nav_episode_collision.gif`.

## First commands tomorrow (option 1)
```
# find a moderate regime: sweep density so frozen-SNN ~ 30-40%
conda run -n nmc python scripts/pilot_m4.py --mode compare --reward-mode td --eta 0.05 \
    --episodes 30   # after editing the compare env density down from 16+6
```
Decide density first (edit `compare()` env config), confirm frozen ~30-40%, THEN judge
R-STDP vs frozen with headroom.
