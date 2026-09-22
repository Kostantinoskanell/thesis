# M5 — full comparison, ≥10 seeds w/ CIs (nav layer, sensor-dropout shift)

_2026-09-05 · outcome: **honest negative for H1 on this shift** — R-STDP does not
significantly beat a frozen SNN, confirmed across two shift severities, a
hyperparameter re-sweep, and a plasticity-scope/third-factor ablation. R-STDP
*does* reliably beat plain Hebbian STDP everywhere it was tested. Contrast with
M4c ([ROADMAP.md](../../ROADMAP.md) row, `archive/M4b_terrain_walk_compare/`),
which found the opposite result on the terrain/sand shift — H1 is
shift-dependent, not universally false._

## The question
M4's pilot (single model-seed, single env-seed block) reported R-STDP recovering
to 30% success under 30%-sensor-dropout, clearly beating frozen SNN's 17% — but
flagged this explicitly as unconfirmed ("CIs overlap at n=30... a positive
*signal*, not a proven effect"). M5 exists to find out if that signal survives
proper multi-seed testing.

## What we found, in the order we found it

### 1. The initial M5 run (built by an autonomous agent session) had two bugs
- `scripts/m5_full_comparison.py` built the shift with `sensor_dropout_start`
  unset, defaulting to `-1` = a **new random dead-beam location every episode**.
  R-STDP's whole mechanism (M4's D9) depends on a *consistent* corrupted
  input→action mapping to learn; randomizing it every episode defeats the
  premise. Fixed: pin `sensor_dropout_start=8`, matching `pilot_m4.py`'s default.
- `src/nmc/controllers/tmnorm.py` (the TM-NORM reward-free baseline, D10) had an
  unbounded feedback loop: its threshold recalibration could push `v_th` negative,
  which flips the sign of the ALIF reset (`mem -= spk*thr`) from subtractive to
  *additive* — a runaway divergence, confirmed empirically (threshold drifting to
  -5 within one episode, 0.0% success with **zero variance** across all 10 seeds,
  a statistically implausible result for a real method). Fixed: floor `v_th` at
  a small positive value so reset physics stays sane.
- (Also: `ProcessPoolExecutor()` with no `max_workers` spawned one process per
  CPU core (20), each loading MuJoCo+PyTorch — exhausted the page file. Capped
  at 4.)

### 2. Fixed-config result (dropout=0.30, 10 seeds × 30 eps)
| Controller | Success (mean±SD) | Recovery (eps) | p vs R-STDP |
|---|---|---|---|
| Frozen MLP | 30.3%±7.7% | 4.2 | 1.9e-05 * |
| Online MLP | 21.0%±12.5% | 9.2 | 0.050 |
| Frozen SNN | 11.0%±2.6% | 20.3 | 0.14 |
| Pure-STDP SNN | 5.0%±3.4% | 28.9 | 0.086 |
| **R-STDP SNN** | **8.7%±3.7%** | 18.4 | – |
| TM-NORM SNN | 1.3%±2.2% | 30.0 | 5.7e-04 * |

Sanity check the fixes worked: Frozen MLP (30.3%) and Frozen SNN (11.0%) both
land within M4's original CIs (30% and 17% resp.) — the shift condition
replicates correctly. **R-STDP (8.7%) does not beat frozen SNN (11.0%), and is
numerically below it** (not significant, p=0.14). Diagnostics
(`diagnostics.md`, per-seed + termination-reason breakdown): falls are rare
everywhere (1–6%) — the underlying locomotion policy never breaks down. The
dominant failure is **collision** (79–94% for the whole SNN family vs 59–62%
for the MLPs), and it's essentially flat across frozen/pure-STDP/R-STDP/TM-NORM.
No per-seed outliers — R-STDP's 10 seeds sit in a tight 3–17% band, so this is a
clean, uniform effect, not a couple of bad seeds dragging the mean.

### 3. Hyperparameter re-sweep (`rstdp_sweep.md`, `rstdp_validate.md`)
M4's recipe (eta=0.05, anchor=0.005) was tuned on one seed. Coarse screen (eta
∈ {0.01,0.02,0.05,0.1}, anchor ∈ {0,0.005,0.02}, 5 seeds × 15 eps) found an
apparent winner (eta=0.1, anchor=0.02 → 16.0%±6.8%) — **but validating it
properly at full rigor (10 seeds × 30 eps) gave 10.3%±3.1%, p=0.63 vs frozen
SNN: it regressed to noise.** The "improvement" was a small-sample artifact.

### 4. Severity screen (`severity_screen.md`) — the real confound
Screened frozen-SNN success across dropout ∈ {0.10, ..., 0.40} (5 seeds × 15
eps): base (unshifted) 42.7%, but **0.30 (M4's pick, chosen from a single-seed
screen) is already floored** — 0.25–0.40 all sit at 2.7–11% success with
75–85% collision. The real headroom zone (a meaningful drop from baseline
without floor-collapse) is **0.15–0.20**. This mirrors M4c's own terrain lesson
(ice at mu=0.20 was too harsh; mu=0.28 worked).

### 5. Corrected severity re-run (dropout=0.20, `../M5_full_comparison_dropout0.2/README.md`)
| Controller | Success (mean±SD) | Recovery (eps) | p vs R-STDP |
|---|---|---|---|
| Frozen MLP | 34.0%±9.5% | 6.4 | 3.6e-03 * |
| Online MLP | 24.0%±13.4% | 6.3 | 0.70 |
| Frozen SNN | 21.3%±7.2% | 7.7 | 0.70 |
| Pure-STDP SNN | 8.7%±4.5% | 20.9 | 9.97e-04 * |
| **R-STDP SNN** | **19.3%±5.3%** | 11.8 | – |
| TM-NORM SNN | 1.7%±2.2% | 30.0 | 4.4e-06 * |

Real headroom this time (base 42.7% → ~20%, not floored) — **still no R-STDP
advantage over frozen SNN** (19.3% vs 21.3%, tied). So severity was a real,
independent bug worth fixing, but it wasn't why R-STDP failed to show recovery.

### 6. Plasticity-scope / third-factor follow-up (`../M5_full_comparison_dropout0.2/plasticity_scope.md`)
At the corrected severity, tested the M4b ablation-grid cells that were never
actually run (10 seeds × 30 eps each, full rigor from the start this time):

| Variant | Success | SD |
|---|---|---|
| Frozen SNN (reference) | 21.3% | 7.2% |
| R-STDP, input+readout, TD (M4 recipe) | 19.3% | 5.3% |
| R-STDP, **all-layers**, TD | 20.7% | 2.0% |
| R-STDP, input+readout, **RPE** | 21.7% | 7.2% |
| R-STDP, **all-layers**, **RPE** | 21.7% | 6.2% |

Every variant lands within noise of frozen SNN. Neither plasticity scope nor
third-factor signal choice changes the outcome.

## Conclusion
Six independent checks (config fix, two severities, an eta/anchor sweep
validated at full rigor, two plasticity scopes, two reward-mode signals) all
agree: **R-STDP does not produce a statistically significant navigation-recovery
advantage over a frozen SNN on the sensor-dropout shift**, with this pretrained
network. M4's 30% was very likely a single-seed outlier — exactly the risk M4's
own writeup flagged and M5 exists to catch. The part of H1 that *does* hold up
in every single test: **R-STDP reliably, significantly beats pure Hebbian STDP**
(e.g. 19.3% vs 8.7% at dropout=0.20) — reward modulation does real work, just
not enough to beat not-adapting-at-all here. M4c (`archive/M4b_terrain_walk_compare/`,
ROADMAP.md row) already showed the opposite result on the terrain/sand shift
(R-STDP closes the full gap to the MLP there) — so H1 is **shift-dependent**: real on a
dynamics-level fault the navigator can velocity-compensate for, not shown on
this sensor-level fault, at least not with this network/interface. A legitimate,
nuanced finding, not a thesis-ending one.

## Reproduce
```bash
conda run -n nmc python scripts/m5_full_comparison.py --dropout 0.30   # original (now-fixed) severity
conda run -n nmc python scripts/m5_full_comparison.py --dropout 0.20   # corrected severity
conda run -n nmc python scripts/m5_diagnostics.py                      # per-seed + termination breakdown
conda run -n nmc python scripts/m5_rstdp_sweep.py                      # eta/anchor coarse screen
conda run -n nmc python scripts/m5_rstdp_validate.py                   # full-rigor validation of the sweep's winner
conda run -n nmc python scripts/m5_severity_screen.py                  # dropout-severity headroom screen
conda run -n nmc python scripts/m5_plasticity_scope.py                 # all-layers / rpe follow-up
```

## Watch it live (not a pre-rendered video)
```bash
conda run -n nmc python scripts/watch_go2_nav.py --controller frozen_snn --seed 6000
conda run -n nmc python scripts/watch_go2_nav.py --controller rstdp --seed 6003 --dropout 0.20
conda run -n nmc python scripts/watch_go2_nav.py --controller expert --no-shift
```
Opens a real interactive MuJoCo GUI window (drag to rotate, scroll to zoom,
space to pause) and steps the chosen controller through an episode in real
time — see `scripts/watch_go2_nav.py`'s docstring for the full controller list
and options (severity, seed, R-STDP eta/anchor/scope/reward-mode, playback speed).

## Known-incomplete
Only R-STDP and TM-NORM have rendered GIFs from the original run
(`r-stdp_snn.gif`, `tm-norm_snn.gif`); frozen SNN, pure-STDP, and both MLPs
don't. Use `watch_go2_nav.py` above to inspect any of them live instead.
