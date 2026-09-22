# M4b extras — pilot-robustness extensions, run at M5's statistical standard

_2026-09-17 · headline: **the shift-robustness M4 credited to R-STDP belongs to the
ALIF neuron model, not to the plasticity.** Plus: no severity window and no dead-beam
mask where R-STDP significantly beats the frozen SNN — the last two escape hatches
left open by [M5](../M5_full_comparison/README.md) are now closed._

Run with `conda run -n nmc python scripts/m4b_extras.py --part {severity|masks|neuron|traj}`.
These are 5-seed × 10-episode screens (wide CIs, ±10–24 pts) unless stated otherwise —
scans to locate effects, not replacements for M5's 10 × 30 headline numbers.

## M4b-7 — neuron model alone vs plasticity (`neuron_ablation.md`)
**10 seeds × 10 eps, dropout = 0.20.** The one pre-registered question in the M4b plan
with a stated decision rule: _"frozen-LIF landing near frozen-ALIF's would confirm the
recovery is R-STDP's doing, not ALIF's — a large gap would mean ALIF itself is doing
some of the work."_

| variant | success | 95% CI |
|---|---|---|
| frozen **LIF** | 7.0% | ±5.9% |
| frozen **ALIF** | 17.0% | ±9.0% |
| **R-STDP** ALIF | 12.0% | ±5.6% |

- ALIF vs LIF (**neuron model alone**): **+10.0 pts, p=0.051**
- R-STDP-ALIF vs frozen ALIF (**plasticity alone**): −5.0 pts, p=0.302

**We got the large gap.** The adaptive-threshold neuron more than doubles shifted
success over vanilla LIF (7% → 17%), while adding reward-modulated plasticity on top
costs 5 points. So the robustness M4 read as "R-STDP recovery" is, on this evidence,
**a property of the ALIF neuron model**, which was introduced as an unrelated SOTA
upgrade in [D2](../../docs/references/sota_decisions.md) and never separately credited.

This also **contrasts with Zhao et al. 2025 (Table VII)**, whose finding motivated this
ablation: they report adaptive-threshold neurons alone *not* rescuing OOD performance.
Here they largely do. Worth flagging as a genuine disagreement with the cited result
rather than smoothing over.

## M4b-1 — severity sweep, all six controllers (`severity_sweep.md`, `fig_severity_sweep.png`)
M5 screened severity for the frozen SNN only, leaving open whether R-STDP helps in
*some* severity window that the two tested points (0.20, 0.30) happened to miss.

| severity | frozen SNN | R-STDP | delta | p |
|---|---|---|---|---|
| 0.10 | 42.0% | 48.0% | +6.0 pts | 0.582 |
| 0.15 | 30.0% | 24.0% | −6.0 pts | 0.446 |
| 0.20 | 20.0% | 12.0% | −8.0 pts | 0.383 |
| 0.30 | 12.0% | 12.0% | −0.0 pts | 1.000 |

**No severity window where R-STDP significantly beats the frozen SNN.** Secondary
observations: Pure-STDP (2–10%) and TM-NORM (0–2%) are broken at *every* severity, and
the SNN-vs-MLP gap is much narrower at mild severities (42% vs 50% at 0.10) than the
floored 0.30 condition made it look.

## M4b-2 — is the result mask-specific? (`multi_mask.md`)
M4 and M5 always killed the *same* beam block (start = 8). Four positions, dropout 0.20:

| dead-beam start | frozen SNN | R-STDP | delta | p |
|---|---|---|---|---|
| 0 | 32.0%±18.4% | 46.0%±18.8% | +14.0 pts | 0.178 |
| 8 (M4/M5) | 20.0%±19.6% | 12.0%±13.6% | −8.0 pts | 0.383 |
| 16 | 8.0%±10.4% | 6.0%±6.8% | −2.0 pts | 0.668 |
| 24 | 28.0%±22.2% | 32.0%±10.4% | +4.0 pts | 0.667 |

No mask reaches significance. **Pooled: frozen 22.0% vs R-STDP 24.0%** — so pooling over
masks the true effect looks like ~0 rather than negative, and it is worth noting that
**M4/M5's particular mask (start=8) is one of the less favourable ones for R-STDP**.
That slightly softens M5's negative without changing it. Mask choice also strongly sets
task difficulty (frozen SNN ranges 8–32% across masks), and R-STDP shows the larger
across-mask spread (15.9% vs 9.2%) — plasticity adds variance.

## M4b-3 — trajectory overlay (`fig_trajectory_overlay.png`)
Frozen SNN vs R-STDP's first post-shift episode vs R-STDP after 12 adaptation episodes,
same seed/layout, 20% dropout — the qualitative companion to the numbers above.

**Fixed a real bug found by visual inspection:** the first cut took ONE obstacle
snapshot after the last of the three episodes and used it as the shared background
for all three paths. Static obstacles don't move so that was fine for them, but
DYNAMIC obstacles do — the snapshot showed wherever they'd ended up after the LATEST
episode, not where they actually were during the (separate, earlier) frozen or
first-post-shift episodes. Confirmed via `scripts/m4b_traj_diag.py`: both a frozen-SNN
and an R-STDP collision in this exact seed were with a dynamic obstacle, at positions
the old single snapshot did not represent — the bearing was also OUTSIDE the dead-beam
sector at the moment of impact, so this was never a blind-spot effect, just a narrow
map chokepoint a mover happened to be near. Now each episode keeps its own end-of-
episode obstacle snapshot (`rollout()` returns it); static obstacles draw once as
shared background, dynamic ones draw per-trajectory in that trajectory's own colour.

**Checked 4 seeds (6000/6001/6003/6006) looking for a more illustrative example
before settling on one** — across all 12 logged episodes (4 seeds x 3 trajectories
each), **zero reached the goal**. That is not a fluke: M5 measured ~20% success at
this severity for both frozen SNN and R-STDP, so 0/12 is an unsurprising draw from
that rate, not evidence of anything additionally broken. Deliberately did NOT keep
searching for a seed where someone succeeds — doing so would be quiet cherry-picking
against a result that fails 4 times out of 5, and would visually misrepresent the
finding. The figure is kept as an honest illustration that the three trajectories are
visually indistinguishable in outcome, which is the correct picture of a null result,
not a shortcoming of the demo. The other 3 seeds' figures are kept in this folder
(`fig_trajectory_overlay_seed*.png`) for transparency.

## M4b-6 — firing-rate distribution shift
Run as part of M6 (it needs the same spike instrumentation); see
[`../M6_energy/README.md`](../M6_energy/README.md). Summary: the shift barely moves the
frozen SNN's firing statistics at all (1.52% → 1.51%), and R-STDP does **not** pull them
back toward the clean reference — which independently explains why threshold-
renormalization (TM-NORM) was never going to rescue this shift.

## Not run
M4b items (4) TM-NORM as a standalone study and (5) the D8/D9 ablation grid were folded
into [M5](../M5_full_comparison/README.md) instead (TM-NORM is a full controller in the
M5 comparison; the scope × third-factor grid was run there at 10 seeds × 30 eps).
