# M2 — MLP baselines on the dynamic Go2 env (frozen via BC+DAgger, online-TD verified)

_Archived 2026-07-17 · controllers 1–2 of the proposal's five, now on full dynamics._

## Frozen MLP (controller 1)
Behavior cloning on D3's 15,131 A* demo steps trained to **86.5% val accuracy**
(inverse-frequency class weights; per-class 0.79–0.93, no majority collapse:
`fig_mlp_training.png`) — but closed-loop it scored only **23% (7/30)**: textbook
**covariate shift** (compounding errors drift the student into undemonstrated states).
War story: [bc-covariate-shift-dagger](../../docs/debug-log/2026-07-17_bc-covariate-shift-dagger.md).

**Fix: DAgger** (`scripts/dagger_go2.py`) — student drives, A* teacher labels every
visited state, aggregate, retrain:

| stage | held-out success (30 eps, seeds 2000+) |
|---|---|
| BC only | 23% |
| DAgger ×2 (+41,744 labeled steps → 56,875) | **33% — PASS** (bar: 30%) |

Student success *during* rollouts rose 35% → 50% across iterations; more iterations
would likely keep helping (diminishing wall-time returns: retrain grows with dataset).
Zero falls in every run. Teacher-with-privileged-info reference: 62%. The remaining gap
is expected — a reactive lidar-only student vs a full-map planner.

Artifacts: `fig_mlp_eval.png` (outcome fractions), `mlp_episode.gif` (successful episode),
`assets/mlp_frozen_go2.pt` (final weights), `data/imitation_go2_dagger.npz` (aggregated
dataset — **M3's SNN must train on this**, not the raw demos).

## Online-TD MLP (controller 2)
Mechanical property verified (`scripts/eval_mlp_go2.py`): one `observe(reward, ...)`
call changes weights (max|ΔW| = 1e-3) with `frozen=False`, and provably does not with
`frozen=True` (max|ΔW| = 0). The online baseline updates from **environment reward
only** — the same signal R-STDP gets — with no expert labels at adaptation time.
Its closed-loop adaptation behaviour is measured in M4/M5, not here.

## Reproduce
```
python scripts/train_mlp_go2.py                       # BC pretrain (86.5% val acc)
python scripts/dagger_go2.py --iters 2 --episodes 40  # DAgger -> 33% held-out
python scripts/eval_mlp_go2.py --episodes 30          # eval + figure + GIF
```
