# L4 gait check — does the spiking Go2 policy actually walk? (No.)

_2026-07-22 · a diagnostic that reframes the whole L3 "reward ~8 ceiling" finding._

## Why this exists
L3 concluded the spiking (PopSAN) locomotion policy plateaued at reward ~8 (vs. the
MLP baseline's 36), and L4 started testing R-STDP "gait recovery" on ice. But we never
actually *watched* the policy — only tracked reward. This check dumps a real rollout
trajectory (base pose + 12 joint angles, headless compute-only) and inspects it two ways:
a quantitative figure and a MuJoCo kinematic replay (Isaac Sim's RTX renderer won't
initialize on this headless WSL box — graphics device creation fails — so we replay the
recorded poses on our MuJoCo Menagerie Go2, which renders fine).

## Finding: it does NOT walk — it belly-flops and survives
`fig_gait_diagnostic.png` (quantitative) + `walk_crouch.gif` (MuJoCo replay):
- **Base height collapses 0.40 m → ~0.11 m in ~0.5 s and stays there** the whole 20 s
  episode. Normal Go2 standing base is ~0.30 m. At 0.11 m the body is essentially on the
  floor with legs splayed (see the GIF).
- **Forward velocity never tracks the command** — commanded vx steps to +0.3 m/s, actual
  stays ~0. Mean |vx| ≈ 0.02 m/s.
- **Travels 0.49 m total in 20 s.** No directed locomotion.
- Joint angles settle to near-constant values — no periodic gait cycle.

The reward ~8-13 came from *not triggering the base-contact termination* while collecting
alive/orientation reward — a **degenerate crouch/belly-flop local optimum**, not walking.

## What this reframes
- **L3's "reward ~8 is a hard architectural ceiling" (D12) is corrected:** ~8 is a
  *non-walking local optimum*, not a capacity limit. The MLP (reward 36, vel-err 0.16 m/s)
  actually walked; the spiking net got stuck in a bad optimum — an **optimization/
  exploration failure, not a representational-capacity one**. Consistent with the whole L3
  saga pointing at optimization difficulty (the adaptive-LR-schedule crush, etc.). The
  persistent ~1.4 m/s velocity-tracking error across every L3 variant was the unheeded tell.
- **L4 (R-STDP ice recovery) was ill-posed on this base policy:** you can't study
  "recovering a gait under a terrain shift" when the base policy has no gait to begin with.
  L4's negative result (R-STDP didn't recover) is uninformative given this — it needs a
  genuinely-walking base policy first.

## Next levers (for a genuinely-walking spiking policy, if pursued)
Reward/optimization-side, not capacity-side: penalize low base height / reward standing
explicitly; stronger exploration (NoisySAN, arXiv:2403.04162); a curriculum; or revisit
whether population-coding + this reward shaping can escape the crouch optimum at all. The
MLP walks under the identical reward, so the reward isn't the whole story — the spiking
optimization is.

## RESOLUTION (2026-07-22, autonomous session): distillation made it WALK
Reward-shaping alone couldn't fix it — anti-crouch (base-height reward + <0.20 m
termination) raised the base 0.11 m → 0.24 m and killed the belly-flop, but the policy
then found a *second* stationary optimum (stand still; two variants, incl. 4× velocity
reward + entropy boost, both refused to walk under a forced forward command). The spiking
policy simply won't **discover** walking via PPO exploration — it converges to whatever
stationary optimum survives.

**What worked — distill the walking MLP teacher (the thesis's own M3 recipe):**
1. Verified the MLP baseline genuinely walks (forced vx=0.5 → actual 0.501, err 0.026,
   ~9.4 m). It walks at base height ~0.18 m (a low but real gait — so the distilled net's
   ~0.19 m below is faithful reproduction, NOT a crouch).
2. Collected 128k (obs, action) pairs from the MLP across 256 parallel envs
   (`l4_collect_mlp_data.py`).
3. BC-trained the PopSAN spiking actor to match (`l4_distill_spiking.py`, val-MSE 0.025,
   surrogate-grad BPTT).
4. **Result (`distilled_walk.gif`, `fig_distilled_gait.png`): the distilled spiking policy
   WALKS** — 5/5 episodes full 1000 steps, ZERO falls, return 26–41 (matches/beats the MLP
   baseline's 13–36 under the same reward), tracks the commanded velocity, upright
   quadruped stance with a stepping gait (verified by eye in the MuJoCo replay — legs
   underneath, body off the ground, stepping + translating across frames). This is, as far
   as the literature search found, the first spiking Go2 locomotion policy that walks.

**Honest limitation:** under an *artificially sustained constant* forward command (vx held
at 0.3 or 0.5 for 20 s) it falls at ~5 s — a covariate-shift effect (the MLP teacher rarely
experienced held-constant commands, which the env resamples periodically, so the distilled
net didn't learn that out-of-distribution regime). Under normal (resampling) commands it's
stable. Fix = DAgger (relabel student-visited states) or include sustained-command rollouts
in the distillation set — future work. PPO fine-tuning from the BC init was tried and
*degraded* the gait (velocity err 0.036 → 0.29), so it was abandoned — distillation alone
is the better result here.

## Reproduce
```
# dump a trajectory (headless, compute-only -- no RTX render needed):
bash scripts/wsl_isaac_l4.sh --mode baseline --episodes 1 --dump-traj data/l4_walk_traj.npz
python scripts/l4_plot_gait.py data/l4_walk_traj.npz archive/L4_gait_check/fig_gait_diagnostic.png
python scripts/l4_render_traj_mujoco.py data/l4_walk_traj.npz archive/L4_gait_check/walk_crouch.gif
```
