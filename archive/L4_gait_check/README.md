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

**Covariate-shift limitation — then FIXED with DAgger.** The BC-distilled net fell ~5 s
into an *artificially sustained constant* command (vx held at 0.3/0.5 for 20 s) — the MLP
teacher rarely saw held commands (the env resamples them), so the student never learned to
recover in that out-of-distribution regime. **DAgger fix (`dagger_walk_forward.gif`,
the M2 nav-layer recipe):** rolled the STUDENT under held forward commands (so it visits
its own drift/pre-fall states), labeled each state with the TEACHER's action
(`l4_dagger_collect.py`, 153.6k pairs), aggregated with the original 128k and retrained
warm-started (`l4_distill_spiking.py --init-weights`, val-MSE 0.025). **Result: robust.**
Under sustained vx=0.5 (which used to fall at step ~244): **3/3 episodes full 1000 steps,
0 falls, body vx 0.495 vs commanded 0.5 (err 0.028 — matches the MLP teacher), walked
~9.3 m forward.** So the spiking Go2 now walks robustly under both normal AND sustained
commands. (PPO fine-tuning from the BC init was tried first and *degraded* the gait
0.036→0.29, so abandoned — supervised distillation + DAgger is the winning path.)

## UPRIGHT FIX (2026-07-23): the "faithful" 0.18-0.19 m gait was still too low — retrained the teacher taller
The walking result above was real (verified by trajectory + video) but still visibly
**crouched** — user feedback: "it walks too crouched i dont like it." Root cause: the MLP
*teacher* itself only walked at **0.184 m** (normal Go2 standing ≈0.30 m) because the
original flat-velocity reward has no height incentive strong enough to prevent settling low.
The spiking student can only ever be as tall as its teacher, so distillation faithfully
reproduced the crouch — it needed a taller teacher, not a distillation fix.

**Fix: retrained the MLP teacher itself with the same anti-crouch reward-shaping idea as
L3/D14** (`scripts/make_isaac_train_upright.py` + `scripts/wsl_isaac_upright.sh`,
`UPRIGHT_ANTICROUCH=1`: `base_height_l2` reward, target 0.30 m, weight −10.0, + `<0.18 m`
termination), but applied to the **standard MLP actor**, not the spiking one — the MLP
has no PPO-discovery problem (D14's stand-still trap was spiking-specific), so it simply
learns to walk taller under the added incentive. 1200 iters, reward converged to ~35
(matching the original 36.25 baseline) with a near-zero height-error term.

**Verified by trajectory (not reward): new teacher walks at 0.279–0.305 m** (essentially
Go2's natural standing height), vx 0.46–0.51 vs commanded 0.5, path ~18–20 m per 20 s
episode, 0 falls. Re-ran the full pipeline on top of it — recollect (128k pairs) → BC
distill (val-MSE 0.025) → DAgger for sustained-command robustness (153.6k pairs,
`dagger_walk_forward_upright.gif`): **3/3 episodes × 1000 steps, 0 falls, base height
0.305 m, vx 0.492 (err 0.033), path 19.2 m.** Confirmed upright by eye in the MuJoCo replay,
not just by the height number — legs mostly under the body, not splayed.

Best upright checkpoints: `data/l4_dagger_upright_spiking.pt` (dense, faithful walker) and
`data/l4_sparse_t5_upright_v2.pt` (sparse + T=5, energy-positive — see `archive/L5_energy/`).
The old crouched checkpoints/GIFs above are kept as the historical record of how the
walking result was first achieved; they are superseded by the upright versions for any
use beyond that history.

## DRAGGING REAR LEG FIX (2026-07-23, user feedback watching the video: "the 4th leg on behind is almost just dragging")
Quantified it first, not just eyeballed: `scripts/l4_leg_amplitude.py` measures each leg's
per-joint range-of-motion + oscillation-sign-change count during steady walking. On the
upright walker, **RR (rear-right) was a clear, real outlier** — hip swing amplitude 0.123
(less than half of every other leg's 0.25–0.32), smallest thigh/calf amplitude too. Checked
the root cause the same way as the crouch: **the MLP teacher itself already had this
asymmetry** (confirmed on `data/upright_teacher_walktest.npz`), so the spiking student was
just faithfully (and slightly more jitterily) copying it. Nothing in the reward penalized
uneven leg usage — PPO found a "good enough" asymmetric trot.

**Fix: added Isaac Lab's built-in `mdp.feet_slide` reward term** (penalizes a foot's linear
velocity while in ground contact — the literal definition of "dragging") to the teacher's
reward, alongside the existing base-height term. **First attempt at weight −1.0 was a
disaster**: the policy learned to freeze rigidly (all 4 joint amplitudes collapsed to
~0.001–0.02, vx→0, path 0.61 m in 20 s) — avoiding the slide penalty by never stepping at
all, a clean illustration of over-strong reward shaping breaking a task entirely. Checked
what weight real Isaac Lab humanoid configs (G1/H1/Digit) use for this same term: **−0.1**,
10× gentler — retrained with that and it converged normally (reward ~30 vs the original
~35, 1200 iters from scratch).

**Verified by trajectory + per-leg amplitude + an actual multi-frame visual check (not a
single GIF thumbnail — a MuJoCo frame sequence across a full stride cycle,
`scripts/l4_frame_sequence.py`):** all four legs visibly lift and cycle, no leg stays flat
on the ground while the others step. Quantitatively, RR's hip is now the *largest*
amplitude of all four legs (0.24–0.36) and its calf motion falls squarely within the other
three legs' range (previously the smallest and jitteriest of the four) — the whole-leg drag
is resolved. RR's *thigh* joint specifically still shows somewhat smaller amplitude than
the other three (a residual, joint-level asymmetry, not a whole-leg one) — an honest partial
result, not a perfect fix.

Re-ran the full pipeline (recollect 128k → BC-distill val-MSE 0.030 → DAgger 153.6k →
retrain val-MSE 0.029) on the new teacher (`/home/hapos/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-07-24_00-41-58/model_1199.pt`).
**Result (`dagger_walk_forward_v3.gif`): 3/3 episodes × 1000 steps, 0 falls, base height
0.338 m (taller than before), vx 0.523 (err 0.041), path 20.47 m** — the best walking
metrics of any checkpoint in this project, with the leg-drag substantially reduced. Best
checkpoint: `data/l4_dagger_v3_spiking.pt`. Energy-positive sparse variant carries the fix
through too — see `archive/L5_energy/README.md`.

## Reproduce
```
# dump a trajectory (headless, compute-only -- no RTX render needed):
bash scripts/wsl_isaac_l4.sh --mode baseline --episodes 1 --dump-traj data/l4_walk_traj.npz
python scripts/l4_plot_gait.py data/l4_walk_traj.npz archive/L4_gait_check/fig_gait_diagnostic.png
python scripts/l4_render_traj_mujoco.py data/l4_walk_traj.npz archive/L4_gait_check/walk_crouch.gif

# upright teacher (WSL isaac env):
NUM_ENVS=2048 MAX_ITER=1200 TARGET_H=0.30 HEIGHT_W=-10.0 MIN_H=0.18 \
    bash scripts/wsl_isaac_upright.sh

# + feet_slide anti-drag penalty (weight -0.1, NOT -1.0 -- see the disaster above):
NUM_ENVS=2048 MAX_ITER=1200 TARGET_H=0.30 HEIGHT_W=-10.0 MIN_H=0.18 \
    FEET_SLIDE=1 FEET_SLIDE_W=-0.1 bash scripts/wsl_isaac_upright.sh

# per-leg amplitude/symmetry check + multi-frame visual check:
python scripts/l4_leg_amplitude.py data/l4_dagger_v3_walktest.npz
python scripts/l4_frame_sequence.py data/l4_dagger_v3_walktest.npz archive/L4_gait_check/seq 300,312,324,336
```
