# D2 — quadruped walks in MuJoCo via a trained RL policy (full dynamics)

_Archived 2026-07-17 · the SOTA locomotion layer (replaces the CPG interim)_

- **Policy:** PPO velocity-tracking policy trained in **MuJoCo Playground** (brax, MJX)
  on the **RTX 4060 via WSL2** — 200M steps, ~35 min, final eval reward **29.3**.
- **Deployment:** params loaded and rolled out in MuJoCo (`rollout_go1.py`); the qpos
  trajectory replayed through an offscreen renderer.
- **Result (`fig_rl_tracking.png`):** commanded 1.0 m/s → **actual 0.95 ± 0.05 m/s**
  (tight tracking), base height locked at 0.30 m — **upright the whole time**. Compare
  the CPG interim (`../D2_go2_walk/`): ~60% speed, heavy yaw wobble. Night and day.
- `walk.gif` — the quadruped trotting forward under the RL policy, full dynamics.

## Important caveat: this is the Unitree **Go1**, not Go2
MuJoCo Playground ships the **Go1** (Go2's near-identical 12-DOF predecessor), not the
Go2. The plasticity science is robot-agnostic, so Go1 is fine for the sim experiments;
the real-Go2 hardware demo uses the onboard SDK sport-mode (velocity commands), so it is
unaffected. Adapting the policy/env to the exact Go2 model is a later polish (a Go2
Playground env would need porting).

## Reproduce (WSL2 GPU)
```
bash scripts/wsl_repin_rl.sh          # working stack (see debug-log)
bash scripts/wsl_train_go1_full.sh    # train (background task); checkpoints each eval
bash scripts/wsl_rollout_go1.sh       # verify velocity tracking
bash scripts/wsl_render_go1.sh        # GIF + tracking graph
```
Stack: playground 0.0.5 · mujoco/mjx 3.4.0 · brax 0.12.1 · jax[cuda12] 0.4.38.
