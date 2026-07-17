# D2 (final) — the actual **Go2** walks via a trained RL policy (full dynamics)

_Archived 2026-07-17 · supersedes `../D2_go2_rl/` (the Go1-model result, kept as reference)_

Closes the caveat in the earlier D2 archive: MuJoCo Playground has no built-in Go2, so we
**ported the Go1 joystick env to the official Menagerie Go2 model** ourselves —
`src/nmc/rl/envs/go2/` (constants/base/joystick/randomize + our own XML pair). Train and
deploy now agree on the same robot as the real one in the lab.

- **Env port:** Menagerie's `go2_mjx.xml` + the sensors Playground's joystick task needs
  that Menagerie doesn't ship (`local_linvel` velocimeter, `upvector` framezaxis, per-foot
  `global_linvel`). Registered as **`Go2JoystickFlatTerrain`** into Playground's locomotion
  registry at import (documented custom-env pattern; no site-packages edits). Obs/action
  shapes match Go1's exactly (48 state / 123 privileged / 12 act).
- **Policy:** PPO (brax, MJX) on the **RTX 4060 via WSL2** — 200M steps, **~2.8 h**
  (~6× slower per-step than the Go1 run: Go2's Menagerie XML keeps full body
  self-collision; Go1's Playground variant is feet-only), final eval reward **23.98**.
- **Result (`fig_rl_tracking.png`):** commanded 1.0 m/s → **actual 0.84 ± 0.07 m/s**,
  base height 0.27–0.30 m, **upright the whole 20 s**, periodic gait ripple (real trot,
  not skating). Tracking is looser than the Go1-model run (0.95 m/s) — plausibly the
  extra self-collision contacts eating reward-shaping margin; revisit reward/collision
  tuning if D3 needs tighter tracking.
- `walk.gif` — the Go2 trotting forward under the RL policy, full dynamics.

## Reproduce (WSL2 GPU)
```
bash scripts/wsl_repin_rl.sh            # working stack (see debug-log)
bash scripts/wsl_train_go2_smoke.sh     # 3M-step smoke test (~8 min)
bash scripts/wsl_train_go2_full.sh      # full train (background task); checkpoints each eval
bash scripts/wsl_rollout_go2.sh         # verify velocity tracking
bash scripts/wsl_render_go2.sh          # GIF + tracking graph
```
Stack: playground 0.0.5 · mujoco/mjx 3.4.0 · brax 0.12.1 · jax[cuda12] 0.4.38.
Policy params: `assets/go2_policy.params` (+ `.curve.json` reward curve; gitignored).
