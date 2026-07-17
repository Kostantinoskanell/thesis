# D3a — the trained Go2 policy runs on WINDOWS (pure NumPy, parity-verified)

_Archived 2026-07-17 · first step of D3: the locomotion layer now lives where the
science stack lives (Windows conda `nmc`), no JAX/WSL needed at experiment time._

The D2 policy was trained in WSL2/JAX; the M2–M8 experiments run in the Windows MuJoCo
loop. This step bridges the two runtimes and **proves the bridge is exact**:

- **Export** (`scripts/rl/export_go2_policy.py`, WSL): normalizer mean/std + MLP weights
  → `assets/go2_policy_export.npz`, plus 64 ground-truth (obs, action) pairs from brax's
  own deterministic inference fn → `assets/go2_parity_vectors.npz`.
- **NumPy runtime** (`src/nmc/rl/numpy_policy.py`): normalize → (512,256,128) swish MLP
  → tanh(loc). ~60 lines, no JAX/torch.
- **Parity gate** (`scripts/verify_policy_parity.py`, Windows): **max|err| = 2.7e-7**
  over all 64 vectors — float32 noise, i.e. the two runtimes are the same network.
  This was the classic silent-failure risk (obs-ordering / normalization / activation
  mismatch degrades the walk without erroring); it is now closed *for the network*.
- **Walker** (`src/nmc/platform/go2_rl_walker.py`): Windows MuJoCo loop on the same
  Playground scene XML, same control scheme (50 Hz policy / 250 Hz physics, targets =
  default_pose + 0.5·action). **Remaining hand-written bridge:** the 48-dim obs
  construction mirrors `nmc/rl/envs/go2/joystick.py::_get_obs` (noise-free) — if that
  env obs ever changes, change `Go2RLWalker.obs()` in lockstep.

## Result (`fig_windows_tracking.png`, `walk_windows.gif`)
Four-phase command schedule — the exact interface the SNN navigator will drive:

| phase | command | actual (1 s transient skipped) |
|---|---|---|
| walk fwd | vx 0.8 | **0.72 ± 0.05 m/s** |
| arc left | vx 0.6, ω 0.7 | **vx 0.51, ω 0.65 ± 0.08 rad/s** |
| fast fwd | vx 1.0 | **0.88 ± 0.07 m/s** |
| stop | 0 | **0.00, stands** |

Height 0.28–0.32 m throughout, no falls; top-down trajectory shows the clean U-turn arc
during the ω phase. Same mild (~10–15%) vx undershoot as the WSL rollout — property of
the trained policy, not of the export.

## Reproduce
```
# WSL (once, after training):
bash scripts/wsl_rollout_go2.sh              # (optional sanity)
wsl python scripts/rl/export_go2_policy.py   # export + parity vectors
# Windows (conda nmc):
python scripts/verify_policy_parity.py       # must print PARITY OK
python scripts/render_go2_rl_windows.py      # GIF + tracking figure
```
