# Compute Environments

Three isolated environments are used, deliberately kept separate so a change in
one can't break another.

| Env | Where | Stack | Used for |
|-----|-------|-------|----------|
| `nmc` (conda) | Windows | Python 3.11, torch **CPU**, snntorch, mujoco, pybullet | The science stack: Go2NavEnv, MLP/SNN controllers, all closed-loop eval, figures |
| `nmc-rl2` (venv) | WSL2 | Python 3.12, JAX+CUDA, mujoco-playground 0.0.5, mjx 3.4.0, brax 0.12.1 | Go2 RL locomotion-policy training (D2) — GPU |
| `nmc-snn` (venv) | WSL2 | Python 3.12, **torch 2.13.0+cu130 (CUDA)**, snntorch 1.0.0 | **M5** SNN surrogate-gradient training on GPU (≥10 seeds) |

## Why the split
- `nmc-rl2` pins a pre-Warp Playground stack that was hard-won (see debug-log
  `playground-mjx-warp-conflict`); adding torch-CUDA to it risks a CUDA-version clash.
  `nmc-snn` is a clean venv so the RL stack stays untouched.
- Windows `nmc` torch is CPU-only, fine for MLP training (seconds) and closed-loop eval,
  but SNN BPTT is ~45 min/seed on CPU. M3 (few seeds) ran on Windows CPU in parallel
  (20 cores, thread-capped). **M5 (≥10 seeds) uses `nmc-snn` on the RTX 4060** — GPU
  matmul verified 2026-07-17 (~10–30× per-seed speedup expected).

## Data handoff
WSL reaches the repo at `/mnt/c/Users/hapos/Desktop/thesis`; models/datasets in
`assets/` and `data/` are shared across all three envs via the filesystem. Cross-runtime
numerical parity (WSL-trained → Windows-run) is guarded by
`scripts/verify_policy_parity.py` (D3a).

## Gotcha
`conda run -n nmc python -c "<multiline>"` fails ("scripts where arguments contain
newlines not implemented"). Use a script file, or chain single-line `conda run` calls
with `&&`. (Debug-log: `conda-run-inline-newlines`.)
