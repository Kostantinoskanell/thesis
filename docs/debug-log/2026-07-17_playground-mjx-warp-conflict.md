# MuJoCo Playground ↔ MJX ↔ Warp version conflict (RL locomotion timebox)

_2026-07-17 · severity: blocking (D2 RL path) · timeboxed per plan_

## Context
Chose to train our own Go2/Go1 velocity policy (no pretrained artifact exists). GPU
path confirmed working: WSL2 Ubuntu + RTX 4060, `jax 0.11` sees `CudaDevice(id=0)`.

## The wall
`pip install playground` (MuJoCo Playground 0.2.0) pulls `mujoco / mujoco-mjx 3.10.0`,
whose MJX now defaults to the **MuJoCo-Warp** backend:
```
mjx.put_model(..., impl=...) -> graph_mode = graph_mode or getattr(mjxw.types.GraphMode, 'WARP')
AttributeError: type object 'int' has no attribute 'WARP'
```
`mjxw` (the warp wrapper) isn't wired up, so `GraphMode` falls back to `int`. Installing
`mujoco-warp` + `warp-lang` did NOT resolve it (one bounded attempt). Also: Playground has
**Go1**, not Go2 (Go1 = Go2's predecessor; near-identical 12-DOF quadruped).

Installed stack: playground 0.2.0, mujoco/mujoco-mjx 3.10.0, jax 0.11.0, brax 0.14.2.

## Decision
Timeboxed the RL-stack dependency archaeology (promised not to burn days on a
NON-contribution component). Options recorded for the user:
1. **Pin a pre-Warp combo** (playground<0.2 + mujoco 3.3.x classic MJX) — one more RL
   attempt; may work, may be more version-hell.
2. **Convex-MPC locomotion** — robust, Windows-native, no JAX; model-based not RL.
3. **Proceed on the CPG interim** — unblock the plasticity science now (real Go2 walks
   via SDK sport-mode on hardware anyway), revisit locomotion quality later.

## RESOLUTION — the working pinned stack (three layers of version-hell)
User chose to keep pushing RL. Peeled three conflicts:
1. **Warp backend** — playground ≥0.1.0 + mjx 3.10 require MuJoCo-Warp (`GraphMode.WARP`)
   which wouldn't resolve → pin **playground 0.0.5** (last pre-Warp).
2. **brax ↔ jax** — brax's PPO calls `jax.device_put_replicated`, removed in jax ≥0.10 →
   pin **jax[cuda12] 0.4.38** (still has it).
3. **mjx `_impl`** — playground 0.0.5's collision code needs `data._impl`, absent in mjx
   3.3.1 but present (and pre-Warp) in **mjx 3.4.0**.

Working WSL2 GPU stack (verified `env.step` runs, reward computed):
```
playground==0.0.5  mujoco==3.4.0  mujoco-mjx==3.4.0  brax==0.12.1  jax[cuda12]==0.4.38
```
Reproduce with `scripts/wsl_repin_rl.sh`. ("Failed to import warp" warnings are benign —
it falls back to the JAX backend.)

## Lesson
"Train our own RL policy" is feasible on this GPU, but the JAX/MJX/Playground/Warp stack
is fast-moving and version-fragile. Budget for a pinned, known-good environment (or a
container) before relying on it — and don't let a support component block the thesis core.
