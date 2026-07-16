# M1 smoke test hangs (never terminates)

_2026-07-17 · severity: bug (test harness)_

## Symptom
`smoke_m1.py` produced no output and ran indefinitely; had to kill the process.

## Cause
The test respawned the episode on every `terminated` (collision):
```python
if terminated:
    obs, _ = env.reset(...)   # <-- resets env.t back to 0
```
A random policy collides within a few steps, so `env.t` was reset to 0 constantly and
**never accumulated to the 60 s** needed for `truncated=True`. The break condition was
therefore essentially unreachable → infinite loop. (Bonus latent bug: `n_static_after`
would have been undefined if the shift at t=30 s was never reached.)

## Fix
The smoke test's job is to verify the **clock + shift logic**, not to survive. Run **one
continuous 60 s episode**, ignore collision-termination, and just count collisions:
```python
while True:
    obs, r, terminated, truncated, info = env.step(action)
    ...
    if truncated:   # t >= episode_len_s
        break
```
Result: `60.0 s, shift fires at t=30.0 s, static 8->16, dynamic 3->6` → PASS.

## Lesson
Keep "does the mechanism fire?" tests separate from "does the agent perform?" tests.
Never reset the episode clock inside a loop whose exit condition depends on that clock.
Real evaluation runs (M5+) will instead run **fixed-length episodes and log per-step
outcomes**, not respawn on termination.
