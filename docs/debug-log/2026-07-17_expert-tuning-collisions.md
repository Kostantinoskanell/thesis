# Building a navigation expert: five failed attempts before A* + defensive braking

_2026-07-17 · severity: design (M1b) · a long one — the core lesson of the milestone_

## Goal
A scripted "expert" that navigates to the goal collision-free, to generate imitation
demonstrations for the frozen MLP-DNN baseline.

## The journey (each row = one measured attempt, 40 episodes)

| Attempt | Method | Success | Failure mode |
|---------|--------|---------|--------------|
| 1 | Reactive gap-follower (pick best front beam) | 55% | collides (steers into obstacles) |
| 2 | Potential field (attraction + repulsion) | 20% | **local minima** — robot stalls, times out |
| 3 | Follow-The-Gap + safety bubble | 40% | still collides |
| 4 | + hard forward-safety override | 35% | collisions NOT head-on (override useless) |
| 5 | + scan erosion (body-width gaps) | 37% | fixed collisions but caused **timeouts** |
| 6 | **Privileged A\*** (true obstacle positions) | 55% | 0 static, all-dynamic collisions |
| 6b | A\* + more path inflation (0.45→0.75 m) | 93% static / 55% full | dynamic only |
| 6c | + dynamic speed cap (1.0→0.5 m/s) | 65% | dynamic only |
| 6d | + defensive braking (yield to crossings) | **70%** | residual dynamic |

## The three key diagnoses (why data-driven debugging mattered)

1. **Reactive methods are fundamentally limited here.** Aggressive → collide; conservative
   → stall. No single threshold works. Switched to a *privileged* A\* teacher (it may use
   ground-truth obstacle positions; the student imitates from LiDAR only — standard
   teacher-student IL).
2. **"Corner-cutting", not planning, caused collisions.** A\* paths were collision-free,
   but the discrete-action robot (fixed 0.6 m/s, limited turn rate) cut corners tracking
   them. Fix: inflate obstacles generously (0.75 m) so the path stays clear of the body.
   → static collisions went to ~0.
3. **A collision-kind counter proved all remaining hits were DYNAMIC** (static=0). Two
   fixes: (a) cap dynamic-obstacle speed *below* the robot's, else a faster obstacle can
   catch it from behind → unavoidable for ANY controller (also compresses the
   inter-controller differences the thesis measures); (b) defensive braking — since the
   kinematic robot stops instantly, yield to an obstacle predicted to cross just ahead.

## Outcome
A\* + prediction + defensive braking. Overall success ~65–70% on the stochastic
dynamic task, **~100% static-obstacle avoidance**. Since imitation demonstrations are
drawn only from collision-free successful episodes, the residual dynamic failures are
simply discarded — 80 episodes yielded **13,387 clean demo steps**, plenty to train the
MLP. The exit criterion was corrected to measure *clean-demo count*, not teacher success
rate.

## Lessons
- Add a **failure-mode breakdown** (collided vs timed-out; static vs dynamic) to any
  eval harness *first* — it turned guesswork into three targeted fixes.
- For imitation data, the teacher may be **privileged**; don't cripple it to use only
  the student's observations.
- Make the task **well-posed** (obstacles must be evadable) before blaming the policy.
- Discrete-action + speed-limited robots need **generous path clearance** and the
  ability to **yield (brake)**, not just steer.

## Task-design changes committed as a result
- `NavConfig.dynamic_speed_range`: (0.3, 1.0) → **(0.2, 0.5)** m/s (below robot speed).
- Added `env.privileged_state()` and `env.info["collision_kind"]` (static/dynamic).
