# M1b — privileged A* teacher + imitation dataset

_Archived 2026-07-17_

- A* planner on an inflated occupancy grid + pure-pursuit + defensive braking.
- ~100% static-obstacle avoidance; residual failures are dynamic crossings.
- 13k+ clean (obs,action) demo steps collected from successful episodes.
- `fig_expert_trajectory.png` — one collision-free run from start to goal.
- See docs/debug-log/2026-07-17_expert-tuning-collisions.md for the full saga.
