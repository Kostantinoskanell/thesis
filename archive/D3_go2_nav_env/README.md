# D3 — dynamic nav env: Go2 + LiDAR + obstacles + mid-episode shift (full dynamics)

_Archived 2026-07-17 · the dynamics foundation is complete; M2–M8 run on this env._

`src/nmc/envs/go2_nav_env.py` replaces the kinematic `nav_env.py` with the real MuJoCo
Go2 walked by the D2 RL policy (`Go2RLWalker`), keeping every interface the science
stack expects — **same 37-dim obs** [lidar(32), goal_dx, goal_dy, heading_err, v, ω],
**same 4 discrete actions** (→ velocity commands the policy tracks), same reward
(progress + collision penalty), same `privileged_state()` contract, so `PrivilegedExpert`
and the M2+ controllers plug in unchanged.

**Engineering notes**
- MuJoCo can't add bodies mid-episode → the shift set is **pre-allocated as mocap
  cylinders parked 30 m outside the arena** and teleported in at `shift_time_s`
  (obstacle count doubles, no reset). Dynamic obstacles move by mocap_pos updates and
  bounce off walls, capped at 0.4 m/s (< the Go2's ~0.5 m/s effective speed).
- **LiDAR** = 32 × `mj_ray` filtered to geom group 4 (obstacles+walls only) — rays can
  never hit the robot's own legs or the floor.
- New dynamics-only failure mode: **falls** (terminate, `info["fell"]`, penalty).
- **Speed: ~20× realtime** (100 env steps = 4 s sim in 0.2 s wall) — 10-seed sweeps are
  cheap; the feared full-dynamics slowdown did not materialize.

## Verification episode (`nav_episode.gif`, `fig_nav_episode.png`)
Seed 7, shift pulled forward to t=8 s so it's visible: the A* teacher navigates the
BASE field, the obstacle count doubles mid-run (orange discs), and it re-plans through
the densified field to the goal — **REACHED at t=28.5 s** (post-shift phase). Min-lidar
drops after the shift (denser field), the 13–19 s plateau is a defensive yield to a
crossing obstacle.

**Bug found & fixed via these figures** (see debug-log
[astar-freeze-after-shift](../../docs/debug-log/2026-07-17_astar-freeze-after-shift.md)):
the teacher froze permanently post-shift — the inflated occupancy (82 % blocked)
disconnected free space, A* returned no path, and "no path → brake" deadlocked.
Fix: **progressive inflation fallback** (0.9 → 0.7 → 0.55 m) in `PrivilegedExpert`;
floor at 0.55 m because 0.45 m (= the bare bounding circle) collided — a trotting base
sways and turns lag, so clearance must exceed the static footprint.

## Demo collection on dynamics (`scripts/collect_expert_go2.py`)
Base distribution (shift disabled), 60 episodes, seed 1000+, wall time 187 s:

| outcome | n | note |
|---|---|---|
| success (clean) | **37 (62 %)** | all demos drawn from these |
| collided-dynamic | 13 | crossing obstacles; known-hard for a speed-capped discrete robot |
| collided-static | 6 | turn lag / tight corridors under dynamics |
| fell | **0** | the RL walker never went down |
| timeout | 4 | |

**15,131 clean imitation steps** → `data/imitation_go2.npz` (bar ≥ 5,000; kinematic
M1b had 13k). Success is lower than the kinematic teacher's — real dynamics (turn lag,
gait sway, slower effective speed) are simply harder; only clean episodes are kept, so
demo quality is unaffected.

## Reproduce (Windows, conda nmc)
```
python scripts/render_go2_nav.py --seed 7 --shift-time 8   # verification episode
python scripts/collect_expert_go2.py --episodes 60         # demo collection
```
