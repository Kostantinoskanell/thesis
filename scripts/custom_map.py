"""A deliberately-designed test map, as opposed to the random procedural
layout M0-M5 use.

Motivation: the M4b-3 trajectory-overlay and M4c sand-GIF bugs both turned out
to be dynamic obstacles that had nothing to do with the shift under study,
adding pure outcome noise. A hand-built map fixes obstacle/robot/goal
positions completely -- the ONLY thing varying seed-to-seed is the
controller's own stochasticity (spike-encoding noise, plasticity), not map
difficulty -- which is a cleaner instrument for isolating a controller's
behavior. It also lets us target the sensor-dropout blind sector on purpose:
dead beams (start=8, frac=0.20-0.35) cover bearing offsets ~-90..-34 degrees
(the robot's RIGHT side, see scripts/m4b_traj_diag.py). This map forces a
RIGHT turn around the wall's gap, so the shift directly obscures the correct
detour instead of a random/unrelated part of the scene.

Layout (world frame, arena is 10x10 m, usable radius ~4.2 m after margin):
  robot start (-3.5, 0), facing +x (yaw=0)
  goal        ( 3.5, 0)
  static wall at x=-0.3, y in {-1,0,1,2,3,4} (6 cylinders, edge gap ~0.2m,
    solid) -- blocks the direct straight path and everything "ahead-left"
  open gap at the BOTTOM: y in [-4.2, -1.5] at x~=-0.3 -- reachable only by
    turning RIGHT from the start heading, i.e. through the dead-beam sector
    under the sensor-dropout shift
  one dynamic obstacle patrolling x=+0.3 (offset from the wall so it can't
    clip through it), bouncing along y between the arena edges -- crosses
    the gap periodically, so timing the crossing matters
"""
from __future__ import annotations
import numpy as np
import mujoco

from nmc.envs.go2_nav_env import OBST_HALF_H, _PARK_X

WALL_X = -0.3
WALL_YS = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
MOVER_X = 0.3
MOVER_Y0 = -2.5
MOVER_SPEED = 0.35
START_XY = np.array([-3.5, 0.0])
START_YAW = 0.0
GOAL_XY = np.array([3.5, 0.0])


def apply_custom_map(env, mover_vel=(0.0, 1.0)) -> None:
    """Call right after env.reset(seed=...). Overwrites the random layout
    reset() just placed with this fixed one, using the SAME active obstacle
    slots (so cfg.n_static_obstacles must be >= 6, n_dynamic_obstacles >= 1)."""
    if len(env._active_s) < len(WALL_YS):
        raise ValueError(f"need n_static_obstacles >= {len(WALL_YS)}, "
                         f"got {len(env._active_s)} active")
    if len(env._active_d) < 1:
        raise ValueError("need n_dynamic_obstacles >= 1")

    q = env.data.qpos
    q[0:2] = START_XY
    q[3:7] = [np.cos(START_YAW / 2), 0, 0, np.sin(START_YAW / 2)]
    env.start_pos = START_XY.copy()

    env.goal = GOAL_XY.copy()
    env.data.mocap_pos[env._goal_mocap] = [GOAL_XY[0], GOAL_XY[1], 0.05]

    for name, y in zip(env._active_s, WALL_YS):
        env.data.mocap_pos[env._mocap[name]] = [WALL_X, y, OBST_HALF_H + 0.01]
    for name in env._active_s[len(WALL_YS):]:
        i = int(name[1:])
        env.data.mocap_pos[env._mocap[name]] = [_PARK_X + 2 * i, 0, OBST_HALF_H + 0.01]

    d0 = env._active_d[0]
    env.data.mocap_pos[env._mocap[d0["name"]]] = [MOVER_X, MOVER_Y0, OBST_HALF_H + 0.01]
    spd = float(np.hypot(*mover_vel)) or MOVER_SPEED
    unit = np.array(mover_vel) / spd if spd else np.array([0.0, 1.0])
    d0["vel"] = unit * MOVER_SPEED
    for d in env._active_d[1:]:
        i = int(d["name"][1:])
        env.data.mocap_pos[env._mocap[d["name"]]] = [_PARK_X + 2 * i, 4, OBST_HALF_H + 0.01]

    mujoco.mj_forward(env.model, env.data)
    env._prev_goal_dist = env._goal_distance()


def make_custom_env(shift_type="sensor", **shift_kwargs):
    """Env sized correctly for this map (>=6 static, >=1 dynamic slots)."""
    from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
    cfg = Go2NavConfig(n_static_obstacles=6, n_dynamic_obstacles=1,
                       shift_type=shift_type, episode_len_s=60.0, **shift_kwargs)
    return Go2NavEnv(cfg)
