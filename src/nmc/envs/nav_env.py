"""Differential-drive obstacle-avoidance environment with mid-episode shift.

Gymnasium-style API over PyBullet (proposal Sec. 4.1-4.2).  The defining
feature is `shift_time_s`: at that moment the obstacle density doubles and new
dynamic obstacles are injected *without resetting the episode*, which is the
distribution shift the whole thesis measures recovery from.

Observation (flat, non-spatial -> justifies MLP over CNN, proposal Sec. 3.1):
    [ lidar(32) normalized to [0,1], goal_dx, goal_dy, heading_err, v, omega ]

Action space: 4 discrete {forward, left, right, brake}.

NOTE: PyBullet is imported lazily so the rest of the package (encoders,
plasticity, metrics) can be unit-tested without a physics backend installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NavConfig:
    arena_size_m: float = 10.0
    n_lidar_beams: int = 32
    lidar_max_range_m: float = 8.0
    n_static_obstacles: int = 8
    n_dynamic_obstacles: int = 3
    # Capped below the robot's 0.6 m/s forward speed so evasion is always
    # kinematically possible (a faster obstacle can catch the robot from behind,
    # making some collisions unavoidable for ANY controller and compressing the
    # inter-controller differences the thesis measures). See debug-log entry.
    dynamic_speed_range: tuple[float, float] = (0.2, 0.5)
    episode_len_s: float = 60.0
    shift_time_s: float = 30.0          # distribution shift injected here
    control_hz: float = 20.0
    goal_radius_m: float = 0.5
    collision_penalty: float = -1.0     # r=-1 on collision  -> R-STDP reward
    progress_reward_scale: float = 1.0  # shaping: progress toward goal
    seed: int = 0
    gui: bool = False
    sensor_noise_std: float = 0.0       # additive Gaussian on lidar (robustness sweep)


# Discrete action -> (linear v [m/s], angular omega [rad/s])
ACTIONS = {
    0: (0.6, 0.0),    # forward
    1: (0.3, 1.5),    # left
    2: (0.3, -1.5),   # right
    3: (0.0, 0.0),    # brake
}


class NavEnv:
    def __init__(self, cfg: NavConfig | None = None):
        self.cfg = cfg or NavConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._p = None            # pybullet client, created on reset()
        self.robot = None
        self.obstacles: list[int] = []
        self.dynamic: list[dict] = []
        self.goal = np.zeros(2)
        self.t = 0.0
        self._shifted = False
        self._prev_goal_dist = None
        self.last_collision_kind = None
        self.dt = 1.0 / self.cfg.control_hz

    # -- lifecycle -------------------------------------------------------
    def _connect(self):
        import pybullet as p
        import pybullet_data
        if self._p is None:
            mode = p.GUI if self.cfg.gui else p.DIRECT
            self._client = p.connect(mode)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self._p = p

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._connect()
        p = self._p
        p.resetSimulation()
        p.setGravity(0, 0, 0)          # top-down 2D navigation; kinematic unicycle
        p.setTimeStep(self.dt)
        self.plane_id = p.loadURDF("plane.urdf")
        self._occupied = []            # (xy, radius) list to keep spawns clear
        self._build_walls()
        self.robot = self._spawn_robot()
        self.obstacles = self._spawn_static(self.cfg.n_static_obstacles)
        self.dynamic = self._spawn_dynamic(self.cfg.n_dynamic_obstacles)
        self.goal = self._sample_goal()
        self.t = 0.0
        self._shifted = False
        self._prev_goal_dist = self._goal_distance()
        return self._observation(), {}

    def step(self, action: int):
        p = self._p
        v, omega = ACTIONS[int(action)]
        self._apply_diff_drive(v, omega)
        self._advance_dynamic()
        p.stepSimulation()
        self.t += self.dt

        # Mid-episode distribution shift (the core experiment).
        if (not self._shifted) and self.t >= self.cfg.shift_time_s:
            self._inject_shift()
            self._shifted = True

        obs = self._observation()
        collided = self._check_collision()
        goal_dist = self._goal_distance()
        reached = goal_dist < self.cfg.goal_radius_m

        # Reward = collision penalty + progress shaping.  This is exactly the
        # environment-derived signal broadcast to R-STDP and used as the TD
        # target for the online-MLP baseline (proposal Sec. 3.1).
        progress = (self._prev_goal_dist - goal_dist) * self.cfg.progress_reward_scale
        self._prev_goal_dist = goal_dist
        reward = progress + (self.cfg.collision_penalty if collided else 0.0)

        terminated = bool(collided or reached)
        truncated = self.t >= self.cfg.episode_len_s
        info = {"collision": collided, "reached": reached,
                "collision_kind": self.last_collision_kind,
                "phase": 2 if self._shifted else 1, "t": self.t}
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._p is not None:
            self._p.disconnect()
            self._p = None

    # -- observation -----------------------------------------------------
    def _observation(self) -> np.ndarray:
        lidar = self._raycast_lidar()
        if self.cfg.sensor_noise_std > 0:
            lidar = lidar + self.rng.normal(0, self.cfg.sensor_noise_std, lidar.shape)
            lidar = np.clip(lidar, 0.0, 1.0)
        pos, yaw = self._robot_pose()
        goal_vec = self.goal - pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_ang = np.arctan2(goal_vec[1], goal_vec[0]) - yaw
        heading_err = np.arctan2(np.sin(goal_ang), np.cos(goal_ang))
        v, omega = self._robot_velocity()
        extra = np.array([goal_vec[0], goal_vec[1], heading_err, v, omega], dtype=np.float32)
        return np.concatenate([lidar.astype(np.float32), extra])

    @property
    def obs_dim(self) -> int:
        return self.cfg.n_lidar_beams + 5

    @property
    def n_actions(self) -> int:
        return len(ACTIONS)

    # -- geometry / spawning ---------------------------------------------
    # Bodies use simple primitives and a kinematic unicycle model (motion set
    # via base velocity under zero gravity).  Swap in a proper wheeled URDF
    # once the control loop and comparisons are validated.
    ARENA_MARGIN = 0.6
    ROBOT_RADIUS = 0.25
    OBST_RADIUS = 0.4
    BODY_H = 0.25          # half-height of obstacle/robot bodies

    def _build_walls(self):
        p = self._p
        s = self.cfg.arena_size_m / 2.0
        th, h = 0.1, 0.5
        specs = [((0, s, h / 2), (s, th, h / 2)), ((0, -s, h / 2), (s, th, h / 2)),
                 ((s, 0, h / 2), (th, s, h / 2)), ((-s, 0, h / 2), (th, s, h / 2))]
        self.wall_ids = []
        for pos, half in specs:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half))
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half), rgbaColor=[0.55, 0.55, 0.6, 1])
            self.wall_ids.append(p.createMultiBody(0, col, vis, basePosition=list(pos)))
        return self.wall_ids

    def _spawn_robot(self):
        p = self._p
        xy = self._sample_free_point(self.ROBOT_RADIUS)
        self.start_pos = np.array(xy)
        half = [self.ROBOT_RADIUS] * 3
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.1, 0.7, 0.2, 1])
        yaw = self.rng.uniform(0, 2 * np.pi)
        quat = p.getQuaternionFromEuler([0, 0, yaw])
        return p.createMultiBody(1.0, col, vis, basePosition=[xy[0], xy[1], self.ROBOT_RADIUS],
                                 baseOrientation=quat)

    def _make_obstacle(self, xy, rgba):
        p = self._p
        if self.rng.random() < 0.5:
            half = [self.OBST_RADIUS, self.OBST_RADIUS, self.BODY_H]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
        else:
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.OBST_RADIUS, height=2 * self.BODY_H)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=self.OBST_RADIUS, length=2 * self.BODY_H, rgbaColor=rgba)
        return col, vis

    def _spawn_static(self, n):
        p = self._p
        out = []
        for _ in range(n):
            xy = self._sample_free_point(self.OBST_RADIUS)
            col, vis = self._make_obstacle(xy, [0.2, 0.4, 0.8, 1])
            out.append(p.createMultiBody(0, col, vis, basePosition=[xy[0], xy[1], self.BODY_H]))
        return out

    def _spawn_dynamic(self, n):
        p = self._p
        out = []
        for _ in range(n):
            xy = self._sample_free_point(self.OBST_RADIUS)
            col, vis = self._make_obstacle(xy, [0.9, 0.5, 0.1, 1])
            bid = p.createMultiBody(1.0, col, vis, basePosition=[xy[0], xy[1], self.BODY_H])
            ang = self.rng.uniform(0, 2 * np.pi)
            spd = self.rng.uniform(*self.cfg.dynamic_speed_range)
            out.append({"id": bid, "vel": [spd * np.cos(ang), spd * np.sin(ang)]})
        return out

    def _inject_shift(self):
        # Double static density + add dynamic obstacles, mid-episode.
        self.obstacles += self._spawn_static(self.cfg.n_static_obstacles)
        self.dynamic += self._spawn_dynamic(self.cfg.n_dynamic_obstacles)

    def _sample_free_point(self, radius: float = 0.4, max_tries: int = 300) -> np.ndarray:
        s = self.cfg.arena_size_m / 2.0 - self.ARENA_MARGIN
        xy = self.rng.uniform(-s, s, size=2)
        for _ in range(max_tries):
            xy = self.rng.uniform(-s, s, size=2)
            if all(np.linalg.norm(xy - o) > radius + orad + 0.3 for o, orad in self._occupied):
                break
        self._occupied.append((xy, radius))
        return xy

    def _sample_goal(self, max_tries: int = 300) -> np.ndarray:
        s = self.cfg.arena_size_m / 2.0 - self.ARENA_MARGIN
        min_dist = self.cfg.arena_size_m * 0.4
        xy = self.rng.uniform(-s, s, size=2)
        for _ in range(max_tries):
            xy = self.rng.uniform(-s, s, size=2)
            clear = all(np.linalg.norm(xy - o) > self.cfg.goal_radius_m + orad + 0.3
                        for o, orad in self._occupied)
            if clear and np.linalg.norm(xy - self.start_pos) >= min_dist:
                break
        return xy

    def _raycast_lidar(self) -> np.ndarray:
        """n_lidar_beams distances normalized to [0,1] (1 = max range)."""
        p = self._p
        pos, yaw = self._robot_pose()
        n, rng = self.cfg.n_lidar_beams, self.cfg.lidar_max_range_m
        angles = yaw + np.linspace(-np.pi, np.pi, n, endpoint=False)
        offset = self.ROBOT_RADIUS + 0.05
        z = self.ROBOT_RADIUS
        froms = [[pos[0] + offset * np.cos(a), pos[1] + offset * np.sin(a), z] for a in angles]
        tos = [[pos[0] + rng * np.cos(a), pos[1] + rng * np.sin(a), z] for a in angles]
        res = p.rayTestBatch(froms, tos)
        span = rng - offset
        dists = np.array([rng if r[0] < 0 else offset + r[2] * span for r in res])
        return np.clip(dists / rng, 0.0, 1.0)

    def _apply_diff_drive(self, v, omega):
        _, yaw = self._robot_pose()
        self._p.resetBaseVelocity(self.robot,
                                  linearVelocity=[v * np.cos(yaw), v * np.sin(yaw), 0.0],
                                  angularVelocity=[0.0, 0.0, omega])

    def _advance_dynamic(self):
        p = self._p
        s = self.cfg.arena_size_m / 2.0 - self.ARENA_MARGIN
        for d in self.dynamic:
            (x, y, _), _ = p.getBasePositionAndOrientation(d["id"])
            vx, vy = d["vel"]
            if abs(x) > s:
                vx = -abs(vx) if x > 0 else abs(vx)
            if abs(y) > s:
                vy = -abs(vy) if y > 0 else abs(vy)
            d["vel"] = [vx, vy]
            p.resetBaseVelocity(d["id"], linearVelocity=[vx, vy, 0.0])

    def _robot_pose(self):
        pos, orn = self._p.getBasePositionAndOrientation(self.robot)
        yaw = self._p.getEulerFromQuaternion(orn)[2]
        return np.array([pos[0], pos[1]]), yaw

    def _robot_velocity(self):
        lin, ang = self._p.getBaseVelocity(self.robot)
        return float(np.hypot(lin[0], lin[1])), float(ang[2])

    def _check_collision(self) -> bool:
        static_b = set(self.wall_ids) | set(self.obstacles)
        dyn_b = {d["id"] for d in self.dynamic}
        self.last_collision_kind = None
        for c in self._p.getContactPoints(bodyA=self.robot):
            b = c[2]             # c[2] = bodyUniqueIdB
            if b in dyn_b:
                self.last_collision_kind = "dynamic"
                return True
            if b in static_b:
                self.last_collision_kind = "static"
                return True
        return False

    def _goal_distance(self) -> float:
        pos, _ = self._robot_pose()
        return float(np.linalg.norm(self.goal - pos))

    # -- privileged state (for the teacher/expert only, never the SNN/MLP) ----
    def privileged_state(self):
        """Ground-truth pose, goal, and obstacle discs. Used by the A* teacher to
        generate imitation demonstrations; the learned controllers never see this."""
        p = self._p
        pos, yaw = self._robot_pose()
        obst = []  # (x, y, radius, vx, vy) — static have zero velocity
        for bid in self.obstacles:
            (x, y, _), _ = p.getBasePositionAndOrientation(bid)
            obst.append((x, y, self.OBST_RADIUS, 0.0, 0.0))
        for d in self.dynamic:
            (x, y, _), _ = p.getBasePositionAndOrientation(d["id"])
            obst.append((x, y, self.OBST_RADIUS, d["vel"][0], d["vel"][1]))
        return pos, yaw, self.goal.copy(), obst
