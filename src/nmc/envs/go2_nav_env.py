"""Go2 obstacle-avoidance navigation env with mid-episode shift -- FULL DYNAMICS.

D3: the dynamic replacement for the kinematic nav_env.py. Same Gymnasium-style
API, same 37-dim observation [lidar(32) in [0,1], goal_dx, goal_dy, heading_err,
v, omega], same 4 discrete actions, same privileged_state() for the A* teacher --
so M2+ controllers and PrivilegedExpert plug in unchanged. Underneath, the robot
is the real MuJoCo Go2 walked by the trained RL policy (Go2RLWalker); discrete
actions map to velocity commands the policy tracks.

MuJoCo cannot add bodies mid-episode, so the distribution shift pre-allocates
the shift set of obstacles as mocap bodies PARKED far outside the arena and
teleports them in at shift_time_s. All obstacles are kinematic mocap cylinders
(matches the planner's disc model; dynamic ones move by mocap_pos updates and
bounce off walls). LiDAR is mj_ray filtered to geom group 4 (obstacles + walls
only), so rays never hit the robot's own legs or the floor.

New dynamics-only failure mode vs the kinematic env: the robot can FALL
(collision with a moving obstacle, aggressive turns). Falls terminate the
episode (info["fell"]) and count as failures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

from nmc.platform.go2_rl_walker import Go2RLWalker, CTRL_DT


@dataclass
class Go2NavConfig:
    arena_size_m: float = 10.0
    n_lidar_beams: int = 32
    lidar_max_range_m: float = 8.0
    n_static_obstacles: int = 8      # base distribution; same count injected at shift
    n_dynamic_obstacles: int = 3
    # Go2's effective speed under the RL policy is ~0.5 m/s (cmd 0.6); cap moving
    # obstacles below that so evasion stays kinematically possible (same reasoning
    # as the kinematic env -- see debug-log 2026-07-17_expert-tuning-collisions).
    dynamic_speed_range: tuple[float, float] = (0.15, 0.4)
    episode_len_s: float = 60.0
    shift_time_s: float = 30.0       # distribution shift injected here
    control_hz: float = 25.0         # env decision rate; 2 walker ticks per step
    goal_radius_m: float = 0.6       # slightly wider than kinematic (0.5): Go2 base
    collision_penalty: float = -1.0  #   center sways ~5cm while trotting
    progress_reward_scale: float = 1.0
    fall_penalty: float = -1.0       # falling is as bad as colliding
    seed: int = 0
    sensor_noise_std: float = 0.0    # additive Gaussian on lidar (robustness sweep)

    # -- distribution-shift type (M4). The mid-episode shift can be one of:
    #   "obstacles" : density doubling (weak -- mapping stays valid, just harder)
    #   "sensor"    : a contiguous block of LiDAR beams fails (dead = max range) --
    #                 corrupts the navigator's INPUT so the pretrained mapping is
    #                 wrong (strong test; the proposal's "sensor degradation")
    #   "terrain"   : floor friction changes road->ice/sand -- a DYNAMICS shift the
    #                 navigator sees only via its v/omega channels (medium test;
    #                 precedent: Juarez-Lora 2022 changing-friction R-STDP)
    shift_type: str = "obstacles"
    sensor_dropout_frac: float = 0.35   # fraction of contiguous beams killed at shift
    sensor_dropout_start: int = -1      # first dead beam index; -1 = random per episode
    terrain_mode: str = "ice"           # "ice" (low friction) | "sand" (high friction)
    terrain_friction: dict = None       # {"ice":0.08,"sand":1.6}; None -> defaults


# Discrete action -> velocity command [vx, vy, omega] for the RL walker.
# Turn rate 1.0 rad/s stays inside the policy's training command range (|w|<=1.2).
ACTIONS = {
    0: (0.6, 0.0, 0.0),    # forward
    1: (0.3, 0.0, 1.0),    # left
    2: (0.3, 0.0, -1.0),   # right
    3: (0.0, 0.0, 0.0),    # brake
}

OBST_RADIUS = 0.4
OBST_HALF_H = 0.25
_PARK_X = 30.0             # parking row for not-yet-injected / unused obstacles


def _arena_xml(cfg: Go2NavConfig, n_static_max: int, n_dyn_max: int) -> str:
    """Scene XML: Go2 + floor + arena walls + pre-allocated mocap obstacles +
    goal marker. Obstacle/wall geoms live in group 4 (the LiDAR ray filter);
    obstacles hover 1cm above the floor so parked ones generate no contacts."""
    s = cfg.arena_size_m / 2.0
    th, wh = 0.1, 0.25     # wall thickness / half-height
    parts = [f"""
<mujoco model="go2 nav arena">
  <include file="go2_playground.xml"/>
  <statistic center="0 0 0.1" extent="0.8" meansize="0.04"/>
  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <global azimuth="120" elevation="-20"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.92 0.92 0.92" rgb2="0.86 0.86 0.86" markrgb="0.7 0.7 0.7" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0"/>
  </asset>
  <worldbody>
    <geom name="floor" size="0 0 0.01" type="plane" material="groundplane" contype="1" conaffinity="0" priority="1" friction="0.6" condim="3"/>
    <geom name="wall_n" type="box" size="{s} {th} {wh}" pos="0 {s} {wh}" rgba="0.55 0.55 0.6 1" group="4" contype="1" conaffinity="1"/>
    <geom name="wall_s" type="box" size="{s} {th} {wh}" pos="0 -{s} {wh}" rgba="0.55 0.55 0.6 1" group="4" contype="1" conaffinity="1"/>
    <geom name="wall_e" type="box" size="{th} {s} {wh}" pos="{s} 0 {wh}" rgba="0.55 0.55 0.6 1" group="4" contype="1" conaffinity="1"/>
    <geom name="wall_w" type="box" size="{th} {s} {wh}" pos="-{s} 0 {wh}" rgba="0.55 0.55 0.6 1" group="4" contype="1" conaffinity="1"/>
    <body name="goal_marker" mocap="true" pos="{_PARK_X} -4 0.05">
      <geom type="cylinder" size="0.35 0.02" rgba="0.1 0.85 0.2 0.6" contype="0" conaffinity="0" group="1"/>
    </body>"""]
    for i in range(n_static_max):
        parts.append(f"""
    <body name="obst_s{i}" mocap="true" pos="{_PARK_X + 2 * i} 0 {OBST_HALF_H + 0.01}">
      <geom name="obst_s{i}_g" type="cylinder" size="{OBST_RADIUS} {OBST_HALF_H}" rgba="0.2 0.4 0.8 1" group="4" contype="1" conaffinity="1"/>
    </body>""")
    for i in range(n_dyn_max):
        parts.append(f"""
    <body name="obst_d{i}" mocap="true" pos="{_PARK_X + 2 * i} 4 {OBST_HALF_H + 0.01}">
      <geom name="obst_d{i}_g" type="cylinder" size="{OBST_RADIUS} {OBST_HALF_H}" rgba="0.9 0.5 0.1 1" group="4" contype="1" conaffinity="1"/>
    </body>""")
    parts.append("""
  </worldbody>
</mujoco>""")
    return "".join(parts)


class Go2NavEnv:
    ARENA_MARGIN = 0.8
    ROBOT_RADIUS = 0.45     # Go2 bounding circle (0.7 m long x 0.4 m wide body)

    def __init__(self, cfg: Go2NavConfig | None = None):
        self.cfg = cfg or Go2NavConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.dt = 1.0 / self.cfg.control_hz
        self._ticks_per_step = int(round(self.dt / CTRL_DT))
        assert abs(self._ticks_per_step * CTRL_DT - self.dt) < 1e-9, \
            "control_hz must be an integer multiple of the walker's 50 Hz tick"

        # Pre-allocate both the base and the shift obstacle sets.
        self._n_s_max = 2 * self.cfg.n_static_obstacles
        self._n_d_max = 2 * self.cfg.n_dynamic_obstacles
        xml = _arena_xml(self.cfg, self._n_s_max, self._n_d_max)
        self.bot = Go2RLWalker(scene_xml=xml)
        self.model, self.data = self.bot.model, self.bot.data

        m = self.model
        self._mocap = {}    # name -> mocap index
        for i in range(self._n_s_max):
            self._mocap[f"s{i}"] = m.body(f"obst_s{i}").mocapid[0]
        for i in range(self._n_d_max):
            self._mocap[f"d{i}"] = m.body(f"obst_d{i}").mocapid[0]
        self._goal_mocap = m.body("goal_marker").mocapid[0]

        # Obstacle/wall geom ids for collision checks; robot geom ids = the rest
        # (minus floor, which the feet legitimately touch).
        self._hazard_geoms = set()
        for i in range(self._n_s_max):
            self._hazard_geoms.add(m.geom(f"obst_s{i}_g").id)
        for i in range(self._n_d_max):
            self._hazard_geoms.add(m.geom(f"obst_d{i}_g").id)
        for w in ("wall_n", "wall_s", "wall_e", "wall_w"):
            self._hazard_geoms.add(m.geom(w).id)
        self._floor_geom = m.geom("floor").id
        goal_geom = m.body("goal_marker").geomadr[0]
        self._robot_geoms = set(range(m.ngeom)) - self._hazard_geoms \
            - {self._floor_geom, goal_geom}

        # LiDAR: rays only see geom group 4 (obstacles + walls).
        self._raygroup = np.zeros(6, dtype=np.uint8)
        self._raygroup[4] = 1

        # Terrain-shift plumbing: give the floor higher contact priority so ITS
        # friction fully determines foot-ground contact (otherwise contact friction
        # = max(floor, foot) and lowering the floor alone wouldn't make it slippery).
        self.model.geom_priority[self._floor_geom] = 2
        self._floor_friction0 = self.model.geom_friction[self._floor_geom].copy()
        self._floor_mat = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "groundplane")
        self._floor_rgba0 = (self.model.mat_rgba[self._floor_mat].copy()
                             if self._floor_mat >= 0 else None)
        self._dead_beams = None          # bool mask (n_lidar,) of failed LiDAR beams

        self._active_s: list[str] = []   # names of obstacles currently in-arena
        self._active_d: list[dict] = []  # {"name", "vel"}
        self.goal = np.zeros(2)
        self.t = 0.0
        self._shifted = False
        self._prev_goal_dist = None
        self.last_collision_kind = None

    # -- lifecycle ---------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.bot.reset()
        # Restore any shift-mutated state to the base distribution.
        self.model.geom_friction[self._floor_geom] = self._floor_friction0
        if self._floor_mat >= 0:
            self.model.mat_rgba[self._floor_mat] = self._floor_rgba0
        self._dead_beams = None
        # Park everything.
        for i in range(self._n_s_max):
            self.data.mocap_pos[self._mocap[f"s{i}"]] = [_PARK_X + 2 * i, 0, OBST_HALF_H + 0.01]
        for i in range(self._n_d_max):
            self.data.mocap_pos[self._mocap[f"d{i}"]] = [_PARK_X + 2 * i, 4, OBST_HALF_H + 0.01]
        self._active_s, self._active_d = [], []
        self._next_s = self._next_d = 0
        self._occupied = []

        # Robot spawn (random free pose).
        xy = self._sample_free_point(self.ROBOT_RADIUS)
        yaw = self.rng.uniform(0, 2 * np.pi)
        q = self.data.qpos
        q[0:2] = xy
        q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.start_pos = xy.copy()
        mujoco.mj_forward(self.model, self.data)

        self._place_static(self.cfg.n_static_obstacles)
        self._place_dynamic(self.cfg.n_dynamic_obstacles)
        self.goal = self._sample_goal()
        self.data.mocap_pos[self._goal_mocap] = [self.goal[0], self.goal[1], 0.05]

        self.t = 0.0
        self._shifted = False
        self._prev_goal_dist = self._goal_distance()
        self.last_collision_kind = None
        return self._observation(), {}

    def step(self, action: int):
        vx, vy, om = ACTIONS[int(action)]
        self.bot.set_command(vx, vy, om)
        for _ in range(self._ticks_per_step):
            self._advance_dynamic(CTRL_DT)
            self.bot.step()
        self.t += self.dt

        if (not self._shifted) and self.t >= self.cfg.shift_time_s:
            self._inject_shift()
            self._shifted = True

        obs = self._observation()
        collided = self._check_collision()
        fell = self._check_fall()
        goal_dist = self._goal_distance()
        reached = goal_dist < self.cfg.goal_radius_m

        progress = (self._prev_goal_dist - goal_dist) * self.cfg.progress_reward_scale
        self._prev_goal_dist = goal_dist
        reward = progress \
            + (self.cfg.collision_penalty if collided else 0.0) \
            + (self.cfg.fall_penalty if fell else 0.0)

        terminated = bool(collided or fell or reached)
        truncated = self.t >= self.cfg.episode_len_s
        info = {"collision": collided, "reached": reached, "fell": fell,
                "collision_kind": self.last_collision_kind,
                "phase": 2 if self._shifted else 1, "t": self.t}
        return obs, reward, terminated, truncated, info

    def close(self):
        self.bot.close()

    # -- observation (identical layout to kinematic NavEnv) ----------------
    def _observation(self) -> np.ndarray:
        lidar = self._raycast_lidar()
        if self.cfg.sensor_noise_std > 0:
            lidar = lidar + self.rng.normal(0, self.cfg.sensor_noise_std, lidar.shape)
            lidar = np.clip(lidar, 0.0, 1.0)
        pos, yaw = self._robot_pose()
        goal_vec = self.goal - pos
        goal_ang = np.arctan2(goal_vec[1], goal_vec[0]) - yaw
        heading_err = np.arctan2(np.sin(goal_ang), np.cos(goal_ang))
        s = self.bot.base_state()
        extra = np.array([goal_vec[0], goal_vec[1], heading_err,
                          s["vx"], s["yaw_rate"]], dtype=np.float32)
        return np.concatenate([lidar.astype(np.float32), extra])

    @property
    def obs_dim(self) -> int:
        return self.cfg.n_lidar_beams + 5

    @property
    def n_actions(self) -> int:
        return len(ACTIONS)

    def _raycast_lidar(self) -> np.ndarray:
        pos, yaw = self._robot_pose()
        n, rng_m = self.cfg.n_lidar_beams, self.cfg.lidar_max_range_m
        angles = yaw + np.linspace(-np.pi, np.pi, n, endpoint=False)
        z = 0.2                      # below obstacle top (0.51), above ground
        geomid = np.zeros(1, dtype=np.int32)
        dists = np.empty(n)
        pnt = np.array([pos[0], pos[1], z])
        for i, a in enumerate(angles):
            vec = np.array([np.cos(a), np.sin(a), 0.0])
            d = mujoco.mj_ray(self.model, self.data, pnt, vec,
                              self._raygroup, 1, -1, geomid)
            dists[i] = rng_m if d < 0 else min(d, rng_m)
        out = np.clip(dists / rng_m, 0.0, 1.0)
        if self._dead_beams is not None:      # failed beams read max range ("nothing")
            out[self._dead_beams] = 1.0
        return out

    # -- obstacles ----------------------------------------------------------
    def _place_static(self, n):
        for _ in range(n):
            name = f"s{self._next_s}"; self._next_s += 1
            xy = self._sample_free_point(OBST_RADIUS)
            self.data.mocap_pos[self._mocap[name]] = [xy[0], xy[1], OBST_HALF_H + 0.01]
            self._active_s.append(name)

    def _place_dynamic(self, n):
        for _ in range(n):
            name = f"d{self._next_d}"; self._next_d += 1
            xy = self._sample_free_point(OBST_RADIUS)
            self.data.mocap_pos[self._mocap[name]] = [xy[0], xy[1], OBST_HALF_H + 0.01]
            ang = self.rng.uniform(0, 2 * np.pi)
            spd = self.rng.uniform(*self.cfg.dynamic_speed_range)
            self._active_d.append({"name": name,
                                   "vel": np.array([spd * np.cos(ang), spd * np.sin(ang)])})

    def _inject_shift(self):
        """Dispatch the mid-episode distribution shift by type (M4)."""
        t = self.cfg.shift_type
        if t == "obstacles":
            self._inject_obstacles()
        elif t == "sensor":
            self._inject_sensor()
        elif t == "terrain":
            self._inject_terrain()
        else:
            raise ValueError(f"unknown shift_type {t!r}")

    def _inject_obstacles(self):
        # Double static density + add moving obstacles, mid-episode, no reset.
        self._occupied = [(self.data.mocap_pos[self._mocap[n]][:2].copy(), OBST_RADIUS)
                          for n in self._active_s] \
            + [(self.data.mocap_pos[self._mocap[d["name"]]][:2].copy(), OBST_RADIUS)
               for d in self._active_d]
        pos, _ = self._robot_pose()
        self._occupied.append((pos, 1.2))   # don't spawn on top of the robot
        self._occupied.append((self.goal.copy(), 1.0))
        self._place_static(self.cfg.n_static_obstacles)
        self._place_dynamic(self.cfg.n_dynamic_obstacles)

    def _inject_sensor(self):
        """Fail a contiguous block of LiDAR beams (they read max range = 'nothing
        there'). Corrupts the navigator's input so its pretrained obs->action
        mapping is wrong -- the proposal's 'sensor degradation' shift. The block
        start is fixed per episode-block (deterministic given the env seed) so a
        plastic controller CAN learn the specific remapping across the block."""
        n = self.cfg.n_lidar_beams
        k = int(round(self.cfg.sensor_dropout_frac * n))
        start = (self.cfg.sensor_dropout_start
                 if self.cfg.sensor_dropout_start >= 0
                 else int(self.rng.integers(0, n)))
        mask = np.zeros(n, dtype=bool)
        mask[(start + np.arange(k)) % n] = True
        self._dead_beams = mask

    def _inject_terrain(self):
        """Change floor friction road->ice/sand (a dynamics shift). The RL walker
        was trained with friction ~U(0.4,1.0); ice (0.08) / sand (1.6) are outside
        that, so velocity tracking degrades and the navigator must compensate via
        its v/omega channels. Floor is recolored so the change is visible."""
        fr = self.cfg.terrain_friction or {"ice": 0.08, "sand": 1.6}
        mu = fr[self.cfg.terrain_mode]
        f = self.model.geom_friction[self._floor_geom].copy()
        f[0] = mu
        self.model.geom_friction[self._floor_geom] = f
        if self._floor_mat >= 0:
            self.model.mat_rgba[self._floor_mat] = (
                [0.70, 0.85, 1.0, 1.0] if self.cfg.terrain_mode == "ice"
                else [0.85, 0.72, 0.45, 1.0])

    def _advance_dynamic(self, dt: float):
        s = self.cfg.arena_size_m / 2.0 - self.ARENA_MARGIN
        for d in self._active_d:
            mid = self._mocap[d["name"]]
            p = self.data.mocap_pos[mid]
            if abs(p[0]) > s:
                d["vel"][0] = -abs(d["vel"][0]) if p[0] > 0 else abs(d["vel"][0])
            if abs(p[1]) > s:
                d["vel"][1] = -abs(d["vel"][1]) if p[1] > 0 else abs(d["vel"][1])
            self.data.mocap_pos[mid] = [p[0] + d["vel"][0] * dt,
                                        p[1] + d["vel"][1] * dt, p[2]]

    # -- spawning helpers ----------------------------------------------------
    def _sample_free_point(self, radius: float, max_tries: int = 300) -> np.ndarray:
        s = self.cfg.arena_size_m / 2.0 - self.ARENA_MARGIN
        xy = self.rng.uniform(-s, s, size=2)
        for _ in range(max_tries):
            xy = self.rng.uniform(-s, s, size=2)
            if all(np.linalg.norm(xy - o) > radius + orad + 0.35
                   for o, orad in self._occupied):
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

    # -- termination checks ----------------------------------------------------
    def _check_collision(self) -> bool:
        self.last_collision_kind = None
        dyn_geoms = {self.model.geom(f"obst_{d['name']}_g").id for d in self._active_d}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            hazard = robot = None
            if g1 in self._hazard_geoms and g2 in self._robot_geoms:
                hazard = g1
            elif g2 in self._hazard_geoms and g1 in self._robot_geoms:
                hazard = g2
            if hazard is not None:
                self.last_collision_kind = "dynamic" if hazard in dyn_geoms else "static"
                return True
        return False

    def _check_fall(self) -> bool:
        s = self.bot.base_state()
        up_z = self.data.site_xmat[self.bot._imu_site].reshape(3, 3)[2, 2]
        return bool(s["height"] < 0.15 or up_z < 0.3)

    def _goal_distance(self) -> float:
        pos, _ = self._robot_pose()
        return float(np.linalg.norm(self.goal - pos))

    def _robot_pose(self):
        s = self.bot.base_state()
        return s["pos"][:2], s["yaw"]

    # -- privileged state (for the A* teacher only, never the SNN/MLP) --------
    def privileged_state(self):
        """Ground-truth pose, goal, obstacle discs (x, y, r, vx, vy) -- identical
        contract to the kinematic env, so PrivilegedExpert works unchanged."""
        pos, yaw = self._robot_pose()
        obst = []
        for n in self._active_s:
            p = self.data.mocap_pos[self._mocap[n]]
            obst.append((float(p[0]), float(p[1]), OBST_RADIUS, 0.0, 0.0))
        for d in self._active_d:
            p = self.data.mocap_pos[self._mocap[d["name"]]]
            obst.append((float(p[0]), float(p[1]), OBST_RADIUS,
                         float(d["vel"][0]), float(d["vel"][1])))
        return pos, yaw, self.goal.copy(), obst

    # -- rendering --------------------------------------------------------------
    def render(self, **kw):
        return self.bot.render(**kw)
