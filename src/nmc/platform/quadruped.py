"""CPG-driven quadruped locomotion for the Go2 platform track (P1).

A central pattern generator (CPG) produces a trot gait for the A1 URDF (Go2's
predecessor, shipped with pybullet_data; swap the Go2 URDF later). The CPG takes
a high-level velocity command [vx, omega] -- exactly the interface the SNN
navigator emits -- and modulates stride/turn. Using a CPG (rather than an MPC)
is deliberate: CPGs are a neuromorphic locomotion primitive, so the low-level
layer stays on-theme with the thesis.

Decoupled from the plasticity science: this is for realistic visuals + a final
integrated demo (ROADMAP "Platform track"). The science runs on the fast
kinematic model in nmc.envs.nav_env.

Trot = diagonal leg pairs in antiphase: (FR, RL) lead, (FL, RR) trail by half a
cycle. Each leg's thigh sweeps fore/aft; the knee lifts during the swing half.
Joint targets modulate the known-stable standing pose, so no per-leg IK sign
hunting is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaitConfig:
    stand_upper: float = 0.9        # standing thigh angle (rad)
    stand_lower: float = -1.8       # standing calf angle (rad)
    gait_freq_hz: float = 2.0       # trot cycle frequency
    sweep_amp: float = 0.20         # thigh fore/aft amplitude at full speed
    lift_amp: float = 0.30          # knee lift amplitude during swing
    turn_amp: float = 0.18          # differential sweep for turning
    hip_turn: float = 0.08          # ab/ad hip bias for turning
    kp: float = 90.0                # position-control gain (stiff -> holds posture)
    kd: float = 1.0
    max_force: float = 55.0
    control_hz: float = 50.0

    # Trot phase offsets (cycles) per leg: diagonal pairs in antiphase.
    phase: tuple = (0.0, 0.5, 0.5, 0.0)   # FR, FL, RR, RL


LEGS = ["FR", "FL", "RR", "RL"]


class QuadrupedLocomotion:
    """A1 quadruped in PyBullet, walked by a CPG from [vx, omega] commands."""

    def __init__(self, cfg: GaitConfig | None = None, gui: bool = False):
        self.cfg = cfg or GaitConfig()
        self.gui = gui
        self._p = None
        self.robot = None
        self.phi = 0.0                      # global gait phase (cycles)
        self.cmd = np.zeros(3)              # [vx, vy, omega]
        self.dt = 1.0 / self.cfg.control_hz
        self.joint_idx = {}                # (leg, part) -> joint index

    def connect(self):
        import pybullet as p
        import pybullet_data
        if self._p is None:
            self._client = p.connect(p.GUI if self.gui else p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self._p = p
        return self._p

    def reset(self, base_pos=(0, 0, 0.42), base_yaw=0.0):
        p = self.connect()
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.dt)
        self.plane = p.loadURDF("plane.urdf")
        quat = p.getQuaternionFromEuler([0, 0, base_yaw])
        self.robot = p.loadURDF("a1/a1.urdf", list(base_pos), quat, useFixedBase=False)
        self._index_joints()
        self.phi = 0.0
        self.cmd[:] = 0.0
        # settle into the stance so the first frames aren't a fall
        for _ in range(80):
            self._apply(self._stance_targets())
            p.stepSimulation()
        return self.base_state()

    def _index_joints(self):
        p = self._p
        self.joint_idx.clear()
        for j in range(p.getNumJoints(self.robot)):
            name = p.getJointInfo(self.robot, j)[1].decode()
            for leg in LEGS:
                if name == f"{leg}_hip_joint":
                    self.joint_idx[(leg, "hip")] = j
                elif name == f"{leg}_upper_joint":
                    self.joint_idx[(leg, "upper")] = j
                elif name == f"{leg}_lower_joint":
                    self.joint_idx[(leg, "lower")] = j

    def _stance_targets(self):
        c = self.cfg
        t = {}
        for leg in LEGS:
            t[(leg, "hip")] = 0.0
            t[(leg, "upper")] = c.stand_upper
            t[(leg, "lower")] = c.stand_lower
        return t

    def set_command(self, vx: float, vy: float = 0.0, omega: float = 0.0):
        self.cmd[:] = [vx, vy, omega]

    def _gait_targets(self):
        """CPG trot joint targets modulated by the current [vx, omega] command."""
        c = self.cfg
        vx, _, omega = self.cmd
        speed = np.clip(abs(vx), 0.0, 1.0)
        t = {}
        for i, leg in enumerate(LEGS):
            ph = 2 * np.pi * (self.phi + c.phase[i])
            # thigh sweeps fore/aft; sign of vx sets walking direction
            sweep = c.sweep_amp * speed * np.sign(vx if vx != 0 else 1.0) * np.cos(ph)
            # Differential sweep for turning: for a CCW (left, omega>0) turn the
            # RIGHT legs must travel further, so they get more sweep (side=-1).
            side = +1.0 if leg in ("FL", "RL") else -1.0
            sweep += c.turn_amp * omega * (-side) * np.cos(ph)
            # knee lifts only during the swing half (sin > 0)
            lift = c.lift_amp * max(0.0, np.sin(ph)) * (0.4 + 0.6 * speed)
            t[(leg, "upper")] = c.stand_upper - sweep
            t[(leg, "lower")] = c.stand_lower + lift
            t[(leg, "hip")] = c.hip_turn * omega * (-side)
        return t

    def _apply(self, targets):
        p, c = self._p, self.cfg
        for (leg, part), q in targets.items():
            p.setJointMotorControl2(self.robot, self.joint_idx[(leg, part)],
                                    p.POSITION_CONTROL, targetPosition=q,
                                    positionGain=c.kp / 100.0, velocityGain=c.kd,
                                    force=c.max_force)

    def step(self, n_sub: int = 1):
        """Advance the gait by one control tick (optionally n_sub physics steps)."""
        moving = np.linalg.norm(self.cmd) > 1e-3
        if moving:
            self.phi = (self.phi + self.cfg.gait_freq_hz * self.dt) % 1.0
        targets = self._gait_targets() if moving else self._stance_targets()
        for _ in range(n_sub):
            self._apply(targets)
            self._p.stepSimulation()
        return self.base_state()

    def base_state(self):
        p = self._p
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(orn)[2]
        lin, ang = p.getBaseVelocity(self.robot)
        return dict(pos=np.array(pos), yaw=yaw,
                    v=np.hypot(lin[0], lin[1]), omega=ang[2], up=orn)

    def close(self):
        if self._p is not None:
            self._p.disconnect()
            self._p = None
