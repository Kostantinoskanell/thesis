"""Unitree Go2 in MuJoCo with FULL rigid-body dynamics (platform track, new engine).

Decided 2026-07-17: the whole thesis moves to MuJoCo + the official Go2 model with
real dynamics (the user has a Go2 in the lab; kinematic shortcut retired). Two layers:
  * a low-level locomotion controller turns [vx, vy, omega] into stable walking, and
  * (later) the SNN navigator commands those velocities.

The Go2 Menagerie model uses TORQUE actuators (ctrlrange ~±23.7 Nm, knees ±45 Nm), so
even standing needs an active PD controller; robust walking needs the pretrained RL
policy (next milestone). This class provides: model loading, a PD stand controller (the
verified foundation), a base-state readout, and an offscreen renderer for GIFs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco

_MODEL = (Path(__file__).resolve().parents[3]
          / "assets" / "mujoco_menagerie" / "unitree_go2" / "scene.xml")


@dataclass
class Go2Config:
    model_path: str = str(_MODEL)
    kp: float = 60.0             # PD stiffness for stance hold
    kd: float = 3.0             # PD damping
    control_decimation: int = 10  # policy/PD ticks : physics steps (dt_ctrl = 10*dt_phys)
    # CPG trot gait (interim locomotion layer; PD tracks these joint targets).
    gait_freq_hz: float = 2.0
    sweep_amp: float = 0.18      # thigh fore/aft
    lift_amp: float = 0.45       # knee lift during swing
    turn_amp: float = 0.12       # differential thigh sweep for turning
    hip_turn: float = 0.06       # ab/ad hip bias for turning


class Go2MuJoCo:
    def __init__(self, cfg: Go2Config | None = None):
        self.cfg = cfg or Go2Config()
        self.model = mujoco.MjModel.from_xml_path(self.cfg.model_path)
        self.data = mujoco.MjData(self.model)
        self.nu = self.model.nu
        self._renderer = None

        # Map each actuator -> (qpos addr, qvel addr) of the joint it drives, and
        # record the 'home' keyframe joint angle as the PD setpoint.
        self.q_adr = np.zeros(self.nu, dtype=int)
        self.d_adr = np.zeros(self.nu, dtype=int)
        for i in range(self.nu):
            jid = self.model.actuator_trnid[i, 0]
            self.q_adr[i] = self.model.jnt_qposadr[jid]
            self.d_adr[i] = self.model.jnt_dofadr[jid]
        self.home = self.model.key_qpos[0].copy()
        self.q_stand = self.home[self.q_adr]
        self.ctrl_range = self.model.actuator_ctrlrange.copy()
        self.dt_ctrl = self.model.opt.timestep * self.cfg.control_decimation
        self.phi = 0.0
        self.cmd = np.zeros(3)

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        return self.base_state()

    def _pd(self, q_des):
        q = self.data.qpos[self.q_adr]
        qd = self.data.qvel[self.d_adr]
        tau = self.cfg.kp * (q_des - q) - self.cfg.kd * qd
        return np.clip(tau, self.ctrl_range[:, 0], self.ctrl_range[:, 1])

    def stand_step(self):
        """One control tick holding the home stance (decimated physics steps)."""
        tau = self._pd(self.q_stand)
        self.data.ctrl[:] = tau
        for _ in range(self.cfg.control_decimation):
            mujoco.mj_step(self.model, self.data)
        return self.base_state()

    def set_command(self, vx: float, vy: float = 0.0, omega: float = 0.0):
        self.cmd[:] = [vx, vy, omega]

    def walk_step(self):
        """One control tick of the CPG trot (PD tracks the gait joint targets).

        Actuator order: [FL,FR,RL,RR] x [hip,thigh,calf]. Trot = diagonal pairs
        (FL,RR) and (FR,RL) in antiphase. Interim locomotion layer until a proper
        RL/MPC controller is dropped in (see ROADMAP D2)."""
        c = self.cfg
        vx, _, omega = self.cmd
        moving = abs(vx) + abs(omega) > 1e-3
        targets = self.q_stand.copy()
        if moving:
            self.phi = (self.phi + c.gait_freq_hz * self.dt_ctrl) % 1.0
            leg_phase = [0.0, 0.5, 0.5, 0.0]        # FL, FR, RL, RR
            speed = float(np.clip(abs(vx), 0.0, 1.0))
            sgn = np.sign(vx) if vx != 0 else 1.0
            for i in range(self.nu):
                leg, part = i // 3, i % 3
                ph = 2 * np.pi * (self.phi + leg_phase[leg])
                right = leg in (1, 3)               # FR, RR
                tside = -1.0 if right else 1.0       # right legs lead a CCW (left) turn
                if part == 1:                        # thigh: fore/aft sweep
                    sweep = c.sweep_amp * speed * sgn * np.cos(ph)
                    sweep += c.turn_amp * omega * (-tside) * np.cos(ph)
                    targets[i] = self.q_stand[i] - sweep
                elif part == 2:                      # calf: lift during swing half
                    targets[i] = self.q_stand[i] + c.lift_amp * max(0.0, np.sin(ph)) * (0.4 + 0.6 * speed)
                else:                                # hip: small ab/ad turn bias
                    targets[i] = self.q_stand[i] + c.hip_turn * omega * (-tside)
        self.data.ctrl[:] = self._pd(targets)
        for _ in range(c.control_decimation):
            mujoco.mj_step(self.model, self.data)
        return self.base_state()

    def base_state(self):
        pos = self.data.qpos[:3].copy()
        quat = self.data.qpos[3:7].copy()            # w, x, y, z
        vel = self.data.qvel[:3].copy()
        # yaw from quaternion
        w, x, y, z = quat
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return dict(pos=pos, quat=quat, yaw=float(yaw),
                    v=float(np.hypot(vel[0], vel[1])), height=float(pos[2]))

    # -- rendering -------------------------------------------------------
    def render(self, w=480, h=360, cam_dist=1.6, azimuth=120, elevation=-20):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = self.data.qpos[:3]
        cam.distance = cam_dist
        cam.azimuth = azimuth
        cam.elevation = elevation
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
