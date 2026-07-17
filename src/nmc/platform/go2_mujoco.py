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
