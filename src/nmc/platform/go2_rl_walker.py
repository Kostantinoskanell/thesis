"""Go2 walking in the WINDOWS MuJoCo loop via the exported RL policy (NumPy).

This is the deployment side of the D2 policy: same Playground scene XML the
policy was trained on (src/nmc/rl/envs/go2/xmls/), same control scheme
(ctrl_dt=0.02 / sim_dt=0.004, motor targets = default_pose + action*0.5 into
the XML's baked-in PD position servos), and the same 48-dim observation the
env builds in joystick.py -- minus the training-time noise injection.

The obs construction here MUST mirror nmc.rl.envs.go2.joystick._get_obs
ordering exactly: [local_linvel(3), gyro(3), gravity(3), qpos-default(12),
qvel(12), last_act(12), command(3)]. The cross-runtime parity gate
(scripts/verify_policy_parity.py) verifies the network; this class is the
only remaining hand-written bridge, so keep it in lockstep with joystick.py.

Replaces the CPG trot in go2_mujoco.py as the locomotion layer the SNN
navigator (D3+) commands.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

from nmc.rl.numpy_policy import NumpyGo2Policy

_ROOT = Path(__file__).resolve().parents[3]
_XMLS = _ROOT / "src" / "nmc" / "rl" / "envs" / "go2" / "xmls"
_MENAGERIE_GO2 = _ROOT / "assets" / "mujoco_menagerie" / "unitree_go2"
_EXPORT = _ROOT / "assets" / "go2_policy_export.npz"

CTRL_DT = 0.02
SIM_DT = 0.004
N_SUBSTEPS = int(round(CTRL_DT / SIM_DT))
ACTION_SCALE = 0.5


def _load_model(scene_xml: str | None = None) -> mujoco.MjModel:
    """Load the Playground Go2 scene exactly as base.py does in WSL: XML strings
    + an assets dict (scene include + robot xml + menagerie meshes).

    scene_xml: optional scene XML *string* that includes go2_playground.xml
    (e.g. the D3 nav arena); defaults to the flat-terrain training scene."""
    assets = {}
    for f in _XMLS.glob("*.xml"):
        assets[f.name] = f.read_bytes()
    for f in (_MENAGERIE_GO2 / "assets").glob("*"):
        if f.is_file():
            assets[f.name] = f.read_bytes()
    if scene_xml is None:
        scene_xml = (_XMLS / "scene_go2_playground_flat.xml").read_text()
    model = mujoco.MjModel.from_xml_string(scene_xml, assets=assets)
    model.opt.timestep = SIM_DT
    return model


class Go2RLWalker:
    """Velocity-commanded Go2 (full dynamics) driven by the trained RL policy."""

    def __init__(self, export_path: str | Path = _EXPORT,
                 scene_xml: str | None = None):
        self.policy = NumpyGo2Policy(export_path)
        self.model = _load_model(scene_xml)
        self.data = mujoco.MjData(self.model)

        key = self.model.keyframe("home")
        self._init_qpos = key.qpos.copy()
        self.default_pose = key.qpos[7:].copy()

        self._imu_site = self.model.site("imu").id
        self._sens = {}
        for name in ("local_linvel", "gyro"):
            s = self.model.sensor(name)
            self._sens[name] = (s.adr[0], s.dim[0])

        self.cmd = np.zeros(3)
        self.last_act = np.zeros(self.policy.action_size)
        self._renderer = None
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data,
                                    self.model.keyframe("home").id)
        mujoco.mj_forward(self.model, self.data)
        self.last_act[:] = 0.0
        self.cmd[:] = 0.0
        return self.base_state()

    def set_command(self, vx: float, vy: float = 0.0, omega: float = 0.0):
        self.cmd[:] = [vx, vy, omega]

    # -- observation (mirrors joystick.py _get_obs, noise-free) ------------

    def _sensor(self, name: str) -> np.ndarray:
        adr, dim = self._sens[name]
        return self.data.sensordata[adr: adr + dim].copy()

    def _gravity(self) -> np.ndarray:
        rot = self.data.site_xmat[self._imu_site].reshape(3, 3)
        return rot.T @ np.array([0.0, 0.0, -1.0])

    def obs(self) -> np.ndarray:
        return np.hstack([
            self._sensor("local_linvel"),              # 3
            self._sensor("gyro"),                      # 3
            self._gravity(),                           # 3
            self.data.qpos[7:] - self.default_pose,    # 12
            self.data.qvel[6:],                        # 12
            self.last_act,                             # 12
            self.cmd,                                  # 3
        ])

    # -- control tick -------------------------------------------------------

    def step(self):
        """One 50 Hz policy tick (5 physics substeps). Returns base state."""
        act = self.policy(self.obs())
        self.data.ctrl[:] = self.default_pose + act * ACTION_SCALE
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self.last_act = act
        return self.base_state()

    def base_state(self) -> dict:
        q = self.data.qpos
        w, x, y, z = q[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        vworld = self.data.qvel[:3]
        vx_body = float(vworld[0] * np.cos(yaw) + vworld[1] * np.sin(yaw))
        return dict(pos=q[:3].copy(), yaw=float(yaw), height=float(q[2]),
                    vx=vx_body, yaw_rate=float(self.data.qvel[5]))

    # -- rendering -----------------------------------------------------------

    def render(self, w=480, h=360, cam_dist=1.8, azimuth=130, elevation=-20,
               show_collision=False):
        """Render one frame. show_collision=True overlays the collision geometry
        (robot collision capsules = geom group 3, obstacle/wall collision geoms =
        group 4) and contact points/forces, so you can see the actual collision
        boxes and where contacts happen -- useful for verifying the moving
        obstacles and near-misses."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = self.data.qpos[:3]
        cam.distance, cam.azimuth, cam.elevation = cam_dist, azimuth, elevation
        opt = None
        if show_collision:
            opt = mujoco.MjvOption()
            opt.geomgroup[3] = 1     # robot collision primitives (normally hidden)
            opt.geomgroup[4] = 1     # obstacle + wall collision geoms
            opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # see boxes through meshes
        self._renderer.update_scene(self.data, camera=cam, scene_option=opt)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
