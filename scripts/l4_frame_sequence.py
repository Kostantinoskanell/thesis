"""Render a handful of individual PNG frames at specific timesteps, to visually inspect
gait-cycle details (e.g. does a specific leg lift off the ground during its swing phase)
that a full GIF's first-frame thumbnail can't show."""
import sys
sys.path.insert(0, "src")

import numpy as np
import mujoco
from PIL import Image

from nmc.platform.go2_rl_walker import _load_model

traj_path = sys.argv[1]
out_prefix = sys.argv[2]
timesteps = [int(x) for x in sys.argv[3].split(",")]

d = np.load(traj_path, allow_pickle=True)
base_pos, base_quat, joint_pos = d["base_pos"], d["base_quat"], d["joint_pos"]
isaac_names = [str(x) for x in d["joint_names"]]

model = _load_model()
mj_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
mj_actuated = [n for n in mj_joint_names if n and n != "floating_base" and "_joint" in n]
remap = [isaac_names.index(n) for n in mj_actuated]

data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=300, width=400)
opt = mujoco.MjvOption()

for t in timesteps:
    data.qpos[0:3] = base_pos[t]
    data.qpos[3:7] = base_quat[t]
    data.qpos[7:19] = joint_pos[t][remap]
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [base_pos[t][0], base_pos[t][1], 0.15]
    cam.distance, cam.azimuth, cam.elevation = 2.0, 135, -25  # 3/4 rear view, all 4 legs visible
    renderer.update_scene(data, camera=cam, scene_option=opt)
    img = Image.fromarray(renderer.render())
    img.save(f"{out_prefix}_t{t}.png")
    print(f"wrote {out_prefix}_t{t}.png")
renderer.close()
