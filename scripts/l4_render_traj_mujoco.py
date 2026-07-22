"""Render a dumped Isaac-Lab Go2 trajectory as a GIF using the MuJoCo Go2 model.

Isaac Sim's RTX offscreen renderer won't initialize on this headless WSL box
(compute/PhysX works, graphics device can't be created). So we do a KINEMATIC
replay: take the base pose + joint angles logged from a headless Isaac rollout
and play them back on our MuJoCo Menagerie Go2 (which renders fine on Windows,
as in D2/D3). This visualizes exactly what the policy did in Isaac -- no physics
re-sim, just the recorded poses.

Isaac joint order is grouped by joint-type (FL_hip,FR_hip,RL_hip,RR_hip,FL_thigh
...); MuJoCo groups by leg (FL_hip,FL_thigh,FL_calf,FR_hip...). We remap by name.
Both use wxyz quaternions, so the base orientation transfers directly.

Run:  conda run -n nmc python scripts/l4_render_traj_mujoco.py data/l4_walk_traj.npz \
          archive/L4_gait_check/walk_crouch.gif
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import mujoco
from PIL import Image

from nmc.platform.go2_rl_walker import _load_model

traj_path = sys.argv[1] if len(sys.argv) > 1 else "data/l4_walk_traj.npz"
out_path = sys.argv[2] if len(sys.argv) > 2 else "archive/L4_gait_check/walk_crouch.gif"
max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 500
stride = int(sys.argv[4]) if len(sys.argv) > 4 else 2

d = np.load(traj_path, allow_pickle=True)
base_pos, base_quat, joint_pos = d["base_pos"], d["base_quat"], d["joint_pos"]
isaac_names = [str(x) for x in d["joint_names"]]

model = _load_model()
mj_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
mj_actuated = [n for n in mj_joint_names if n and n != "floating_base" and "_joint" in n]
remap = [isaac_names.index(n) for n in mj_actuated]  # mj order index -> isaac index
print(f"remap (mj->isaac): {remap}")

data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)
opt = mujoco.MjvOption()

frames = []
n = min(max_steps, base_pos.shape[0])
for t in range(0, n, stride):
    data.qpos[0:3] = base_pos[t]
    data.qpos[3:7] = base_quat[t]
    data.qpos[7:19] = joint_pos[t][remap]
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [base_pos[t][0], base_pos[t][1], 0.15]
    cam.distance, cam.azimuth, cam.elevation = 2.2, 130, -15
    renderer.update_scene(data, camera=cam, scene_option=opt)
    frames.append(Image.fromarray(renderer.render()))

renderer.close()
import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=40, loop=0)
print(f"wrote {out_path} ({len(frames)} frames)")
