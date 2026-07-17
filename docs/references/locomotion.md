# Quadruped Locomotion (Go2 Platform Track)

The SNN outputs high-level velocity commands `[vx, vy, omega]`; a low-level locomotion
layer realizes them. This file collects the options for that layer. **The layers are
decoupled** — the locomotion policy is for realistic sim visuals + a final integrated
demo; the plasticity science (M2–M8) runs on the fast kinematic model.

## Key architectural fact
The **real Unitree Go2 walks itself**: its onboard SDK "sport/high-level mode" accepts
`Move(vx, vy, vyaw)`. So on hardware we send velocity commands and never deploy a custom
gait. (unitree_sdk2 / unitree_sdk2_python.)

## Sim locomotion — PyBullet-native (recommended, keeps our stack)
- **yxyang/fast_and_efficient** — A1 in PyBullet + **convex MPC**, velocity-command
  interface. Best drop-in for our PyBullet env. https://github.com/yxyang/fast_and_efficient
- **ShuoYangRobotics/A1-QP-MPC-Controller** — MIT-Cheetah-style QP/MPC stack for A1.
  https://github.com/ShuoYangRobotics/A1-QP-MPC-Controller
- **silvery107/rl-mpc-locomotion** — MPC in Python with open sensor/motor interfaces,
  designed to port across simulators. https://github.com/silvery107/rl-mpc-locomotion
- **Built-in URDF:** `pybullet_data` ships `a1/a1.urdf` (Go2's predecessor) and
  `laikago/laikago.urdf` — prototype now, swap the Go2 URDF (Unitree public repos) later.

## Sim locomotion — Go2-specific / SOTA RL (if we want learned gaits or best sim2real)
- **go2-convex-mpc** — convex MPC for Go2 in MuJoCo (Raibert foot planning, swing paths).
  https://github.com/elijah-waichong-chan/go2-convex-mpc
- **unitree_rl_gym / Isaac Lab Go2** — PPO velocity-command policies with pretrained
  weights and 80–95% sim2real; the SOTA learned-locomotion path (heavier GPU stack).
  https://github.com/BrandoUlissi/isaaclab-go2-locomotion
- **Isaac Lab (2025)**, arXiv:2511.04831 — GPU sim framework, RSL-RL integration.

## Recommendation
Sim: **PyBullet + convex MPC** on the A1/Go2 URDF (single simulator, keeps M1/M1b).
Real: **Go2 SDK sport mode**. Only reach for Isaac/unitree_rl_gym if we specifically
want to *contribute* a learned locomotion policy — which is out of scope for a
neuromorphic *navigation* thesis.
