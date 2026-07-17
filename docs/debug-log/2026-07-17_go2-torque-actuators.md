# MuJoCo Go2 collapses when "holding" its home pose

_2026-07-17 · severity: low (foundation) · quick once the cause was found_

## Symptom
Loading the MuJoCo Go2 and setting `data.ctrl` to the home joint angles, then stepping,
the robot collapsed: base height 0.27 → 0.08 in 1000 steps.

## Cause
The Menagerie `go2.xml` uses **`<motor>` (torque) actuators** (`ctrlrange` ±23.7 Nm,
knees ±45 Nm), not position actuators. Writing joint *angles* (~0.9) into `ctrl` commands
~0.9 Nm of torque — negligible — so the legs fold under gravity.

## Fix
Active **PD control**: `tau = kp*(q_home − q) − kd*qvel`, clipped to the actuator
`ctrlrange`, recomputed every control tick. Map each actuator to its joint's qpos/qvel
address via `actuator_trnid → jnt_qposadr / jnt_dofadr` (actuator order ≠ qpos order).
With kp=60, kd=3 the Go2 holds 0.255–0.270 m indefinitely.

## Lesson
Always check `actuator gaintype`/type before commanding a MuJoCo robot — torque vs
position changes the meaning of `ctrl` entirely. A torque-actuated quadruped is never
passively stable; even standing is a closed-loop control problem (and walking needs the
pretrained RL policy, D2).
