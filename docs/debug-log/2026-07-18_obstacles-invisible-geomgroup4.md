# Obstacles/walls silently invisible in every non-debug render (geom group 4)

**Milestone:** M4 presentation videos (caught while reviewing them with the user)
**Symptom:** the user watched the M4 before/after videos and asked "where are the
obstacles?" — the rendered clips showed only the robot walking on an empty floor
toward the goal marker, across the *entire* episode, in both videos.

## Diagnosis
First suspected the camera (tight 4.2 m chase-cam) just never framed an obstacle by
chance. Checked via `privileged_state()`: **11 real obstacles existed** in the scene at
the expected positions, 1.1-4 m from the robot's actual path — plausible for a single
frame, but a 12-frame contact sheet spanning the *entire* episode (both videos) showed
**zero** obstacles or arena walls in any frame. That ruled out bad luck.

Root cause, confirmed by rendering the identical sim state two ways: default
`env.render()` showed a bare floor; forcing `MjvOption.geomgroup[4] = 1` on the *same*
state revealed the robot standing in a dense field of obstacles and the arena wall.
`Go2NavEnv`'s obstacle/wall geoms are all authored on **geom group 4**
(`src/nmc/envs/go2_nav_env.py`, chosen originally only as the LiDAR ray-cast filter
group) — group 4 is not one of MuJoCo's default-visible groups, so every ordinary
(non-`show_collision`) render has been silently omitting them since D3.

## Fix
`Go2RLWalker.render()` now always constructs an `MjvOption` with `geomgroup[4] = 1`
(obstacles/walls visible), not just when `show_collision=True`. `show_collision` still
gates only the robot's own collision capsules (group 3, redundant with its visual mesh)
and contact-point/force overlays.

## Scope of impact
Physics, LiDAR, collisions, and all M2-M4 metrics were **never affected** — this is a
pure visualization bug in `env.render()`, not the simulation. But it means the D3
archive GIF/figures and every M4 video rendered before this fix likely show an
empty-looking arena despite obstacles being real and functioning. Re-render any
visual artifact meant to show obstacles/walls after this fix.

## Lesson
"Watch the robot" verification only catches what you're actually watching. Here, the
robot's *behavior* (walking, sometimes reaching the goal, sometimes not) looked
plausible even in the broken renders, so nothing seemed obviously wrong until a human
asked "where are the obstacles" — a good reminder that a rendering bug can hide behind
correct-looking motion, and a second pair of eyes catches things automated checks and
a distracted first look do not.
