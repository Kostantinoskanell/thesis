# Quadruped CPG trot collapses (lunges forward and falls)

_2026-07-17 · severity: medium (P1) · caught purely by watching the GIF_

## Symptom
First CPG trot: net displacement +3.5 m but final base height **z=0.056** (collapsed).
The metric "walked >0.5 m" was TRUE — yet the GIF showed the A1 lunging and faceplanting,
not walking. Turning also did nothing (heading change +1°).

## Why the numbers lied
Forward displacement alone can't distinguish "walking" from "falling forward." Only the
base-height check (z should stay ~0.28) + watching the render exposed the collapse. This
is the whole reason for the visual-first rule.

## Root cause
Open-loop trot with large joint amplitudes and soft PD gains is dynamically unstable:
the body pitches and drops faster than the (weak) joint servos can hold posture, so it
topples in the commanded direction. Turning made it worse (extra lateral perturbation
on an already-marginal gait).

## Fix (conservative gait)
- PD stiffness `kp` 60 → 90, `max_force` 40 → 55 N·m (joints actually hold the pose).
- Amplitudes down: `sweep_amp` 0.35 → 0.20, `lift_amp` 0.55 → 0.30, `gait_freq` 2.5 → 2.0.
- Isolated forward stability first, then re-enabled turning.
Result: stable trot, z stays 0.27–0.31, walks ~0.3 m/s, turns +108° (omega>0 = left).

Also fixed a **turn-direction sign** bug: omega>0 must be CCW/left; the right-side legs
need the larger stride, so the differential-sweep term uses `-side`.

## Lessons
- Always pair a "progress" metric with a "didn't-fail" metric (here: base height) — and
  watch the render. A green PASS from displacement alone was misleading.
- Open-loop CPG buys a stable *slow* gait cheaply; precise command tracking needs
  convex-MPC / RL (deferred, see docs/references/locomotion.md).
