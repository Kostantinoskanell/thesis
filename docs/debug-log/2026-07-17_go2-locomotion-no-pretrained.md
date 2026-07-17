# No plug-and-play pretrained RL Go2 locomotion on Windows/classic-MuJoCo

_2026-07-17 · severity: blocking-decision (D2) · the flagged "riskiest integration" bit_

## What we wanted
A pretrained RL velocity-tracking policy to walk the Go2 in our MuJoCo loop.

## What we found
- **unitree_rl_gym** `deploy/` ships pretrained policies only for **humanoids** (G1, H1,
  H1_2) — no Go2. Its deploy convention is generic (47-dim obs: base ang-vel, projected
  gravity, velocity cmd, joint pos/vel vs default, last action, gait phase → action →
  PD), so a Go2 policy *would* drop in — but the weights don't exist here.
- A policy is tightly coupled to its exact joint order, default angles, and training XML;
  can't just grab arbitrary weights.
- **MuJoCo Playground** has Go2 locomotion but is **JAX/MJX**, and JAX has no good native
  Windows GPU support (needs Linux/WSL2). Training on CPU is infeasible.

## Interim
Drove the Go2 with a **CPG trot through the PD controller**. It stays upright but wobbles
in yaw (ω ±1.5 rad/s when commanded straight) and under-tracks speed — inadequate as the
real layer, fine as a placeholder so the pipeline isn't blocked.

## Decision needed (real locomotion layer)
1. **Convex MPC in MuJoCo** — Windows-native, model-based, robust, medium effort
   (reference: go2-convex-mpc). Not RL, but excellent velocity tracking.
2. **RL policy via WSL2 + GPU** — the SOTA pick; set up Linux/CUDA, train (Isaac Gym or
   MuJoCo Playground), export, deploy the policy in our Windows MuJoCo loop. Infra-heavy.

## Lesson
"Use a pretrained RL policy" assumes the artifact + matching model are downloadable.
For quadrupeds on Windows they largely aren't — budget for either a model-based
controller (MPC) or a Linux/GPU training pipeline before committing.

## CORRECTION (same day)
"Unavailable" was too strong — it applies only to *downloadable pretrained* policies.
The machine has an **RTX 4060 + WSL2 Ubuntu already installed**, and GPU CUDA passthrough
into WSL2 is verified (`nvidia-smi` works inside Ubuntu). So the SOTA RL path is feasible:
**train** a Go2 velocity policy in WSL2 (JAX/MJX MuJoCo Playground on the 4060), **export**
the (small MLP) weights, and **deploy** the forward pass in our Windows MuJoCo loop —
JAX only at train time. Chosen plan. (The M1 Mac + UTM Linux is NOT usable: no NVIDIA
CUDA on Apple silicon.)
