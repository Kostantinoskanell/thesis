"""Spiking locomotion layer (L-track).

Extends the neuromorphic-plasticity contribution DOWN into the low-level gait
controller. Until now the locomotion layer was a frozen PPO MLP (Go2RLWalker),
deliberately decoupled from the science (sota_decisions D3). The M4c terrain
experiment exposed the ceiling of that split: R-STDP on the *navigation* layer
can recalibrate velocity *commands*, but cannot fix the *legs slipping* on ice,
because it has no access below the velocity-command interface.

Plan (chosen 2026-07-21): train a **spiking** Go2 locomotion policy directly with
RL in Isaac Lab (rsl_rl PPO, GPU) — the student's hardware runs it (see
debug-log 2026-07-21_isaac-lab-wsl-8gb-bringup). `spiking_actor.py` is the
pure-PyTorch, surrogate-gradient spiking actor network (no snnTorch dependency,
since the Isaac Sim env ships torch but not snnTorch); it plugs into rsl_rl as a
`class_name`-injected custom policy. Once trained, its spiking layers can be
released to the numpy R-STDP golden reference for online gait adaptation — the
locomotion-layer analogue of the H1 recovery claim.
"""
