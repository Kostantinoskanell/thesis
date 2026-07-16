"""Neuromorphic control for robotics — thesis codebase.

Packages:
    envs        PyBullet navigation env with mid-episode distribution shift
    encoding    sensor -> spike-train encoders (rate / TTFS)
    controllers MLP baselines + LIF-SNN controller
    plasticity  online STDP / R-STDP (golden reference for the FPGA)
    eval        evaluation metrics (SynOps, recovery time, ...)
"""

__version__ = "0.0.1"
