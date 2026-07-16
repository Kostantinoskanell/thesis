"""Unit tests for the (R-)STDP golden reference.

Run: pytest -q   (from repo root, with src/ on PYTHONPATH -- see pyproject/conftest)
These tests pin down the arithmetic the FPGA fixed-point datapath must match.
"""

import numpy as np

from nmc.plasticity.stdp import STDPConfig, STDPLearner, kernel_reference


def test_ltp_on_pre_before_post():
    """Pre spike then post spike a step later => weight should increase (LTP)."""
    cfg = STDPConfig(reward_modulated=False, w_min=-1, w_max=1, a_plus=0.1, a_minus=0.1)
    L = STDPLearner(n_pre=1, n_post=1, cfg=cfg)
    W = np.zeros((1, 1))
    L.step(W, pre_spikes=np.array([1.0]), post_spikes=np.array([0.0]))   # pre fires
    L.step(W, pre_spikes=np.array([0.0]), post_spikes=np.array([1.0]))   # post fires later
    assert W[0, 0] > 0, "pre-before-post must potentiate"


def test_ltd_on_post_before_pre():
    """Post spike then pre spike a step later => weight should decrease (LTD)."""
    cfg = STDPConfig(reward_modulated=False, w_min=-1, w_max=1, a_plus=0.1, a_minus=0.1)
    L = STDPLearner(n_pre=1, n_post=1, cfg=cfg)
    W = np.zeros((1, 1))
    L.step(W, pre_spikes=np.array([0.0]), post_spikes=np.array([1.0]))   # post fires
    L.step(W, pre_spikes=np.array([1.0]), post_spikes=np.array([0.0]))   # pre fires later
    assert W[0, 0] < 0, "post-before-pre must depress"


def test_weight_clamp():
    cfg = STDPConfig(reward_modulated=False, w_min=0.0, w_max=0.05, a_plus=1.0)
    L = STDPLearner(1, 1, cfg)
    W = np.array([[0.04]])
    L.step(W, np.array([1.0]), np.array([0.0]))
    L.step(W, np.array([0.0]), np.array([1.0]))
    assert W[0, 0] <= 0.05 + 1e-12, "weights must saturate at w_max"


def test_rstdp_no_reward_no_change():
    """With R-STDP, zero reward must leave weights unchanged (only eligibility moves)."""
    cfg = STDPConfig(reward_modulated=True, eta=0.5)
    L = STDPLearner(2, 2, cfg)
    W = np.full((2, 2), 0.5)
    W0 = W.copy()
    for _ in range(5):
        L.step(W, np.array([1.0, 0.0]), np.array([0.0, 1.0]), reward=0.0)
    assert np.allclose(W, W0), "no reward => no consolidation"
    assert np.any(L.elig != 0), "eligibility trace must accumulate even at r=0"


def test_rstdp_reward_sign():
    """Positive reward consolidates accumulated LTP eligibility as a weight increase."""
    cfg = STDPConfig(reward_modulated=True, eta=0.5, a_plus=0.1, a_minus=0.1)
    L = STDPLearner(1, 1, cfg)
    W = np.array([[0.5]])
    L.step(W, np.array([1.0]), np.array([0.0]), reward=0.0)   # build pre trace
    L.step(W, np.array([0.0]), np.array([1.0]), reward=1.0)   # post fires + reward
    assert W[0, 0] > 0.5, "positive reward on LTP eligibility must increase weight"


def test_kernel_reference_shape_and_sign():
    cfg = STDPConfig()
    dt = np.array([-40.0, -1.0, 0.0, 1.0, 40.0])
    k = kernel_reference(dt, cfg)
    assert k[2] == 0.0
    assert k[3] > 0 and k[4] > 0          # LTP side positive
    assert k[0] < 0 and k[1] < 0          # LTD side negative
    assert abs(k[3]) > abs(k[4])          # kernel decays with |dt|
