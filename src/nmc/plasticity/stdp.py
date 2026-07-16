"""Online STDP and reward-modulated STDP (R-STDP) weight-update rules.

This module is deliberately written in plain NumPy for two reasons:

1. It is the *golden reference* for the FPGA fixed-point implementation.  The
   Verilog weight-update datapath must reproduce this arithmetic to within a
   stated tolerance (target < 2% relative error in Delta-w, see proposal
   Sec. 4 "Risk: Resource-Constrained Parallelism").
2. It has no dependency on snntorch/torch, so it can be unit-tested in
   isolation and profiled against a vectorized BLAS baseline for the H3
   speedup study.

Trace-based formulation
-----------------------
Rather than storing every spike time t_j^f and recomputing the exponential
kernel of proposal Eq. (5), we keep two exponentially-decaying traces per
neuron.  This is the standard online-equivalent of pair-based STDP and is the
form that maps cleanly onto hardware (one register + one decay-multiply per
neuron per step):

    pre-trace   x_j  <- x_j * exp(-dt/tau_plus)  ; x_j += 1 on a pre spike
    post-trace  y_i  <- y_i * exp(-dt/tau_minus) ; y_i += 1 on a post spike

The pair-based STDP contribution accumulated at step t is then

    dW_stdp[i, j] =  A_plus  * y_pre_i[j-fired] ... handled as:
      on post spike i:  dW[i, :] += A_plus  * x_pre         (LTP)
      on pre  spike j:  dW[:, j] -= A_minus * y_post         (LTD)

R-STDP (primary plan, proposal Eq. (7))
---------------------------------------
The Hebbian dW_stdp is *not* applied directly to the weights.  It is first
accumulated into a per-synapse eligibility trace e_ij that decays with tau_e,
and only a global scalar reward r(t) consolidates it:

    e_ij  <- e_ij * exp(-dt/tau_e) + dW_stdp[i, j]
    Delta_w_ij = eta * r(t) * e_ij

Setting `reward_modulated=False` recovers pure pair-based STDP (the ablation),
in which dW_stdp is applied directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class STDPConfig:
    # Pair-based STDP kernel (proposal Eq. 5)
    a_plus: float = 0.008          # LTP amplitude
    a_minus: float = 0.009         # LTD amplitude (usually slightly > a_plus)
    tau_plus_ms: float = 20.0      # LTP time window
    tau_minus_ms: float = 20.0     # LTD time window
    w_min: float = 0.0             # conductance bounds  <-> memristor [g_min, g_max]
    w_max: float = 1.0

    # R-STDP (primary plan)
    reward_modulated: bool = True
    tau_e_ms: float = 200.0        # eligibility-trace time constant
    eta: float = 0.05              # reward learning rate

    # Memristive "stochasticity" property (proposal Sec. 5 table).
    weight_noise_std: float = 0.0  # additive N(0, sigma_w^2); 0 disables

    dt_ms: float = 1.0             # simulation step used to precompute decays


class STDPLearner:
    """Stateful online (R-)STDP applied to a dense weight matrix W[post, pre].

    Shapes
    ------
    W          : (N_post, N_pre)
    pre spikes : (N_pre,)   binary/float per step
    post spikes: (N_post,)  binary/float per step
    reward     : scalar broadcast to all synapses (R-STDP only)
    """

    def __init__(self, n_pre: int, n_post: int, cfg: STDPConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.n_pre = n_pre
        self.n_post = n_post
        self.rng = rng if rng is not None else np.random.default_rng(0)

        # Precomputed per-step decay factors (dt fixed -> exp is a constant mul).
        self.decay_pre = float(np.exp(-cfg.dt_ms / cfg.tau_plus_ms))
        self.decay_post = float(np.exp(-cfg.dt_ms / cfg.tau_minus_ms))
        self.decay_e = float(np.exp(-cfg.dt_ms / cfg.tau_e_ms))

        self.x_pre = np.zeros(n_pre, dtype=np.float64)     # LTP trace, one per pre neuron
        self.y_post = np.zeros(n_post, dtype=np.float64)   # LTD trace, one per post neuron
        self.elig = np.zeros((n_post, n_pre), dtype=np.float64)  # eligibility trace

    def reset_traces(self) -> None:
        self.x_pre[:] = 0.0
        self.y_post[:] = 0.0
        self.elig[:] = 0.0

    def step(
        self,
        W: np.ndarray,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        reward: float = 0.0,
    ) -> np.ndarray:
        """Advance one timestep and return the updated weight matrix (in place)."""
        cfg = self.cfg

        # 1) Decay traces, then register this step's spikes.
        self.x_pre *= self.decay_pre
        self.y_post *= self.decay_post

        # 2) Pair-based STDP contribution for this step.
        #    LTP: for each post neuron that fired, reinforce by the current
        #    pre-trace (pre fired recently before post -> Delta_t > 0).
        #    LTD: for each pre neuron that fired, depress by the current
        #    post-trace (post fired recently before pre -> Delta_t < 0).
        dW = np.zeros_like(W)
        post_idx = np.nonzero(post_spikes)[0]
        pre_idx = np.nonzero(pre_spikes)[0]
        if post_idx.size:
            dW[post_idx, :] += cfg.a_plus * self.x_pre[None, :]
        if pre_idx.size:
            dW[:, pre_idx] -= cfg.a_minus * self.y_post[:, None]

        # 3) Now add this step's spikes to the traces (post-update so a neuron
        #    does not potentiate against its own same-step spike).
        self.x_pre += pre_spikes
        self.y_post += post_spikes

        # 4) Consolidate into weights.
        if cfg.reward_modulated:
            self.elig = self.decay_e * self.elig + dW
            delta = cfg.eta * float(reward) * self.elig
        else:
            delta = dW  # pure STDP ablation: apply Hebbian term directly

        if cfg.weight_noise_std > 0.0:
            delta = delta + self.rng.normal(0.0, cfg.weight_noise_std, size=W.shape)

        np.add(W, delta, out=W)
        np.clip(W, cfg.w_min, cfg.w_max, out=W)  # saturation <-> [g_min, g_max]
        return W


def kernel_reference(delta_t_ms: np.ndarray, cfg: STDPConfig) -> np.ndarray:
    """Exact pair-based STDP kernel of proposal Eq. (5), for LUT validation.

    Used to check the FPGA's 64-bin logarithmically-quantized BRAM LUT against
    the floating-point exponential before synthesis.
    """
    out = np.where(
        delta_t_ms > 0,
        cfg.a_plus * np.exp(-delta_t_ms / cfg.tau_plus_ms),
        -cfg.a_minus * np.exp(delta_t_ms / cfg.tau_minus_ms),
    )
    out[delta_t_ms == 0] = 0.0
    return out
