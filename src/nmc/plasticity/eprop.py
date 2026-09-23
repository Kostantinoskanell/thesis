"""e-prop: online learning for ALIF spiking networks via eligibility traces
derived from the actual neuron dynamics (Bellec, Scherr, Subramoney, Hajek,
Salaj, Legenstein & Maass, "A solution to the learning dilemma for recurrent
networks of spiking neurons", Nature Communications 2020).

Why this exists (D1 in docs/references/sota_decisions.md): R-STDP's
eligibility trace is purely Hebbian (pre/post spike-timing correlation) and
its third factor is a single GLOBAL scalar reward broadcast identically to
every synapse. e-prop's eligibility trace is instead an online approximation
of the true BPTT gradient through each neuron's own membrane/adaptation
dynamics, and its third factor is a PER-NEURON learning signal obtained by
projecting an output-level error back through the (symmetric) readout
weights -- structurally a different mechanism, not a retuned R-STDP. M5 found
R-STDP does not recover the sensor-dropout shift; this exists to answer
whether that null result is about the RULE or about the PROBLEM.

Honest scope note: Bellec et al. derive e-prop for supervised sequence
learning (a known per-timestep target). This project's task is closed-loop
RL with a scalar TD-error, not a known target, so the per-neuron learning
signal here is the natural RL extension used in the eprop-for-RL literature:
a REINFORCE-style contrastive credit at the readout layer (push the CHOSEN
action's population, pull the others, scaled by TD-error and zero-mean across
the output layer) broadcast backward through the CURRENT readout weights
(the "symmetric feedback" variant of e-prop, as opposed to fixed random
feedback / full weight-transport BPTT). This is a scoped, honestly-labelled
adaptation, not a claim of exact paper fidelity.

Implemented in plain NumPy, independent of the PyTorch LIFNet used elsewhere,
matching this project's existing convention (stdp.py) of a from-scratch
golden reference: independently testable, and directly portable to the same
FPGA eligibility-trace datapath R-STDP already targets (D1 explicitly notes
both rules share one hardware shape: per-synapse eligibility register x
broadcast third factor).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pseudo_derivative(x: np.ndarray, slope: float = 25.0) -> np.ndarray:
    """The exact backward-pass formula of snnTorch's FastSigmoid surrogate
    (grad = 1 / (slope*|x| + 1)^2), evaluated at x = mem - threshold. Reusing
    the identical surrogate already used for this project's BPTT pretraining
    (M3) makes e-prop's "pseudo-derivative" and BPTT's surrogate gradient the
    same function, as the theory requires -- not an independently-chosen
    approximation.
    """
    return 1.0 / (slope * np.abs(x) + 1.0) ** 2


@dataclass
class EPropConfig:
    beta: float = 0.95          # membrane decay (must match the ALIFCell checkpoint)
    rho: float = 0.98           # threshold-adaptation decay
    beta_adapt: float = 1.5     # threshold-adaptation coupling
    v_th: float = 1.0           # baseline threshold
    surrogate_slope: float = 25.0
    eta: float = 0.05           # learning rate
    w_min: float = -1.0
    w_max: float = 1.0


class ALIFEpropLayer:
    """One plastic ALIF layer: runs its own forward dynamics (so the caller
    has direct access to membrane/threshold/adaptation state, unlike
    LIFNet.forward which only returns spikes) and accumulates the e-prop
    eligibility trace for every synapse feeding it, each timestep.

    Shapes: W (n_post, n_pre). Call `step_forward(pre_spikes)` once per
    timestep of a decision window, then `consolidate(post_error, pre_zbar_used_by_next_layer)`
    once per decision with the per-neuron learning signal for THIS layer's
    post-neurons.
    """

    def __init__(self, W: np.ndarray, cfg: EPropConfig, rng: np.random.Generator | None = None,
                 track_eligibility: bool = True):
        self.cfg = cfg
        self.W = W  # (n_post, n_pre), shared reference -- caller owns the array
        self.n_post, self.n_pre = W.shape
        self.rng = rng or np.random.default_rng(0)
        # PERFORMANCE: the eligibility trace (eps_a, e_acc) is an O(n_post*n_pre)
        # dense update every timestep -- expensive, and pointless for a layer
        # that is never consolidated (e.g. a large non-plastic hidden layer).
        # Only plastic layers need it; non-plastic ones still need the plain
        # forward pass (their spikes feed the next layer).
        self.track_eligibility = track_eligibility
        self.reset_state()

    def reset_state(self):
        self.mem = np.zeros(self.n_post)
        self.a = np.zeros(self.n_post)
        if self.track_eligibility:
            self.zbar = np.zeros(self.n_pre)       # low-pass presynaptic trace
            self.eps_a = np.zeros((self.n_post, self.n_pre))  # adaptation eligibility
            self.e_acc = np.zeros((self.n_post, self.n_pre))  # eligibility accumulated over the window
            self._h_prev = np.zeros(self.n_post)
            self._zbar_prev = np.zeros(self.n_pre)

    def step_forward(self, pre_spikes: np.ndarray) -> np.ndarray:
        """Advance one timestep. pre_spikes: (n_pre,) binary. Returns this
        layer's post spikes (n_post,) -- feed these as `pre_spikes` to the
        next plastic layer, or use for the readout decode."""
        c = self.cfg
        cur = self.W @ pre_spikes
        self.mem = c.beta * self.mem + cur
        thr = c.v_th + c.beta_adapt * self.a
        x = self.mem - thr
        spk = (x > 0.0).astype(np.float64)   # matches snnTorch surrogate forward: torch.gt(input, 0)

        if self.track_eligibility:
            h = pseudo_derivative(x, c.surrogate_slope)
            # ALIF eligibility trace (Bellec et al. 2020, hard-reset ALIF variant):
            # eps_a tracks how past (pre,post) pairing still influences the
            # CURRENT adaptive threshold (this is what gives e-prop -- unlike
            # plain LIF -- a long credit-assignment horizon).
            outer_prev = self._h_prev[:, None] * self._zbar_prev[None, :]
            self.eps_a = outer_prev + (c.rho - self._h_prev[:, None] * c.beta_adapt) * self.eps_a
            e_t = h[:, None] * (self.zbar[None, :] - c.beta_adapt * self.eps_a)
            self.e_acc += e_t
            self._h_prev = h
            self._zbar_prev = self.zbar.copy()
            self.zbar = c.beta * self.zbar + pre_spikes

        # advance state for next step
        self.mem = self.mem - spk * thr             # subtractive reset
        self.a = c.rho * self.a + spk
        return spk

    def consolidate(self, post_learning_signal: np.ndarray, anchor: float = 0.0,
                    W0: np.ndarray | None = None) -> None:
        """Apply eta * L_j * (accumulated eligibility) once per decision, then
        clip and (optionally) pull back toward the pretrained anchor W0 --
        identical stabilization machinery to R-STDP, so a comparison isolates
        the learning RULE, not the anchoring."""
        c = self.cfg
        dW = c.eta * post_learning_signal[:, None] * self.e_acc
        self.W += dW
        if anchor > 0.0 and W0 is not None:
            self.W += anchor * (W0 - self.W)
        np.clip(self.W, c.w_min, c.w_max, out=self.W)
        self.e_acc[:] = 0.0


def readout_credit(chosen_pop: int, n_pops: int, pop_size: int, td_error: float) -> np.ndarray:
    """REINFORCE-style, zero-mean-across-populations contrastive credit at the
    readout layer: +td_error for neurons in the chosen action's population,
    -td_error/(n_pops-1) for the rest. This is the RL stand-in for e-prop's
    "known target minus prediction" supervised error signal (see module
    docstring)."""
    credit = np.full(n_pops, -td_error / max(n_pops - 1, 1))
    credit[chosen_pop] = td_error
    return np.repeat(credit, pop_size)


def symmetric_feedback(post_credit: np.ndarray, W_next: np.ndarray) -> np.ndarray:
    """Broadcast a downstream per-neuron credit vector back to THIS layer's
    neurons through the CURRENT (not fixed-random) forward weights of the
    next layer -- the "symmetric e-prop" feedback variant. W_next: (n_post_next, n_this)."""
    return W_next.T @ post_credit
