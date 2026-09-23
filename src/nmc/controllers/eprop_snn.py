"""EPropNavController: the e-prop analogue of SNNNavController/SNNController,
for a direct, apples-to-apples comparison against R-STDP on the same
pretrained ALIF network, same TD-error third factor, same elastic-weight
anchoring -- differing ONLY in the eligibility trace (Hebbian STDP vs the
neuron-dynamics-derived e-prop trace) and how the third factor reaches each
synapse (one global scalar vs a per-neuron symmetric-feedback signal). See
src/nmc/plasticity/eprop.py for the learning-rule math and honest scope note.

Same act(obs) -> int / observe(reward, next_obs, done) interface as every
other controller in this codebase, so it plugs into the existing M5-style
evaluation scripts unchanged.
"""
from __future__ import annotations

import numpy as np
import torch

from nmc.encoding.spike_encoding import encode_nav_obs
from nmc.plasticity.eprop import ALIFEpropLayer, EPropConfig, readout_credit, symmetric_feedback


class EPropNavController:
    def __init__(self, ckpt_path: str, n_steps: int | None = None, seed: int = 0,
                 eta: float = 0.05, anchor: float = 0.005, gamma: float = 0.97,
                 critic_lr: float = 0.05, plastic_layers=(0, -1),
                 w_bound_scale: float = 2.0):
        ckpt = torch.load(ckpt_path, weights_only=True)
        if ckpt.get("neuron", "lif") != "alif":
            raise ValueError("EPropNavController requires an ALIF checkpoint "
                             "(the adaptation eligibility term needs beta_adapt/rho); "
                             f"got neuron={ckpt.get('neuron')!r}")
        self.n_pops = ckpt["n_pops"]
        self.pop_size = ckpt["pop_size"]
        self.n_steps = n_steps or ckpt["tsteps"]
        self.rng = np.random.default_rng(seed)

        weight_keys = sorted(k for k in ckpt["state_dict"] if k.startswith("fc.") and k.endswith(".weight"))
        Ws = [ckpt["state_dict"][k].detach().cpu().numpy().astype(np.float64).copy() for k in weight_keys]
        self.n_layers = len(Ws)
        self.plastic_layers = sorted({i % self.n_layers for i in plastic_layers})

        self.layers: list[ALIFEpropLayer] = []
        self.W0 = {}
        for i, W in enumerate(Ws):
            bound = w_bound_scale * float(np.abs(W).max())
            cfg = EPropConfig(eta=eta, w_min=-bound, w_max=bound)
            # non-plastic layers skip the O(n_post*n_pre) eligibility bookkeeping
            # entirely (never consolidated, so never read) -- a real, measured
            # perf win: the middle hidden layer (512x512) is the single largest
            # in the network and was previously paying this cost for nothing.
            self.layers.append(ALIFEpropLayer(W, cfg, rng=self.rng,
                                              track_eligibility=(i in self.plastic_layers)))
            if i in self.plastic_layers:
                self.W0[i] = W.copy()

        self.anchor = anchor
        self.gamma = gamma
        self.critic_lr = critic_lr
        self._wv = None
        self._bv = 0.0
        self._cur_obs = None
        self._chosen_pop = None
        self.n_learn = 0

        # spike-activity accounting, same shape as SNNNavController.spike_stats()
        self.total_spikes = 0.0
        self.total_neuron_steps = 0
        self.n_decisions = 0
        self.layer_spikes = [0.0] * self.n_layers
        self.layer_slots = [0] * self.n_layers
        self.enc_spikes = 0.0
        self.enc_slots = 0
        self.decision_rates: list[float] = []

    def _value(self, obs):
        if self._wv is None:
            self._wv = np.zeros(obs.shape[0], dtype=np.float64)
        return float(self._wv @ obs + self._bv)

    def act(self, obs) -> int:
        raster = encode_nav_obs(np.asarray(obs, dtype=np.float64), self.n_steps, rng=self.rng)  # (T, F)
        for layer in self.layers:
            layer.reset_state()

        out_sum = np.zeros(self.layers[-1].n_post)
        dec_spikes = 0.0
        dec_slots = 0
        for t in range(raster.shape[0]):
            x = raster[t]
            self.enc_spikes += float(x.sum()); self.enc_slots += x.size
            for li, layer in enumerate(self.layers):
                x = layer.step_forward(x)
                s = float(x.sum()); n = x.size
                self.layer_spikes[li] += s; self.layer_slots[li] += n
                dec_spikes += s; dec_slots += n
            out_sum += x
        self.total_spikes += dec_spikes; self.total_neuron_steps += dec_slots
        self.decision_rates.append(dec_spikes / max(dec_slots, 1))
        self.n_decisions += 1

        votes = out_sum.reshape(self.n_pops, self.pop_size).sum(-1)
        action = int(np.argmax(votes))
        self._chosen_pop = action
        self._cur_obs = np.asarray(obs, dtype=np.float64)
        return action

    def observe(self, reward: float, next_obs, done: bool) -> None:
        s = self._cur_obs
        if s is None:
            return
        if self._wv is None:
            self._wv = np.zeros(s.shape[0], dtype=np.float64)
        v_s = self._value(s)
        v_ns = 0.0 if (done or next_obs is None) else self._value(np.asarray(next_obs, dtype=np.float64))
        td_error = reward + self.gamma * v_ns - v_s
        self._wv += self.critic_lr * td_error * s
        self._bv += self.critic_lr * td_error
        self.n_learn += 1

        credit = readout_credit(self._chosen_pop, self.n_pops, self.pop_size, td_error)
        credits = [None] * self.n_layers
        credits[-1] = credit
        for i in range(self.n_layers - 2, -1, -1):
            credits[i] = symmetric_feedback(credits[i + 1], self.layers[i + 1].W)

        for i in self.plastic_layers:
            self.layers[i].consolidate(credits[i], anchor=self.anchor, W0=self.W0.get(i))

    def spike_stats(self) -> dict:
        if self.n_decisions == 0:
            return {"spikes_per_decision": 0.0, "firing_rate": 0.0, "enc_rate": 0.0, "layer_rates": []}
        return {"spikes_per_decision": self.total_spikes / self.n_decisions,
                "firing_rate": self.total_spikes / max(self.total_neuron_steps, 1),
                "enc_rate": self.enc_spikes / max(self.enc_slots, 1),
                "layer_rates": [s / max(n, 1) for s, n in zip(self.layer_spikes, self.layer_slots)],
                "decision_rates": list(self.decision_rates)}
