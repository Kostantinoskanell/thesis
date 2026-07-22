"""R-STDP controller for the PopSAN locomotion actor (L-track, L4).

Mirrors nmc.controllers.snn.SNNController's design -- the same three-factor rule
(Hebbian eligibility trace x global third factor), the same D8 TD-error signal
(a linear critic gives dopamine-style reward-prediction-error, not raw reward),
and the same D9 input+readout plasticity + elastic weight anchoring (readout-only
provably can't re-map a corrupted INPUT, and unregularized adaptation over-adapts
and collapses -- both lessons earned the hard way on the nav-layer M4 pilot).

Deliberately NOT a subclass of SNNController: PopSpikingActorNet's population
encoder/decoder differ enough from LIFNet's population-vote discrete decoder
that force-fitting one class would obscure both. Reuses the same underlying
STDPLearner (numpy golden reference) so the FPGA story is unaffected -- L4 is
still "one eligibility-trace-times-third-factor rule," just applied to a
continuous-action population-coded net instead of a discrete-action one.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from nmc.locomotion.popsan_actor import PopSpikingActorNet
from nmc.plasticity.stdp import STDPConfig, STDPLearner


class PopSpikingRSTDPController:
    def __init__(self, net: PopSpikingActorNet, plastic_layers: list[int] | None = None,
                 stdp_cfg: STDPConfig | None = None, anchor: float = 0.005,
                 reward_mode: str = "td", critic_lr: float = 0.05, gamma: float = 0.97,
                 device: str = "cpu"):
        self.net = net.to(device)
        self.device = device
        n_layers = len(net.fc)
        # Default = input layer (0) + last layer (n-1), i.e. "input+readout" (D9):
        # readout-only plasticity provably cannot re-map a shifted/corrupted input.
        if plastic_layers is None:
            plastic_layers = [0, n_layers - 1]
        self.plastic_layers = [i % n_layers for i in plastic_layers]

        cfg = stdp_cfg or STDPConfig()
        self.layer = {}
        for i in self.plastic_layers:
            w = net.fc[i].weight.detach().cpu().numpy().astype(np.float64)
            b = 2.0 * float(np.abs(w).max()) if w.size else 1.0
            lcfg = dataclasses.replace(cfg, w_min=-b, w_max=b)
            self.layer[i] = {
                "W": w,
                "W0": w.copy(),  # pretrained anchor (stability, D9)
                "learner": STDPLearner(n_pre=w.shape[1], n_post=w.shape[0], cfg=lcfg),
            }
        self.anchor = anchor
        self.reward_mode = reward_mode
        self.critic_lr = critic_lr
        self.gamma = gamma
        self._wv = None
        self._bv = 0.0
        self._r_bar = 0.0
        self._win_pre = None
        self._win_post = None
        self._cur_obs = None
        self.n_learn = 0

    def _value(self, obs: np.ndarray) -> float:
        if self._wv is None:
            self._wv = np.zeros(obs.shape[0], dtype=np.float64)
        return float(self._wv @ obs + self._bv)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs: (obs_dim,) numpy -> action (act_dim,) numpy. B=1 (single-episode
        online deployment, matching the nav-layer's per-step R-STDP flow)."""
        self._cur_obs = np.asarray(obs, dtype=np.float64)
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, pre, post = self.net.forward_with_traces(x, self.plastic_layers)
        self._win_pre = {i: [t.squeeze(0).cpu().numpy() for t in pre[i]] for i in self.plastic_layers}
        self._win_post = {i: [t.squeeze(0).cpu().numpy() for t in post[i]] for i in self.plastic_layers}
        return action.squeeze(0).cpu().numpy()

    def learn(self, reward: float, next_obs: np.ndarray | None = None, done: bool = False):
        """One R-STDP decision update -- same structure as SNNController.learn
        (accumulate eligibility over the T-step window, consolidate with the third
        factor, anchor back toward pretrained weights)."""
        if not self._win_post:
            return
        if self.reward_mode == "td":
            s = self._cur_obs
            v_s = self._value(s) if s is not None else 0.0
            v_ns = 0.0 if (done or next_obs is None) else self._value(np.asarray(next_obs, dtype=np.float64))
            modulator = reward + self.gamma * v_ns - v_s
            if s is not None:
                self._wv += self.critic_lr * modulator * s
                self._bv += self.critic_lr * modulator
        elif self.reward_mode == "rpe":
            modulator = reward - self._r_bar
            self._r_bar += 0.02 * (reward - self._r_bar)
        else:
            modulator = reward
        self.n_learn += 1

        for i in self.plastic_layers:
            L = self.layer[i]
            W, learner = L["W"], L["learner"]
            for pre, post in zip(self._win_pre[i], self._win_post[i]):
                learner.step(W, pre, post, reward=0.0)  # accumulate eligibility
            if learner.cfg.reward_modulated:
                learner.step(W, np.zeros(W.shape[1]), np.zeros(W.shape[0]), reward=modulator)
            if self.anchor > 0.0:
                W += self.anchor * (L["W0"] - W)
            with torch.no_grad():
                self.net.fc[i].weight.copy_(torch.as_tensor(W, dtype=self.net.fc[i].weight.dtype))
