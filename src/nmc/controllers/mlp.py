"""MLP baselines: frozen and online-updating (proposal Sec. 3.1, controllers 1-2).

The online-updating MLP is the *fair* adaptive baseline against R-STDP. It learns
online via an **actor-critic with eligibility traces** (TD(lambda), Sutton & Barto
Ch. 13.6), driven only by the robot's own collision + progress signals -- the SAME
environment-derived feedback that gates the R-STDP reward broadcast. This mirrors
R-STDP's own eligibility-trace mechanism (proposal Eq. rstdp): both adaptive
controllers accumulate an eligibility trace and consolidate it under a reward-
derived signal, differing only in the substrate. lambda=0 recovers the proposal's
original one-step TD baseline. It does NOT get expert labels post-shift (unfair),
and it is NOT self-supervision on its own actions (collapses to a fixed point).

Capacity note: the MLP body matches the LIF-SNN (512x512) so the baseline can't be
dismissed as under-powered relative to the proposed controller.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=(512, 512),
                 dropout: float = 0.0, layernorm: bool = True):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            # Always include the Dropout module (p=0 is a no-op) so the module
            # layout -- and therefore the state_dict keys -- are identical
            # whether or not dropout is used. Otherwise a model trained with
            # dropout>0 can't be loaded into a dropout=0 construction.
            layers.append(nn.Dropout(dropout))
            d = h
        self.body = nn.Sequential(*layers)
        self.pi = nn.Linear(d, n_actions)   # action logits
        self.v = nn.Linear(d, 1)            # state value (for TD target)

    def forward(self, x):
        z = self.body(x)
        return self.pi(z), self.v(z).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> int:
        logits, _ = self.forward(obs.unsqueeze(0))
        return int(logits.argmax(-1).item())


class OnlineMLP:
    """Frozen or online-updating wrapper around MLPPolicy.

    frozen=True  -> controller 1 (weights fixed after imitation pretraining)
    frozen=False -> controller 2: online actor-critic with **eligibility traces**
                    (TD(lambda)). This is the direct, fair competitor to R-STDP.

    The eligibility trace is the deliberate design parallel to R-STDP: on every
    step we accumulate per-parameter traces of the policy (actor) and value
    (critic) gradients, decayed by gamma*lambda; when the environment reward
    arrives we form the TD error delta and apply theta += lr * delta * trace.
    Structurally identical to R-STDP's e_ij trace gated by a global reward --
    only the substrate differs. lambda=0 reduces exactly to the proposal's
    one-step TD baseline.
    """

    def __init__(self, policy: MLPPolicy, frozen: bool, lr: float = 1e-3,
                 gamma: float = 0.95, lam: float = 0.9,
                 value_coef: float = 0.5, device: str = "cpu"):
        self.policy = policy.to(device)
        self.frozen = frozen
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.value_coef = value_coef
        self.device = device
        self.params = list(policy.parameters())
        # per-parameter eligibility traces (actor + critic combined stream)
        self._trace = [torch.zeros_like(p) for p in self.params]
        self._prev_value = None   # V(s) with grad-graph pending for delta

    def reset_traces(self):
        for z in self._trace:
            z.zero_()
        self._prev_value = None

    def act(self, obs) -> int:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits, value = self.policy(o.unsqueeze(0))
        action = int(logits.argmax(-1).item())
        if self.frozen:
            return action
        # Accumulate eligibility from THIS (s, a): grad of [logpi(a|s) +
        # value_coef * V(s)] w.r.t. every parameter, decayed into the trace.
        logp = F.log_softmax(logits, dim=-1)[0, action]
        objective = logp + self.value_coef * value.squeeze(0)
        grads = torch.autograd.grad(objective, self.params, retain_graph=False,
                                    allow_unused=True)
        for z, g in zip(self._trace, grads):
            z.mul_(self.gamma * self.lam)
            if g is not None:
                z.add_(g)
        self._prev_value = float(value.squeeze(0).detach())
        return action

    @torch.no_grad()
    def observe(self, reward: float, next_obs, done: bool):
        """TD(lambda) update: delta gates the accumulated eligibility trace."""
        if self.frozen or self._prev_value is None:
            return
        if done:
            next_v = 0.0
        else:
            no = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            _, nv = self.policy(no.unsqueeze(0))
            next_v = float(nv.squeeze(0))
        delta = reward + self.gamma * next_v - self._prev_value
        for p, z in zip(self.params, self._trace):
            p.add_(self.lr * delta * z)
        if done:
            self.reset_traces()
