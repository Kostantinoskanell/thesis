"""MLP baselines: frozen and online-updating (proposal Sec. 3.1, controllers 1-2).

The online-updating MLP is the *fair* adaptive baseline against R-STDP.  Its
loss is a one-step TD-bootstrapped advantage target computed from the robot's
own collision + progress signals -- the SAME environment-derived feedback that
gates the R-STDP reward broadcast.  It does NOT get expert labels during
Phase 2 (that would be an unfair advantage), and it is NOT pure
self-supervision on its own actions (that collapses to a fixed point).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=(256, 256)):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
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
    frozen=False -> controller 2 (one-step TD update on a sliding window)
    """

    def __init__(self, policy: MLPPolicy, frozen: bool, lr: float = 1e-3,
                 gamma: float = 0.95, device: str = "cpu"):
        self.policy = policy.to(device)
        self.frozen = frozen
        self.gamma = gamma
        self.device = device
        self.opt = None if frozen else torch.optim.Adam(policy.parameters(), lr=lr)
        self._prev = None  # (obs, action, value)

    def act(self, obs) -> int:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits, value = self.policy(o.unsqueeze(0))
        action = int(logits.argmax(-1).item())
        self._prev = (o, action, value.squeeze(0))
        return action

    def observe(self, reward: float, next_obs, done: bool):
        """One-step TD/advantage update from environment reward only."""
        if self.frozen or self._prev is None:
            return
        o, action, value = self._prev
        with torch.no_grad():
            no = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            _, next_value = self.policy(no.unsqueeze(0))
            target = reward + (0.0 if done else self.gamma * next_value.squeeze(0))
        logits, value = self.policy(o.unsqueeze(0))
        value = value.squeeze(0)
        advantage = (target - value).detach()
        logp = F.log_softmax(logits, dim=-1)[0, action]
        policy_loss = -(advantage * logp)
        value_loss = F.smooth_l1_loss(value, target)
        loss = policy_loss + 0.5 * value_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
