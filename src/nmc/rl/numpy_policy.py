"""Pure-NumPy forward pass for the exported Go2 velocity-tracking policy.

Windows-side runtime for the policy trained in WSL/JAX (scripts/rl/train_go2.py,
exported by scripts/rl/export_go2_policy.py). No JAX/torch needed -- it's a
small MLP: normalize obs -> (512,256,128) swish layers -> 24 outputs, of which
the first 12 are the pre-squash action means; deterministic action = tanh(loc).
(Matches brax's NormalTanhDistribution mode and ppo_networks' default swish
activation; verified bit-near-exactly by scripts/verify_policy_parity.py.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _swish(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class NumpyGo2Policy:
    def __init__(self, export_path: str | Path):
        z = np.load(export_path)
        self.mean = z["obs_mean"].astype(np.float64)
        self.std = z["obs_std"].astype(np.float64)
        n = int(z["n_layers"])
        self.weights = [z[f"w{i}"].astype(np.float64) for i in range(n)]
        self.biases = [z[f"b{i}"].astype(np.float64) for i in range(n)]
        self.action_size = self.weights[-1].shape[1] // 2

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """obs: (48,) raw 'state' observation -> (12,) action in [-1, 1]."""
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / self.std
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i != last:
                x = _swish(x)
        loc = x[: self.action_size]
        return np.tanh(loc)
