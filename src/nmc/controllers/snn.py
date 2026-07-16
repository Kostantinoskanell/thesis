"""LIF-SNN controller with online (R-)STDP (proposal Sec. 3.2).

Weights are pretrained via surrogate-gradient BPTT to match the MLP-DNN, then
"released" to online plasticity at evaluation.  The plasticity step delegates
to nmc.plasticity.stdp.STDPLearner (the golden reference the FPGA mirrors).

The frozen-SNN ablation (proposal Sec. 3.1, controller 3) is just this class
with `plasticity_enabled=False`: same LIF dynamics, no weight updates.  It
isolates LIF temporal filtering (H4) from online plasticity (H1).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    import snntorch as snn
    from snntorch import surrogate
    _HAS_SNNTORCH = True
except ImportError:  # allow import without the backend for offline testing
    _HAS_SNNTORCH = False

from nmc.plasticity.stdp import STDPConfig, STDPLearner


class LIFNet(nn.Module):
    """Fully-connected LIF-SNN, 3-4 layers, ~512-1024 neurons (proposal Sec. 3.2)."""

    def __init__(self, in_dim: int, hidden=(512, 512), n_pops: int = 4,
                 pop_size: int = 16, beta: float = 0.95):
        super().__init__()
        if not _HAS_SNNTORCH:
            raise ImportError("snntorch is required to instantiate LIFNet")
        spike_grad = surrogate.fast_sigmoid()
        self.out_dim = n_pops * pop_size
        self.n_pops = n_pops
        self.pop_size = pop_size

        dims = [in_dim, *hidden, self.out_dim]
        self.fc = nn.ModuleList(nn.Linear(dims[i], dims[i + 1], bias=False)
                                for i in range(len(dims) - 1))
        self.lif = nn.ModuleList(snn.Leaky(beta=beta, spike_grad=spike_grad)
                                 for _ in range(len(dims) - 1))

    def forward(self, x_seq: torch.Tensor):
        """x_seq: (T, B, in_dim). Returns (out_spike_sum (B, out_dim), spikes list)."""
        mems = [lif.init_leaky() for lif in self.lif]
        out_sum = 0
        all_spikes = []
        for t in range(x_seq.shape[0]):
            cur = x_seq[t]
            step_spikes = []
            for i, (fc, lif) in enumerate(zip(self.fc, self.lif)):
                cur, mems[i] = lif(fc(cur), mems[i])
                step_spikes.append(cur)
            out_sum = out_sum + cur
            all_spikes.append(step_spikes)
        return out_sum, all_spikes

    def decode(self, out_sum: torch.Tensor) -> int:
        """Population vote -> discrete action (proposal Sec. 3.2, step 3)."""
        votes = out_sum.view(-1, self.n_pops, self.pop_size).sum(-1)
        return int(votes.argmax(-1).item())


class SNNController:
    def __init__(self, net: "LIFNet", plasticity_enabled: bool = True,
                 stdp_cfg: STDPConfig | None = None, device: str = "cpu"):
        self.net = net.to(device)
        self.device = device
        self.plasticity_enabled = plasticity_enabled
        # Online plasticity acts on the final (readout) layer weights as the
        # first target; extend to hidden layers once validated.
        w = net.fc[-1].weight.detach().cpu().numpy().astype(np.float64)
        self.W = w
        cfg = stdp_cfg or STDPConfig()
        self.learner = STDPLearner(n_pre=w.shape[1], n_post=w.shape[0], cfg=cfg)
        self._last_hidden_spikes = None
        self._last_out_spikes = None

    def act(self, x_seq: torch.Tensor) -> int:
        out_sum, all_spikes = self.net(x_seq)
        # cache mean spike vectors over the encoding window for the plasticity step
        last = all_spikes[-1]
        self._last_hidden_spikes = last[-2].detach().cpu().numpy().ravel()
        self._last_out_spikes = last[-1].detach().cpu().numpy().ravel()
        return self.net.decode(out_sum)

    def learn(self, reward: float = 0.0):
        """Apply one (R-)STDP update from the cached spikes + reward."""
        if not self.plasticity_enabled or self._last_out_spikes is None:
            return
        self.learner.step(self.W, self._last_hidden_spikes, self._last_out_spikes, reward=reward)
        # write updated weights back into the torch layer
        with torch.no_grad():
            self.net.fc[-1].weight.copy_(torch.as_tensor(self.W, dtype=self.net.fc[-1].weight.dtype))
