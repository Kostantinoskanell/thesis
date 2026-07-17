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


if _HAS_SNNTORCH:
    class _LeakyCell(nn.Module):
        """Thin wrapper giving snn.Leaky a uniform (init_state, step) interface."""

        def __init__(self, beta: float, spike_grad):
            super().__init__()
            self.cell = snn.Leaky(beta=beta, spike_grad=spike_grad)

        def init_state(self):
            return self.cell.init_leaky()

        def step(self, inp, state):
            spk, mem = self.cell(inp, state)
            return spk, mem


    class ALIFCell(nn.Module):
        """Adaptive LIF neuron (Bellec et al. 2020, e-prop): a leaky membrane plus a
        spike-triggered threshold that decays back to baseline. The adaptive
        threshold gives the neuron longer temporal memory than vanilla LIF, which
        is exactly the temporal structure rate-coded LiDAR otherwise lacks for
        STDP to exploit (proposal open risk). snnTorch has no built-in ALIF
        (Leaky.learn_threshold is a *learnable fixed* threshold, not adaptive),
        so this is a compact custom implementation using the same fast-sigmoid
        surrogate gradient.

            mem[t]   = beta * mem[t-1] + input[t]        (leaky integration)
            thr[t]   = v_th + beta_adapt * a[t]          (adaptive threshold)
            spk[t]   = surrogate(mem[t] - thr[t])        (spike)
            mem[t]  -= spk[t] * thr[t]                    (subtract reset)
            a[t+1]   = rho * a[t] + spk[t]               (threshold adaptation)
        """

        def __init__(self, beta: float = 0.95, rho: float = 0.98,
                     beta_adapt: float = 1.5, threshold: float = 1.0,
                     spike_grad=None):
            super().__init__()
            self.beta = beta
            self.rho = rho
            self.beta_adapt = beta_adapt
            self.v_th = threshold
            self.spike_grad = spike_grad or surrogate.fast_sigmoid()

        def init_state(self):
            return (0.0, 0.0)          # (mem, adaptation) -- broadcast on first step

        def step(self, inp, state):
            mem, a = state
            mem = self.beta * mem + inp
            thr = self.v_th + self.beta_adapt * a
            spk = self.spike_grad(mem - thr)
            mem = mem - spk * thr      # subtract reset
            a = self.rho * a + spk
            return spk, (mem, a)


class LIFNet(nn.Module):
    """Fully-connected spiking net, 3-4 layers (proposal Sec. 3.2).

    neuron="lif"  -> vanilla leaky integrate-and-fire (snn.Leaky).
    neuron="alif" -> adaptive-LIF (D2 SOTA upgrade): adaptive threshold for
                     longer temporal memory. Same weights/decode; drop-in.
    """

    def __init__(self, in_dim: int, hidden=(512, 512), n_pops: int = 4,
                 pop_size: int = 16, beta: float = 0.95, neuron: str = "lif"):
        super().__init__()
        if not _HAS_SNNTORCH:
            raise ImportError("snntorch is required to instantiate LIFNet")
        spike_grad = surrogate.fast_sigmoid()
        self.out_dim = n_pops * pop_size
        self.n_pops = n_pops
        self.pop_size = pop_size
        self.neuron = neuron

        dims = [in_dim, *hidden, self.out_dim]
        self.fc = nn.ModuleList(nn.Linear(dims[i], dims[i + 1], bias=False)
                                for i in range(len(dims) - 1))
        if neuron == "alif":
            self.lif = nn.ModuleList(ALIFCell(beta=beta, spike_grad=spike_grad)
                                     for _ in range(len(dims) - 1))
        elif neuron == "lif":
            self.lif = nn.ModuleList(_LeakyCell(beta=beta, spike_grad=spike_grad)
                                     for _ in range(len(dims) - 1))
        else:
            raise ValueError(f"unknown neuron type {neuron!r} (use 'lif' or 'alif')")

    def forward(self, x_seq: torch.Tensor):
        """x_seq: (T, B, in_dim). Returns (out_spike_sum (B, out_dim), spikes list)."""
        states = [cell.init_state() for cell in self.lif]
        out_sum = 0
        all_spikes = []
        for t in range(x_seq.shape[0]):
            cur = x_seq[t]
            step_spikes = []
            for i, (fc, cell) in enumerate(zip(self.fc, self.lif)):
                cur, states[i] = cell.step(fc(cur), states[i])
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


class SNNNavController:
    """Closed-loop navigation controller around a LIFNet.

    Encodes the 37-dim nav observation into a spike raster (encode_nav_obs),
    runs the LIF-SNN, and decodes a discrete action by population vote -- the
    same act(obs)->int / observe(reward, next_obs, done) interface as OnlineMLP,
    so it plugs straight into the M2 evaluation loop.

    plasticity_enabled=False -> the frozen-SNN ablation (proposal controller 3)
      and the M3 pretrained controller before release to plasticity.
    plasticity_enabled=True  -> R-STDP online adaptation (M4), delegating the
      weight update to the STDPLearner golden reference via SNNController.
    """

    def __init__(self, net: "LIFNet", n_steps: int = 20,
                 plasticity_enabled: bool = False, stdp_cfg: "STDPConfig | None" = None,
                 device: str = "cpu", seed: int = 0):
        import numpy as _np
        self.net = net.to(device)
        self.net.eval()
        self.device = device
        self.n_steps = n_steps
        self.plasticity_enabled = plasticity_enabled
        self.rng = _np.random.default_rng(seed)
        self.core = (SNNController(net, plasticity_enabled=True, stdp_cfg=stdp_cfg,
                                   device=device) if plasticity_enabled else None)
        # Spike-activity accounting (energy proxy / H2 preview): total emitted
        # spikes and total neuron-steps, accumulated across act() calls.
        self.total_spikes = 0.0
        self.total_neuron_steps = 0
        self.n_decisions = 0

    def act(self, obs) -> int:
        from nmc.encoding.spike_encoding import encode_nav_obs
        raster = encode_nav_obs(obs, self.n_steps, rng=self.rng)     # (T, F)
        x_seq = torch.as_tensor(raster, device=self.device).unsqueeze(1)  # (T, 1, F)
        if self.plasticity_enabled:
            return self.core.act(x_seq)
        with torch.no_grad():
            out_sum, all_spikes = self.net(x_seq)
        for step in all_spikes:
            for layer_spk in step:
                self.total_spikes += float(layer_spk.sum())
                self.total_neuron_steps += layer_spk.numel()
        self.n_decisions += 1
        return self.net.decode(out_sum)

    def spike_stats(self) -> dict:
        """Sparsity / energy proxy over all decisions so far (H2 preview)."""
        if self.n_decisions == 0:
            return {"spikes_per_decision": 0.0, "firing_rate": 0.0}
        return {"spikes_per_decision": self.total_spikes / self.n_decisions,
                "firing_rate": self.total_spikes / max(self.total_neuron_steps, 1)}

    def observe(self, reward: float, next_obs, done: bool):
        """No-op when frozen; one R-STDP update when plasticity is enabled (M4)."""
        if self.plasticity_enabled:
            self.core.learn(reward)
