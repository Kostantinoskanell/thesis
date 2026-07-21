"""Pure-PyTorch spiking actor for the Isaac Lab / rsl_rl locomotion policy (L-track).

Self-contained: no snnTorch dependency, because the Isaac Sim conda env ships torch
but not snnTorch. The surrogate gradient is implemented directly with a custom
torch.autograd.Function (fast-sigmoid), so this module trains under rsl_rl's PPO
backprop exactly like an MLP actor would.

Design mirrors the navigation SNN's neuron model (nmc.controllers.snn) so the whole
thesis shares one neuron definition conceptually, but the I/O differs:
  - input : the locomotion observation is fed as a CONSTANT input current for T
            steps ("direct/current encoding"). Deterministic (no Poisson), because
            rsl_rl's policy forward gets no RNG key and train<->deploy must match.
  - output: a linear readout over the T-step-summed spikes of the last hidden layer
            -> continuous action means (rsl_rl adds the Gaussian exploration std).

The spiking hidden layers are the R-STDP plasticity sites for the eventual online
gait-adaptation experiment (the ice-recovery motivation from M4c): once trained,
these weights can be released to the numpy STDPLearner just like the nav layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _FastSigmoidSpike(torch.autograd.Function):
    """Heaviside spike in the forward pass; fast-sigmoid surrogate in the backward.

    grad = 1 / (1 + slope*|x|)^2  (Zenke & Ganguli 2018 / snnTorch's fast_sigmoid).
    """

    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + ctx.slope * x.abs()) ** 2
        return grad_output * sg, None


def spike(x: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
    return _FastSigmoidSpike.apply(x, slope)


class SpikingActorNet(nn.Module):
    """obs -> action-mean spiking regressor.

    neuron="lif"  : leaky integrate-and-fire.
    neuron="alif" : adaptive-LIF (spike-triggered decaying threshold) -- longer
                    temporal memory, the D2 SOTA choice from the nav layer.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256),
                 T: int = 6, beta: float = 0.95, rho: float = 0.98,
                 beta_adapt: float = 1.5, v_th: float = 1.0,
                 neuron: str = "alif", surrogate_slope: float = 25.0):
        super().__init__()
        if neuron not in ("lif", "alif"):
            raise ValueError(f"unknown neuron {neuron!r} (use 'lif' or 'alif')")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.T = T
        self.beta = beta
        self.rho = rho
        self.beta_adapt = beta_adapt
        self.v_th = v_th
        self.neuron = neuron
        self.slope = surrogate_slope

        dims = [obs_dim, *hidden]
        # bias-free linear layers = the plastic synapse matrices (R-STDP sites).
        self.fc = nn.ModuleList(nn.Linear(dims[i], dims[i + 1], bias=False)
                                for i in range(len(dims) - 1))
        # continuous readout over the summed spikes of the last hidden layer.
        self.readout = nn.Linear(hidden[-1], act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim) -> action mean (B, act_dim). Unrolls T spiking steps
        with the observation held as constant input current (direct encoding)."""
        B = obs.shape[0]
        mem = [torch.zeros(B, fc.out_features, device=obs.device, dtype=obs.dtype)
               for fc in self.fc]
        adp = [torch.zeros_like(m) for m in mem]
        out_sum = torch.zeros(B, self.fc[-1].out_features,
                              device=obs.device, dtype=obs.dtype)
        for _t in range(self.T):
            cur = obs
            for i, fc in enumerate(self.fc):
                mem[i] = self.beta * mem[i] + fc(cur)
                thr = self.v_th + (self.beta_adapt * adp[i] if self.neuron == "alif" else 0.0)
                spk = spike(mem[i] - thr, self.slope)
                mem[i] = mem[i] - spk * thr
                if self.neuron == "alif":
                    adp[i] = self.rho * adp[i] + spk
                cur = spk
            out_sum = out_sum + cur
        rate = out_sum / float(self.T)          # firing-rate code, (B, hidden[-1])
        return self.readout(rate)               # action means (rsl_rl adds std)

    @torch.no_grad()
    def firing_rate(self, obs: torch.Tensor) -> float:
        """Mean spike rate over all hidden neurons/steps for one obs batch (H2 energy
        proxy). Recomputes the unroll counting spikes."""
        B = obs.shape[0]
        mem = [torch.zeros(B, fc.out_features, device=obs.device, dtype=obs.dtype)
               for fc in self.fc]
        adp = [torch.zeros_like(m) for m in mem]
        total_spikes = 0.0
        total_slots = 0
        for _t in range(self.T):
            cur = obs
            for i, fc in enumerate(self.fc):
                mem[i] = self.beta * mem[i] + fc(cur)
                thr = self.v_th + (self.beta_adapt * adp[i] if self.neuron == "alif" else 0.0)
                spk = (mem[i] - thr > 0).to(obs.dtype)
                mem[i] = mem[i] - spk * thr
                if self.neuron == "alif":
                    adp[i] = self.rho * adp[i] + spk
                cur = spk
                total_spikes += float(spk.sum())
                total_slots += spk.numel()
        return total_spikes / max(total_slots, 1)
