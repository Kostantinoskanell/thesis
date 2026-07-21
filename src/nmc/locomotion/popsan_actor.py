"""PopSAN-style population-coded spiking actor (L-track, L2b).

Implements the population-coded spiking actor network of Tang et al. 2020
(arXiv:2010.09635; legged validation Wang/Wu 2023, arXiv:2310.05022) in pure
PyTorch (no snnTorch -- the Isaac Sim env has torch only). This SUPERSEDES the
plain firing-rate readout in spiking_actor.py, which is PopSAN's own "RateSAN"
baseline shown to under-represent continuous actions (sota_decisions D11).

Three learnable stages (paper Algorithm 1 / eqs 1-5):
  1. Population encoder: each observation dim -> a population of P_in neurons with
     LEARNABLE Gaussian receptive fields. Stimulation A_E = exp(-1/2 ((s-mu)/sig)^2)
     drives deterministic soft-reset IF neurons -> input spike train over T steps.
  2. Current-based LIF hidden layers (current decay d_c, voltage decay d_v, hard reset).
  3. Population decoder: each action dim -> a population of P_out neurons; action =
     W_d . (spike_count / T) + b_d, with LEARNABLE W_d, b_d.

Outputs the action MEAN; the RL algorithm (rsl_rl PPO) supplies the Gaussian std
and the clipped-surrogate loss, exactly as PopSAN integrates PPO. The hidden and
population weight matrices are the R-STDP plasticity sites for the later gait-
recovery experiment (L4).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nmc.locomotion.spiking_actor import spike  # fast-sigmoid surrogate spike


class PopSpikingActorNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256),
                 in_pop: int = 10, out_pop: int = 10, T: int = 5,
                 d_c: float = 0.5, d_v: float = 0.75, v_th: float = 0.5,
                 enc_min: float = -3.0, enc_max: float = 3.0,
                 surrogate_slope: float = 25.0, weight_gain: float = 3.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.in_pop = in_pop
        self.out_pop = out_pop
        self.T = T
        self.d_c = d_c
        self.d_v = d_v
        self.v_th = v_th
        self.slope = surrogate_slope

        # --- learnable Gaussian input populations (mu, sigma per obs dim x neuron) ---
        # mu init: evenly spread across the (normalized) obs range so the populations
        # tile the input space; sigma init large (spacing) for non-zero activity.
        mu0 = torch.linspace(enc_min, enc_max, in_pop).unsqueeze(0).repeat(obs_dim, 1)
        self.mu = nn.Parameter(mu0)                                    # (N, P_in)
        spacing = (enc_max - enc_min) / max(in_pop - 1, 1)
        self._raw_sigma = nn.Parameter(torch.full((obs_dim, in_pop),
                                                   float(spacing)))     # softplus -> sigma>0

        # --- current-based LIF hidden layers (bias-free = R-STDP plastic sites) ---
        dims = [obs_dim * in_pop, *hidden, act_dim * out_pop]
        self.fc = nn.ModuleList(nn.Linear(dims[i], dims[i + 1], bias=False)
                                for i in range(len(dims) - 1))
        # SNNs are silent at init if weights are too small -> no output spikes ->
        # firing_rate=0 -> the population decoder (grad ~ firing rate) can't learn.
        # Scale up the input weights so ~10-30% of neurons fire from step 1.
        with torch.no_grad():
            for fc in self.fc:
                fc.weight.mul_(weight_gain)
        # bias current per hidden/output layer (paper's b^(k))
        self.bias = nn.ParameterList(nn.Parameter(torch.zeros(dims[i + 1]))
                                     for i in range(len(dims) - 1))

        # --- learnable population decoder: action_i = W_d^i . fr^i + b_d^i ----------
        self.dec_w = nn.Parameter(torch.randn(act_dim, out_pop) * 0.1)  # (M, P_out)
        self.dec_b = nn.Parameter(torch.zeros(act_dim))                 # (M,)

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self._raw_sigma) + 1e-3

    def _encode_currents(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B, N) -> Gaussian stimulation A_E (B, N*P_in) in [0,1]."""
        s = obs.unsqueeze(-1)                                   # (B, N, 1)
        ae = torch.exp(-0.5 * ((s - self.mu) / self.sigma) ** 2)  # (B, N, P_in)
        return ae.reshape(obs.shape[0], -1)                    # (B, N*P_in)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B, N) -> action mean (B, M)."""
        B = obs.shape[0]
        dev, dt = obs.device, obs.dtype
        a_e = self._encode_currents(obs)                       # (B, N*P_in) constant current

        # encoder IF neuron state (soft reset), one per input-population neuron
        v_enc = torch.zeros_like(a_e)
        # hidden/output current + voltage states
        cur = [torch.zeros(B, fc.out_features, device=dev, dtype=dt) for fc in self.fc]
        volt = [torch.zeros(B, fc.out_features, device=dev, dtype=dt) for fc in self.fc]
        spk_prev = [torch.zeros(B, fc.out_features, device=dev, dtype=dt) for fc in self.fc]
        out_spike_sum = torch.zeros(B, self.fc[-1].out_features, device=dev, dtype=dt)

        eps = 0.01
        thr_enc = 1.0 - eps
        for _t in range(self.T):
            # 1) encoder: deterministic soft-reset IF driven by constant A_E
            v_enc = v_enc + a_e
            enc_spk = spike(v_enc - thr_enc, self.slope)
            v_enc = v_enc - enc_spk * thr_enc
            x = enc_spk                                        # (B, N*P_in)
            # 2) current-based LIF hidden + output layers
            for k, fc in enumerate(self.fc):
                cur[k] = self.d_c * cur[k] + fc(x) + self.bias[k]
                volt[k] = self.d_v * volt[k] * (1.0 - spk_prev[k]) + cur[k]
                s_k = spike(volt[k] - self.v_th, self.slope)
                spk_prev[k] = s_k
                x = s_k
            out_spike_sum = out_spike_sum + x                  # accumulate output spikes
        # 3) population decoder
        fr = out_spike_sum / float(self.T)                     # (B, M*P_out) firing rate
        fr = fr.view(B, self.act_dim, self.out_pop)            # (B, M, P_out)
        action = (fr * self.dec_w).sum(-1) + self.dec_b        # (B, M)
        return action

    @torch.no_grad()
    def firing_rate(self, obs: torch.Tensor) -> float:
        """Mean hidden+output spike rate (H2 energy proxy), hard-threshold count."""
        B = obs.shape[0]
        dev, dt = obs.device, obs.dtype
        a_e = self._encode_currents(obs)
        v_enc = torch.zeros_like(a_e)
        cur = [torch.zeros(B, fc.out_features, device=dev, dtype=dt) for fc in self.fc]
        volt = [torch.zeros(B, fc.out_features, device=dev, dtype=dt) for fc in self.fc]
        spk_prev = [torch.zeros(B, fc.out_features, device=dev, dtype=dt) for fc in self.fc]
        tot, slots = 0.0, 0
        eps = 0.01; thr_enc = 1.0 - eps
        for _t in range(self.T):
            v_enc = v_enc + a_e
            enc_spk = (v_enc >= thr_enc).to(dt)
            v_enc = v_enc - enc_spk * thr_enc
            x = enc_spk
            for k, fc in enumerate(self.fc):
                cur[k] = self.d_c * cur[k] + fc(x) + self.bias[k]
                volt[k] = self.d_v * volt[k] * (1.0 - spk_prev[k]) + cur[k]
                s_k = (volt[k] >= self.v_th).to(dt)
                spk_prev[k] = s_k
                x = s_k
                tot += float(s_k.sum()); slots += s_k.numel()
        return tot / max(slots, 1)
