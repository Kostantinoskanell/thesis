"""PopEncNavNet: the nav-layer SNN with its stochastic rate/TTFS encoder
(encode_nav_obs) replaced by PopSAN's LEARNABLE Gaussian population encoder
(Tang et al. 2020), already used for the L-track's locomotion actor
(src/nmc/locomotion/popsan_actor.py) -- ported here rather than reinvented.

Why (D4 in docs/references/sota_decisions.md): H4 (this project, M6) measured
that the rate-coding SNN is LESS noise-robust than the MLP, not more, and
traced it to a specific mechanism -- Poisson rate coding SAMPLES the sensor
value stochastically every decision, so additive sensor noise compounds with
sampling noise instead of being filtered out. A population code with
per-neuron LEARNABLE receptive fields (mu, sigma) produces a smooth,
DETERMINISTIC current (no per-timestep resampling) that a downstream ALIF
layer still processes as spikes -- removing that specific noise source while
keeping the network genuinely spiking throughout. This directly tests whether
that mechanism, not "SNNs in general", was the cause of H4's refutation.

Same ALIF hidden layers, same population-vote decode, same architecture shape
as the existing LIFNet -- only the FRONT END changes. Trained from scratch by
BPTT on the same DAgger dataset M3 used (scripts/train_snn_popenc_go2.py),
since the input representation is entirely different and cannot reuse M3's
pretrained weights.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nmc.controllers.snn import ALIFCell, _LeakyCell, _HAS_SNNTORCH

if _HAS_SNNTORCH:
    from snntorch import surrogate


class PopEncNavNet(nn.Module):
    def __init__(self, obs_dim: int = 37, hidden=(512, 512), n_pops: int = 4,
                 pop_size: int = 16, in_pop: int = 10, beta: float = 0.95,
                 neuron: str = "alif", enc_min: float = -1.0, enc_max: float = 1.0,
                 enc_sigma: float = 0.3872983):
        super().__init__()
        if not _HAS_SNNTORCH:
            raise ImportError("snntorch is required to instantiate PopEncNavNet")
        self.obs_dim = obs_dim
        self.in_pop = in_pop
        self.n_pops = n_pops
        self.pop_size = pop_size
        self.out_dim = n_pops * pop_size
        self.neuron = neuron

        mu0 = torch.linspace(enc_min, enc_max, in_pop).unsqueeze(0).repeat(obs_dim, 1)
        self.mu = nn.Parameter(mu0)                                        # (N, P_in)
        raw_sigma0 = math.log(math.exp(enc_sigma) - 1.0)
        self._raw_sigma = nn.Parameter(torch.full((obs_dim, in_pop), float(raw_sigma0)))

        spike_grad = surrogate.fast_sigmoid()
        dims = [obs_dim * in_pop, *hidden, self.out_dim]
        self.fc = nn.ModuleList(nn.Linear(dims[i], dims[i + 1], bias=False)
                                for i in range(len(dims) - 1))
        if neuron == "alif":
            self.lif = nn.ModuleList(ALIFCell(beta=beta, spike_grad=spike_grad)
                                     for _ in range(len(dims) - 1))
        elif neuron == "lif":
            self.lif = nn.ModuleList(_LeakyCell(beta=beta, spike_grad=spike_grad)
                                     for _ in range(len(dims) - 1))
        else:
            raise ValueError(f"unknown neuron type {neuron!r}")

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self._raw_sigma) + 1e-3

    def encode_currents(self, obs_norm: torch.Tensor) -> torch.Tensor:
        """obs_norm: (B, obs_dim), already normalized to roughly [enc_min, enc_max]
        (see normalize_nav_obs). Returns (B, obs_dim*in_pop) Gaussian stimulation,
        a CONSTANT current fed at every timestep -- no per-timestep resampling."""
        s = obs_norm.unsqueeze(-1)                                # (B, N, 1)
        ae = torch.exp(-0.5 * ((s - self.mu) / self.sigma) ** 2)  # (B, N, P_in)
        return ae.view(obs_norm.shape[0], -1)

    def forward(self, obs_norm: torch.Tensor, T: int, return_mem: bool = False):
        """obs_norm: (B, obs_dim). Returns (out_sum (B, out_dim), spikes list[T][layer])."""
        a_e = self.encode_currents(obs_norm)
        states = [cell.init_state() for cell in self.lif]
        out_sum = 0
        all_spikes = []
        all_mems = [] if return_mem else None
        for _t in range(T):
            cur = a_e
            step_spikes = []
            step_mems = [] if return_mem else None
            for i, (fc, cell) in enumerate(zip(self.fc, self.lif)):
                cur, states[i] = cell.step(fc(cur), states[i])
                step_spikes.append(cur)
                if return_mem:
                    mem = states[i][0] if isinstance(states[i], tuple) else states[i]
                    step_mems.append(mem)
            out_sum = out_sum + cur
            all_spikes.append(step_spikes)
            if return_mem:
                all_mems.append(step_mems)
        if return_mem:
            return out_sum, all_spikes, all_mems
        return out_sum, all_spikes

    def decode(self, out_sum: torch.Tensor) -> int:
        votes = out_sum.view(-1, self.n_pops, self.pop_size).sum(-1)
        return int(votes.argmax(-1).item())


# -- shared normalization: maps the raw 37-dim nav observation into roughly
# [-1, 1] per channel, using the SAME per-channel conventions as
# encode_nav_obs / scripts/train_snn_go2.py's _encode_batch, so the population
# encoder sees the same information the rate/TTFS encoder did, just represented
# differently -- an apples-to-apples encoder swap, not a different observation.
N_LIDAR = 32
_ARENA_HALF_M, _V_MAX, _OMEGA_MAX = 5.0, 0.8, 1.5


def normalize_nav_obs(obs):
    """obs: (..., 37) numpy or torch, raw nav observation. Returns same shape,
    each channel roughly in [-1, 1] (lidar and v in [-1,1] too, just using
    the positive half)."""
    import numpy as _np
    is_torch = isinstance(obs, torch.Tensor)
    lib = torch if is_torch else _np
    lidar = obs[..., :N_LIDAR]
    lidar_n = 2.0 * lib.clip(lidar, 0.0, 1.0) - 1.0 if not is_torch else 2.0 * torch.clamp(lidar, 0, 1) - 1.0
    gx = obs[..., N_LIDAR + 0] / _ARENA_HALF_M
    gy = obs[..., N_LIDAR + 1] / _ARENA_HALF_M
    h = obs[..., N_LIDAR + 2] / math.pi
    v = obs[..., N_LIDAR + 3] / _V_MAX
    w = obs[..., N_LIDAR + 4] / _OMEGA_MAX
    if is_torch:
        gx, gy, h, v, w = (torch.clamp(x, -1, 1) for x in (gx, gy, h, v, w))
        scalars = torch.stack([gx, gy, h, v, w], dim=-1)
        return torch.cat([lidar_n, scalars], dim=-1)
    else:
        gx, gy, h, v, w = (_np.clip(x, -1, 1) for x in (gx, gy, h, v, w))
        scalars = _np.stack([gx, gy, h, v, w], axis=-1)
        return _np.concatenate([lidar_n, scalars], axis=-1)


class PopEncNavController:
    """Frozen (no plasticity) closed-loop wrapper -- the ablation this
    encoder swap needs first: does it match M3's original rate/TTFS-encoded
    SNN pre-shift, before drawing any H4/H1 conclusions from it? Same
    act(obs)->int / observe(...) interface as every other controller here."""

    def __init__(self, net: PopEncNavNet, n_steps: int, device: str = "cpu"):
        self.net = net.to(device).eval()
        self.n_steps = n_steps
        self.device = device
        self.total_spikes = 0.0
        self.total_neuron_steps = 0
        self.n_decisions = 0

    def act(self, obs) -> int:
        import numpy as np
        obs_norm = normalize_nav_obs(np.asarray(obs, dtype=np.float32))
        obs_t = torch.as_tensor(obs_norm, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out_sum, all_spikes = self.net(obs_t, self.n_steps)
        for step in all_spikes:
            for layer_spk in step:
                self.total_spikes += float(layer_spk.sum())
                self.total_neuron_steps += layer_spk.numel()
        self.n_decisions += 1
        return self.net.decode(out_sum)

    def observe(self, reward: float, next_obs, done: bool) -> None:
        pass  # frozen -- no plasticity

    def spike_stats(self) -> dict:
        if self.n_decisions == 0:
            return {"spikes_per_decision": 0.0, "firing_rate": 0.0}
        return {"spikes_per_decision": self.total_spikes / self.n_decisions,
                "firing_rate": self.total_spikes / max(self.total_neuron_steps, 1)}
