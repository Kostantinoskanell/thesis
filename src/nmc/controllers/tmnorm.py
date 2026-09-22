"""TM-NORM baseline controller (D10).

Threshold Modulation (TM-NORM) is an online test-time adaptation method for SNNs
that recalibrates each ALIF neuron's firing threshold from an EMA of its own
membrane-potential mean and variance (batch-norm on the membrane potential).
No backprop, no reward signal, no weight updates.
"""
from __future__ import annotations

import numpy as np
import torch
from nmc.controllers.snn import SNNNavController, ALIFCell

class TMNormController(SNNNavController):
    def __init__(self, net, n_steps=20, device="cpu", seed=0, ema_alpha=0.01):
        super().__init__(net, n_steps=n_steps, plasticity_enabled=False, device=device, seed=seed)
        self.ema_alpha = ema_alpha
        self.mus = {}
        self.vars = {}
        self.v_th_targets = {}
        
        # Initialize target thresholds (assumed scalar 1.0 or tensor)
        for i, cell in enumerate(self.net.lif):
            if isinstance(cell, ALIFCell):
                self.mus[i] = None
                self.vars[i] = None
                # Store the original threshold value
                val = cell.v_th
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().numpy()
                self.v_th_targets[i] = float(val) if np.isscalar(val) else val

    def act(self, obs) -> int:
        from nmc.encoding.spike_encoding import encode_nav_obs
        raster = encode_nav_obs(obs, self.n_steps, rng=self.rng)     # (T, F)
        x_seq = torch.as_tensor(raster, device=self.device).unsqueeze(1)  # (T, 1, F)
        
        with torch.no_grad():
            out_sum, all_spikes, all_mems = self.net(x_seq, return_mem=True)
            
        # all_mems has length T, each item is a list of membrane potentials for each layer
        for i, cell in enumerate(self.net.lif):
            if isinstance(cell, ALIFCell):
                # Gather mems for layer i across T timesteps, shape: (T, 1, out_dim)
                layer_mems = torch.stack([torch.as_tensor(m[i]) for m in all_mems]).detach().cpu().numpy()
                
                # Compute mean and variance over time and batch
                mu = np.mean(layer_mems, axis=(0, 1))
                var = np.var(layer_mems, axis=(0, 1))
                
                if self.mus[i] is None:
                    self.mus[i] = mu
                    self.vars[i] = var
                else:
                    self.mus[i] = (1 - self.ema_alpha) * self.mus[i] + self.ema_alpha * mu
                    self.vars[i] = (1 - self.ema_alpha) * self.vars[i] + self.ema_alpha * var
                
                sigma = np.sqrt(self.vars[i] + 1e-8)
                # Recalibrate threshold so that normalized membrane potential >= original target threshold
                new_v_th = self.v_th_targets[i] * sigma + self.mus[i]
                # The ALIF reset step is `mem -= spk * thr`: it assumes thr > 0 (a
                # positive threshold subtracted off after firing). If recalibration
                # ever pushes v_th <= 0, that reset flips sign and *adds* energy to
                # the membrane on every spike instead of removing it -- a runaway
                # feedback loop (confirmed empirically: threshold drifts unbounded
                # negative, net always mis-fires, success -> 0%). Floor it at a small
                # positive value so the reset physics stays sane.
                new_v_th = np.maximum(new_v_th, 0.05)

                if not isinstance(cell.v_th, torch.Tensor):
                    cell.v_th = torch.tensor(new_v_th, dtype=torch.float32, device=self.device)
                else:
                    cell.v_th.data.copy_(torch.tensor(new_v_th, dtype=torch.float32, device=self.device))
                    
        self._account(raster, all_spikes)
        return self.net.decode(out_sum)
