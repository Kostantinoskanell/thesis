"""L5 / H2 energy analysis: spiking locomotion policy vs the MLP baseline.

The core reason to use an SNN is energy: it does sparse accumulate (AC) ops only when
neurons spike, vs the MLP's dense multiply-accumulate (MAC) every forward pass. This
quantifies it for the locomotion policy using the Zhao et al. 2025 / Horowitz 2014
45nm method (already the M6 plan): count SynOps (spiking) vs MACs (MLP), weight by
per-op energy (AC 0.9 pJ, MAC 4.6 pJ, MUL 3.7 pJ), on REAL walking observations.

SynOps model (standard for SNNs): a layer with weight (out, in) costs, per timestep,
(#pre-neurons that fired) x out accumulate ops. Over T steps and measured per-layer
firing rate r_l:  SynOps = sum_l  in_l * out_l * r_l * T. The MLP costs sum_l in_l*out_l
MACs, once. Runs in the `nmc` env (pure torch, CPU).

Run:  conda run -n nmc python scripts/l5_energy_analysis.py
"""

from __future__ import annotations

import argparse
import sys
sys.path.insert(0, "src")

import numpy as np
import torch

from nmc.locomotion.popsan_actor import PopSpikingActorNet, encoder_spike, rect_spike

E_AC, E_MAC, E_MUL = 0.9, 4.6, 3.7   # pJ, 45nm (Horowitz 2014)
DATA = "data/l4_distill_data.npz"
T = 8


def per_layer_firing(net, obs_n):
    """Run the spiking net on a batch and return per-fc-layer mean firing rate
    (fraction of neurons firing per timestep) + the encoder input-population rate."""
    B = obs_n.shape[0]
    a_e = net._encode_currents(obs_n)
    v_enc = torch.zeros_like(a_e)
    cur = [torch.zeros(B, fc.out_features) for fc in net.fc]
    volt = [torch.zeros(B, fc.out_features) for fc in net.fc]
    spk_prev = [torch.zeros(B, fc.out_features) for fc in net.fc]
    enc_spikes = 0.0; enc_slots = 0
    layer_spikes = [0.0] * len(net.fc); layer_slots = [0] * len(net.fc)
    eps = 0.01; thr = 1.0 - eps
    with torch.no_grad():
        for _t in range(net.T):
            v_enc = v_enc + a_e
            es = (v_enc >= thr).float(); v_enc = v_enc - es * thr
            enc_spikes += float(es.sum()); enc_slots += es.numel()
            x = es
            for k, fc in enumerate(net.fc):
                cur[k] = net.d_c * cur[k] + fc(x) + net.bias[k]
                volt[k] = net.d_v * volt[k] * (1.0 - spk_prev[k]) + cur[k]
                sk = (volt[k] >= net.v_th).float(); spk_prev[k] = sk; x = sk
                layer_spikes[k] += float(sk.sum()); layer_slots[k] += sk.numel()
    enc_rate = enc_spikes / enc_slots
    layer_rates = [layer_spikes[k] / layer_slots[k] for k in range(len(net.fc))]
    return enc_rate, layer_rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="data/l4_dagger_spiking.pt",
                    help="distilled spiking policy .pt to analyze")
    ap.add_argument("--T", type=int, default=T, help="timesteps the net was trained at")
    args = ap.parse_args()
    Tsteps = args.T

    print(f"student: {args.student}  (T={Tsteps})\n")
    blob = torch.load(args.student, map_location="cpu", weights_only=True)
    mean = blob.pop("_obs_mean"); std = blob.pop("_obs_std")
    net = PopSpikingActorNet(obs_dim=48, act_dim=12, hidden=(128, 128, 128),
                             in_pop=10, out_pop=10, T=Tsteps, decoder_tanh=False)
    net.load_state_dict(blob); net.eval()

    obs = np.load(DATA)["obs"][:4096].astype(np.float32)
    obs_n = torch.as_tensor((obs - mean.numpy()) / (std.numpy() + 1e-2))
    enc_rate, layer_rates = per_layer_firing(net, obs_n)

    # synapse counts per fc layer
    fc_dims = [(fc.in_features, fc.out_features) for fc in net.fc]
    print("=== spiking policy (PopSAN) ===")
    print(f"encoder input-population firing rate: {enc_rate:.1%}")
    syn_ops = 0.0
    # layer 0's PRE is the encoder input-population (rate=enc_rate); layer k's PRE is layer k-1
    pre_rates = [enc_rate] + layer_rates[:-1]
    for k, ((i, o), r_pre) in enumerate(zip(fc_dims, pre_rates)):
        ops = i * o * r_pre * Tsteps
        syn_ops += ops
        print(f"  fc[{k}] {i}->{o}: pre-rate {r_pre:.1%}, SynOps {ops:,.0f}")
    # encoder MULs (Gaussian A_E) + decoder MACs (rate->action), small
    enc_muls = 48 * net.in_pop
    dec_macs = net.act_dim * net.out_pop
    e_spike = syn_ops * E_AC + enc_muls * E_MUL + dec_macs * E_MAC
    print(f"total SynOps (AC): {syn_ops:,.0f}  + enc {enc_muls} MUL + dec {dec_macs} MAC")
    print(f"mean hidden firing rate: {np.mean(layer_rates):.1%}")

    # MLP baseline: 48->128->128->128->12, dense MACs, one pass
    mlp_dims = [(48, 128), (128, 128), (128, 128), (128, 12)]
    mlp_macs = sum(i * o for i, o in mlp_dims)
    e_mlp = mlp_macs * E_MAC

    print("\n=== MLP baseline ===")
    print(f"MACs/decision: {mlp_macs:,}")
    print("\n=== ENERGY per decision (45nm, Horowitz 2014) ===")
    print(f"spiking: {e_spike/1000:.2f} nJ   MLP: {e_mlp/1000:.2f} nJ   "
          f"-> spiking is {e_mlp/e_spike:.2f}x {'CHEAPER' if e_spike<e_mlp else 'MORE EXPENSIVE'}")


if __name__ == "__main__":
    main()
