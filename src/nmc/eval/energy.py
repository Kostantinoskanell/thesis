"""Operation-level energy model for the nav-layer controllers (M6 / H2).

Method follows the SNN-energy literature (Horowitz 2014 45 nm per-op costs, as
used by Zhao et al. 2025 and PopSAN; same model L5 used for the locomotion
layer, so the two tracks' numbers are directly comparable):

    SNN   : a synapse costs one ACCUMULATE, but only when its presynaptic
            neuron spikes  ->  SynOps = sum_l  in_l * out_l * r_pre_l * T
    ANN   : every synapse costs one MULTIPLY-ACCUMULATE, every forward pass

Three deliberate extensions over the plain literature model, because the plain
model flatters SNNs and this thesis should not:

1. **Neuron-state updates are counted.** At the nav SNN's ~1% firing rate the
   per-neuron membrane/threshold arithmetic (paid every tick for every neuron,
   spiking or not) is *not* negligible next to the sparse SynOps. Reporting
   SynOps alone would hide that.
2. **The spike encoder is counted.** Rate/TTFS encoding of the 37-dim
   observation is real work the MLP never does (the RNG draw is charged at
   MUL cost -- an assumption, flagged in the output).
3. **Online learning is counted.** R-STDP, TD(lambda) backprop and TM-NORM all
   pay a per-decision *update* cost on top of inference. For R-STDP this is
   reported twice: as the dense reference implementation actually runs
   (`stdp.STDPLearner.step`, which touches every synapse every tick), and as
   the **event-driven lower bound** a crossbar/FPGA datapath would achieve
   (only synapses adjacent to a spike, lazy eligibility decay). The gap
   between those two is the quantified target for M7/M8.

Energies are returned in picojoules; helper `.total_pj` / `.as_nj()` aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Horowitz (2014) 45 nm, 32-bit float arithmetic.
E_AC = 0.9      # pJ, add
E_MUL = 3.7     # pJ, multiply
E_MAC = 4.6     # pJ, multiply-accumulate (= MUL + AC)


@dataclass
class OpCount:
    """Raw operation counts, before energy weighting."""
    ac: float = 0.0
    mul: float = 0.0
    mac: float = 0.0

    def __add__(self, other: "OpCount") -> "OpCount":
        return OpCount(self.ac + other.ac, self.mul + other.mul, self.mac + other.mac)

    @property
    def energy_pj(self) -> float:
        return self.ac * E_AC + self.mul * E_MUL + self.mac * E_MAC


@dataclass
class Breakdown:
    """Per-decision energy, split by where it is spent."""
    label: str
    inference: OpCount = field(default_factory=OpCount)   # synapses / MACs
    neurons: OpCount = field(default_factory=OpCount)     # membrane state updates
    encoder: OpCount = field(default_factory=OpCount)     # spike encoding
    learning: OpCount = field(default_factory=OpCount)    # online weight update

    @property
    def total_pj(self) -> float:
        return (self.inference.energy_pj + self.neurons.energy_pj
                + self.encoder.energy_pj + self.learning.energy_pj)

    def as_nj(self) -> dict:
        return {"inference": self.inference.energy_pj / 1e3,
                "neurons": self.neurons.energy_pj / 1e3,
                "encoder": self.encoder.energy_pj / 1e3,
                "learning": self.learning.energy_pj / 1e3,
                "total": self.total_pj / 1e3}


# -- SNN ------------------------------------------------------------------

def snn_synops(fc_dims, pre_rates, T: int) -> float:
    """SynOps = sum_l in_l * out_l * r_pre_l * T.

    fc_dims   : [(in, out), ...] per linear layer
    pre_rates : firing rate of each layer's PREsynaptic population
                (layer 0's is the encoder output rate)
    """
    return sum(i * o * r * T for (i, o), r in zip(fc_dims, pre_rates))


def snn_neuron_ops(layer_sizes, T: int, neuron: str = "alif") -> OpCount:
    """Membrane-state arithmetic, paid per neuron per tick whether it spikes or not.

    LIF  (snn.Leaky):  mem = beta*mem + in; spk = mem>thr; mem -= spk*thr
                       -> 2 MUL, 3 AC
    ALIF (Bellec):     adds  thr = v_th + beta_adapt*a  and  a = rho*a + spk
                       -> 4 MUL, 5 AC
    """
    per_neuron_mul, per_neuron_ac = (4, 5) if neuron == "alif" else (2, 3)
    n_updates = sum(layer_sizes) * T
    return OpCount(ac=n_updates * per_neuron_ac, mul=n_updates * per_neuron_mul)


def nav_encoder_ops(T: int, n_rate_ch: int = 41, n_ttfs_ch: int = 32) -> OpCount:
    """encode_nav_obs: Poisson rate coding on 41 channels + TTFS on 32.

    Rate: one MUL per channel for p_spike, then per tick a uniform draw
    (charged at MUL cost -- an assumption; a hardware LFSR is far cheaper) and
    a compare (AC). TTFS: computed once per decision, ~2 MUL per channel.
    """
    return OpCount(ac=T * n_rate_ch,
                   mul=n_rate_ch + T * n_rate_ch + 2 * n_ttfs_ch)


def snn_decode_ops(n_pops: int, pop_size: int) -> OpCount:
    """Population vote: sum each population, argmax across populations."""
    return OpCount(ac=n_pops * pop_size + n_pops)


def snn_inference(fc_dims, pre_rates, layer_sizes, T, neuron="alif",
                  n_pops=4, pop_size=16, label="SNN") -> Breakdown:
    synops = snn_synops(fc_dims, pre_rates, T)
    return Breakdown(
        label=label,
        inference=OpCount(ac=synops) + snn_decode_ops(n_pops, pop_size),
        neurons=snn_neuron_ops(layer_sizes, T, neuron),
        encoder=nav_encoder_ops(T),
    )


# -- MLP ------------------------------------------------------------------

def mlp_inference(fc_dims, n_layernorm_dims=(512, 512), label="MLP") -> Breakdown:
    """Dense MACs + biases, plus LayerNorm (mean/var/normalize/affine) and ReLU."""
    macs = sum(i * o for i, o in fc_dims)
    biases = sum(o for _, o in fc_dims)
    ops = OpCount(mac=macs, ac=biases)
    for d in n_layernorm_dims:
        # mean (d AC), var (d MUL + d AC), normalize (d MUL + d AC), affine (d MUL + d AC)
        ops = ops + OpCount(ac=4 * d, mul=3 * d)
        ops = ops + OpCount(ac=d)          # ReLU compare
    return Breakdown(label=label, inference=ops)


# -- online learning ------------------------------------------------------

def rstdp_update_dense(plastic_dims, T: int) -> OpCount:
    """Cost of `STDPLearner.step` as actually implemented, per decision.

    Per tick per plastic layer the reference implementation touches the FULL
    (n_post x n_pre) array five times: zero dW, eligibility decay+accumulate
    (MUL+AC), delta = eta*r*elig (MUL), W += delta (AC), clip (AC). The
    Hebbian terms themselves are sparse (only rows/cols of spiking neurons)
    and are folded in by the caller via `spike_driven` below.
    """
    ops = OpCount()
    for n_pre, n_post in plastic_dims:
        n_syn = n_pre * n_post
        ops = ops + OpCount(ac=T * n_syn * 4, mul=T * n_syn * 2)
        ops = ops + OpCount(mul=T * (n_pre + n_post), ac=T * (n_pre + n_post))
    return ops


def rstdp_update_event_driven(plastic_dims, pre_rates, post_rates, T: int) -> OpCount:
    """Event-driven lower bound: the crossbar/FPGA formulation (M7/M8 target).

    Only synapses adjacent to a neuron that actually spiked are touched
    (LTP: fired_post x n_pre, LTD: fired_pre x n_post), eligibility decay is
    lazy (applied on touch via a stored timestamp, no dense sweep), and
    consolidation only visits synapses with nonzero eligibility -- bounded by
    the same touched set. Trace decay stays per-neuron, not per-synapse.
    """
    ops = OpCount()
    for (n_pre, n_post), r_pre, r_post in zip(plastic_dims, pre_rates, post_rates):
        touched = T * (r_post * n_post * n_pre + r_pre * n_pre * n_post)
        ops = ops + OpCount(mul=touched, ac=touched)      # Hebbian + lazy decay
        ops = ops + OpCount(mul=touched, ac=touched)      # consolidate + apply
        ops = ops + OpCount(mul=T * (n_pre + n_post))     # per-neuron trace decay
    return ops


def backprop_update(n_params: int, fwd_macs: float) -> OpCount:
    """TD(lambda) actor-critic update (OnlineMLP): backward pass + trace + apply.

    Backward is charged at the standard 2x forward MACs; the critic's V(s')
    costs one extra forward pass; the eligibility trace is one MUL + one AC per
    parameter and the weight update two MUL + one AC per parameter.
    """
    return OpCount(mac=2 * fwd_macs + fwd_macs,
                   mul=n_params * 3, ac=n_params * 2)


def tmnorm_update(layer_sizes, T: int) -> OpCount:
    """EMA membrane mean/var per neuron + threshold rescale (no weight update)."""
    n_vals = sum(layer_sizes) * T
    n_neurons = sum(layer_sizes)
    return OpCount(ac=2 * n_vals + 2 * n_neurons,     # sum, sum-of-squares, EMA
                   mul=n_vals + 3 * n_neurons)        # squares, sqrt/scale/shift
