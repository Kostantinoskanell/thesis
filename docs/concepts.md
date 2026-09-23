---
layout: default
title: How the SNN Pieces Work
---

<a id="top"></a>

<nav class="site-nav" aria-label="Section navigation">
  <a href="index.md#d-track" class="nav-d">D &middot; foundation</a>
  <a href="index.md#m-track" class="nav-m">M &middot; navigation</a>
  <a href="index.md#l-track" class="nav-l">L &middot; locomotion</a>
  <a href="#top" class="nav-c current">Concepts</a>
  <a href="https://github.com/Kostantinoskanell/thesis/blob/main/docs/references/sota_decisions.md">Decision log</a>
  <a href="https://github.com/Kostantinoskanell/thesis" class="nav-gh">GitHub</a>
</nav>

[&larr; Back to project overview](index.md)

# How the SNN pieces work, and where they sit in this project

This page is a primer on the spiking-network mechanisms used in the thesis —
LIF/ALIF neurons, STDP, R-STDP, e-prop, population coding, and the memristor
analogy — each with the actual equation this codebase implements and a
pointer to where it lives in the code. For *why* each choice was made over
the alternatives, see the full dated log: [`sota_decisions.md`](https://github.com/Kostantinoskanell/thesis/blob/main/docs/references/sota_decisions.md).

<img src="assets/img/concepts.svg" alt="LIF neuron, STDP, R-STDP, memristor analogy">

<table class="quick-links" markdown="0">
<thead><tr><th>#</th><th>Mechanism</th><th>Where it's used here</th></tr></thead>
<tbody>
<tr><td>1</td><td><a href="#lif-alif">LIF / ALIF neuron</a></td><td>every controller; ALIF alone explains most of the shift-robustness found</td></tr>
<tr><td>2</td><td><a href="#stdp">STDP</a></td><td>the Hebbian ablation &mdash; worst performer everywhere tested</td></tr>
<tr><td>3</td><td><a href="#r-stdp">R-STDP</a></td><td>the project's original rule &mdash; no surviving significant win (D16/D18)</td></tr>
<tr><td>4</td><td><a href="#e-prop">e-prop</a></td><td>structurally different 3-factor rule &mdash; also no significant win (D19)</td></tr>
<tr><td>5</td><td><a href="#population-encoding">Population encoding</a></td><td>replaces stochastic rate coding to chase the H4 noise finding (D4)</td></tr>
<tr><td>6</td><td><a href="#memristor-analogy">Memristor analogy</a></td><td>the FPGA co-processor's computational target (M7/M8)</td></tr>
</tbody>
</table>

## 1. LIF / ALIF neuron — the spiking unit {: #lif-alif}

A leaky integrate-and-fire (LIF) neuron accumulates input current onto a
membrane potential that leaks over time, and fires a binary spike when the
potential crosses a threshold:

```
mem[t] = beta * mem[t-1] + W @ x[t]
spike[t] = 1  if mem[t] > v_th   else 0
mem[t]  -= spike[t] * v_th        # subtractive reset
```

**ALIF** (adaptive LIF) adds a second state variable, an adaptation trace `a`
that rises on every spike and decays slowly, and is added to the threshold —
a neuron that has fired recently becomes harder to fire again:

```
thr[t]  = v_th + beta_adapt * a[t]
a[t]    = rho * a[t-1] + spike[t]
```

This single extra term gives the neuron a longer memory of its own recent
activity, which turned out to matter a lot here: an ablation isolating the
neuron model alone (frozen LIF vs. frozen ALIF, no plasticity at all) found
**+10 points of shift-robustness from switching neuron models**, more than
any plasticity rule managed on top of it (see the M-track table on the
[home page](index.md)). Code: [`src/nmc/controllers/snn.py`](https://github.com/Kostantinoskanell/thesis/blob/main/src/nmc/controllers/snn.py), [`src/nmc/plasticity/eprop.py`](https://github.com/Kostantinoskanell/thesis/blob/main/src/nmc/plasticity/eprop.py) (`ALIFEpropLayer`).

## 2. STDP — Hebbian, no notion of task success {: #stdp}

Spike-timing-dependent plasticity strengthens a synapse when the presynaptic
neuron fires *just before* the postsynaptic one (it plausibly contributed to
the spike) and weakens it for the reverse order:

```
dw = +A_plus  * exp(-dt / tau)   if pre fires before post   (LTP)
dw = -A_minus * exp( dt / tau)   if post fires before pre   (LTD)
```

Pure STDP has no idea whether the outcome was good or bad — it just
correlates timing. In every controller comparison run in this project, pure
STDP is the worst performer by a wide margin, which is expected: it is the
ablation that shows plasticity *without* a task signal is actively harmful
here, not neutral. Code: [`src/nmc/plasticity/stdp.py`](https://github.com/Kostantinoskanell/thesis/blob/main/src/nmc/plasticity/stdp.py) (also the from-scratch
golden reference the FPGA datapath is checked against).

## 3. R-STDP — three-factor, one global reward {: #r-stdp}

Reward-modulated STDP adds a third factor. The Hebbian STDP update above
still runs every timestep, but instead of being applied immediately it
accumulates into an **eligibility trace** `e_ij` (a short-term memory of
"this synapse recently correlated with this pattern"). Only when a reward or
TD-error signal `r(t)` arrives is the accumulated trace actually consolidated
into the weight, and the *same scalar* gates every synapse in the network:

```
e_ij(t) = decay * e_ij(t-1) + stdp_update(pre, post)
dw_ij   = eta * r(t) * e_ij(t)          # r(t) identical for every synapse
```

This is the project's original, "memristor-native" plasticity rule: a
conductance that only moves when a single global dopamine-like signal says
to. It is biologically the simplest three-factor rule and maps directly onto
a crossbar (per-synapse eligibility register, one broadcast wire for the
third factor). Code: [`src/nmc/plasticity/stdp.py`](https://github.com/Kostantinoskanell/thesis/blob/main/src/nmc/plasticity/stdp.py), wrapped by [`src/nmc/controllers/snn.py`](https://github.com/Kostantinoskanell/thesis/blob/main/src/nmc/controllers/snn.py)'s R-STDP controller.

**What we found:** across six independent tests (sensor dropout at two
severities, terrain sand, terrain ice, a custom obstacle map, a
severity/plasticity-scope sweep), R-STDP never produced a statistically
significant recovery advantage over a frozen, non-plastic network — and on
terrain sand it was significantly *worse* than a frozen MLP. That null result
is documented honestly on the [home page](index.md) and in [`sota_decisions.md` D16/D18](https://github.com/Kostantinoskanell/thesis/blob/main/docs/references/sota_decisions.md).
It raised an obvious question: is the *shift* unrecoverable by any local
learning rule, or is R-STDP specifically too crude?

## 4. e-prop — same three-factor shape, a different eligibility trace and a per-neuron third factor {: #e-prop}

e-prop (Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein & Maass, *Nature
Communications* 2020) answers that question by changing exactly two things
relative to R-STDP and holding everything else fixed — same pretrained ALIF
network, same TD-error-based reward, same elastic weight anchoring back to
the pretrained checkpoint:

1. **The eligibility trace** is no longer a Hebbian pre/post correlation —
   it is an online approximation of the true backprop-through-time gradient,
   derived from each neuron's own membrane and adaptation dynamics:

   ```
   h[t]      = pseudo_derivative(mem[t] - thr[t])          # surrogate gradient
   eps_a[t]  = h[t-1]*zbar[t-1] + (rho - h[t-1]*beta_adapt) * eps_a[t-1]
   e[t]      = h[t] * (zbar[t] - beta_adapt * eps_a[t])
   ```

   `zbar` is a low-pass filter of the presynaptic spike train. The
   `pseudo_derivative` is the *exact same* FastSigmoid surrogate-gradient
   formula already used to pretrain the network with real BPTT (M3), so
   e-prop's online trace and the offline training gradient are the same
   function — not two independently-chosen approximations.

2. **The third factor is per-neuron, not global.** A contrastive,
   REINFORCE-style credit is computed at the output layer (push the chosen
   action's population up, pull the others down, scaled by TD-error), then
   projected backward through the *current* readout weights layer by layer
   ("symmetric feedback") — so each neuron gets its own share of credit
   instead of one broadcast scalar:

   ```
   credit[chosen_pop]      = td_error
   credit[other_pops]      = -td_error / (n_pops - 1)
   layer_credit[i]         = W[i+1].T @ layer_credit[i+1]      # symmetric feedback
   dW[i]                   = eta * layer_credit[i][:, None] * e_acc[i]
   ```

**Honest scope note:** Bellec et al. derive e-prop for supervised sequence
learning with a known per-timestep target. This project's task is closed-loop
RL with only a scalar TD-error, so the per-neuron learning signal above is
the natural RL adaptation used in the eprop-for-RL literature, not a claim of
exact paper fidelity — it is labelled as such everywhere it appears in the
code and the decision log.

Both rules share the identical hardware shape (a per-synapse eligibility
register times a broadcast or per-neuron third factor), so the FPGA datapath
already scoped for R-STDP supports e-prop too — this was a software
comparison, not a redesign. Code: [`src/nmc/plasticity/eprop.py`](https://github.com/Kostantinoskanell/thesis/blob/main/src/nmc/plasticity/eprop.py), [`src/nmc/controllers/eprop_snn.py`](https://github.com/Kostantinoskanell/thesis/blob/main/src/nmc/controllers/eprop_snn.py). Decision record: [`sota_decisions.md` D1](https://github.com/Kostantinoskanell/thesis/blob/main/docs/references/sota_decisions.md).

**Result:** tested under the identical sensor-dropout protocol R-STDP was
tested under (10 seeds x 30 eps, Holm-corrected, with Frozen SNN and R-STDP
SNN re-run fresh in the same pass for a real matched-data comparison) —
**e-prop also does not show a statistically significant advantage over a
frozen, non-plastic network** (best case +9.3 points at dropout=0.20, Holm
p=0.078). This is a meaningful result, not a dead end: it means the null
result found for R-STDP generalizes beyond R-STDP specifically, pointing at
the neuron model (see §1) rather than online plasticity as the actual source
of shift-robustness in this setup. Full record: [`archive/D1_eprop/README.md`](https://github.com/Kostantinoskanell/thesis/blob/main/archive/D1_eprop/README.md).

## 5. Population encoding — replacing sampling noise with a learned code {: #population-encoding}

Every controller above turns a continuous observation (LiDAR ranges, goal
bearing, etc.) into spikes with **stochastic rate coding**: each channel's
value sets a firing *probability*, and spikes are sampled from it every
timestep. That sampling is a source of noise independent of the underlying
signal — and it turned out to be the root cause of an unexpected finding
(H4): the SNN degraded *faster* than the MLP as sensor noise increased,
because the noise compounded with the encoder's own sampling noise rather
than being smoothed by it.

The alternative implemented here is a **learned Gaussian population code**
(PopSAN-style, deterministic): each input channel is represented by a small
population of neurons with fixed, spread-out preferred values, and the
*injected current* for each neuron is a Gaussian function of the distance
between the true value and that neuron's preferred value — no sampling step
at all, and the tuning widths are learned end-to-end rather than hand-set.
Code: [`src/nmc/controllers/snn_popenc.py`](https://github.com/Kostantinoskanell/thesis/blob/main/src/nmc/controllers/snn_popenc.py), training: [`scripts/train_snn_popenc_go2.py`](https://github.com/Kostantinoskanell/thesis/blob/main/scripts/train_snn_popenc_go2.py).

## 6. The memristor analogy {: #memristor-analogy}

The FPGA co-processor (planned, M7/M8) does not model memristor device
physics — it reproduces the *computational structure* a memristor crossbar
provides: a conductance `g` bounded in `[g_min, g_max]` standing in for a
synaptic weight, which is exactly what every plasticity rule above already
assumes (`np.clip(W, w_min, w_max)` after every update). The three properties
that matter for the hardware mapping — state retention between updates,
saturation at bounds, and locality (each update only needs that synapse's own
pre/post/eligibility state, never a full network pass) — are what all three
rules (STDP, R-STDP, e-prop) share, which is why the same crossbar datapath
was scoped to support all of them without a redesign.

---

*Back to [project overview](index.md) &middot; full decision log: [`sota_decisions.md`](https://github.com/Kostantinoskanell/thesis/blob/main/docs/references/sota_decisions.md) &middot; every bug: [`docs/debug-log`](https://github.com/Kostantinoskanell/thesis/tree/main/docs/debug-log/).*

<a href="#top" class="back-to-top">&uarr; top</a>
