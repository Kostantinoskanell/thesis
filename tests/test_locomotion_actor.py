"""Unit tests for the L-track spiking locomotion actors (pure torch, no snnTorch).

Covers both the plain rate-readout actor (spiking_actor, the RateSAN ablation) and
the PopSAN-style population-coded actor (popsan_actor, the primary, per D11). Verifies
forward shape, surrogate-gradient flow to every learnable stage, and non-silent firing
(the SNN "dead at init" failure mode that zeros the population-decoder gradient).
"""

import torch
import pytest

torch = pytest.importorskip("torch")

from nmc.locomotion.spiking_actor import SpikingActorNet
from nmc.locomotion.popsan_actor import PopSpikingActorNet
from nmc.locomotion.popsan_rstdp import PopSpikingRSTDPController

N, M, B = 48, 12, 8


@pytest.mark.parametrize("neuron", ["lif", "alif"])
def test_rate_actor_forward_and_grad(neuron):
    net = SpikingActorNet(N, M, hidden=(128, 128), T=5, neuron=neuron)
    obs = torch.randn(B, N)
    a = net(obs)
    assert a.shape == (B, M)
    (a ** 2).mean().backward()
    gsum = sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None)
    assert gsum > 0
    assert 0.0 < net.firing_rate(obs) < 1.0


def test_popsan_forward_shape_and_firing():
    net = PopSpikingActorNet(N, M, hidden=(128, 128), in_pop=10, out_pop=10, T=5)
    obs = torch.randn(B, N)
    a = net(obs)
    assert a.shape == (B, M)
    fr = net.firing_rate(obs)
    assert fr > 0.0, "PopSAN silent at init -> decoder cannot learn"


def test_popsan_gradients_reach_every_stage():
    net = PopSpikingActorNet(N, M, hidden=(128, 128), in_pop=10, out_pop=10, T=5)
    obs = torch.randn(B, N)
    (net(obs) ** 2).mean().backward()
    # population decoder + hidden weights must get a real gradient
    assert net.dec_w.grad is not None and float(net.dec_w.grad.abs().sum()) > 0
    assert net.fc[-1].weight.grad is not None and float(net.fc[-1].weight.grad.abs().sum()) > 0
    # learnable Gaussian encoder params must at least be in the graph
    assert net.mu.grad is not None
    assert net._raw_sigma.grad is not None


def test_popsan_trains_on_regression():
    """A few Adam steps should reduce a fitting loss (end-to-end trainability)."""
    torch.manual_seed(0)
    net = PopSpikingActorNet(N, M, hidden=(128, 128), in_pop=10, out_pop=10, T=5)
    X = torch.randn(128, N)
    W1, W2 = torch.randn(N, 16), torch.randn(16, M)
    Y = torch.tanh(torch.tanh(X @ W1) @ W2)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    l0 = float(((net(X) - Y) ** 2).mean())
    for _ in range(150):
        opt.zero_grad(); loss = ((net(X) - Y) ** 2).mean(); loss.backward(); opt.step()
    assert float(loss) < l0, "PopSAN failed to reduce the fitting loss"


def test_rstdp_controller_updates_only_plastic_layers():
    """L4: R-STDP must change the targeted (input+readout) layers and leave the
    untargeted hidden layer alone."""
    torch.manual_seed(0)
    net = PopSpikingActorNet(N, M, hidden=(64, 64), in_pop=10, out_pop=10, T=6,
                             decoder_tanh=False)
    ctrl = PopSpikingRSTDPController(net, plastic_layers=[0, 2], anchor=0.005, reward_mode="td")
    w0, w1, w2 = (net.fc[i].weight.detach().clone().numpy() for i in range(3))

    rng = torch.Generator().manual_seed(1)
    obs = torch.randn(N, generator=rng).numpy()
    for _ in range(10):
        a = ctrl.act(obs)
        assert a.shape == (M,)
        next_obs = torch.randn(N, generator=rng).numpy()
        ctrl.learn(float(torch.randn(1, generator=rng)), next_obs=next_obs, done=False)
        obs = next_obs

    w0a, w1a, w2a = (net.fc[i].weight.detach().numpy() for i in range(3))
    assert abs(w0a - w0).sum() > 0, "plastic input layer (0) did not change"
    assert abs(w2a - w2).sum() > 0, "plastic readout layer (2) did not change"
    assert (w1a == w1).all(), "non-plastic hidden layer (1) must be untouched"
