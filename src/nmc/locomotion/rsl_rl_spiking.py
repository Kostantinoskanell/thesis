"""rsl_rl 5.x integration: a spiking actor that drops into Isaac Lab PPO (L-track L3).

rsl_rl 5.x builds the actor via `construct_algorithm`:
    actor_class = resolve_callable(cfg["actor"]["class_name"])
    actor = actor_class(obs, obs_groups, "actor", num_actions, **cfg["actor"])
so we inject by setting the actor's `class_name` to this module's class path -- no
edits to rsl_rl or Isaac Lab source.

`rsl_rl.models.MLPModel` already handles observation-group selection, normalization,
the output Gaussian distribution, log-prob / KL, and JIT/ONNX export. The ONLY thing
that must change for a spiking policy is the network that maps the observation latent
to the distribution parameters. So SpikingActorMLPModel subclasses MLPModel and
replaces `self.mlp` with the PopSAN population-coded spiking net (D11) -- the classic
PopSAN hybrid: spiking actor, standard everything-else (and the critic stays a plain
MLPModel). Only the deployed *policy* is spiking, exactly as PopSAN prescribes.

This module imports rsl_rl, so it only runs inside the Isaac Sim conda env (not the
`nmc` env). Keep the spiking net itself (popsan_actor.py) rsl_rl-free so it stays
unit-testable on Windows.
"""

from __future__ import annotations

from rsl_rl.models import MLPModel

from nmc.locomotion.popsan_actor import PopSpikingActorNet


class SpikingActorMLPModel(MLPModel):
    """MLPModel whose body is a PopSAN population-coded spiking network.

    Extra kwargs (beyond MLPModel's) come straight from cfg["actor"] and tune the
    spiking net; all have safe defaults so a minimal cfg override works.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim,
                 hidden_dims=(256, 256), activation: str = "elu",
                 obs_normalization: bool = False, distribution_cfg=None,
                 in_pop: int = 10, out_pop: int = 10, spiking_T: int = 5,
                 neuron_d_c: float = 0.5, neuron_d_v: float = 0.75,
                 v_th: float = 0.5, weight_gain: float = 3.0, **_ignored):
        # Build the standard MLPModel first (obs handling, normalizer, distribution,
        # and a throwaway MLP body we immediately replace).
        super().__init__(obs, obs_groups, obs_set, output_dim,
                         hidden_dims=hidden_dims, activation=activation,
                         obs_normalization=obs_normalization,
                         distribution_cfg=distribution_cfg)

        in_dim = self._get_latent_dim()
        # network must output the distribution's parameter vector (mean for a
        # state-independent-std Gaussian) -> distribution.input_dim, else output_dim.
        out_dim = self.distribution.input_dim if self.distribution is not None else output_dim

        # Replace the ANN body with the PopSAN spiking net (same in/out contract).
        self.mlp = PopSpikingActorNet(
            obs_dim=in_dim, act_dim=out_dim, hidden=tuple(hidden_dims),
            in_pop=in_pop, out_pop=out_pop, T=spiking_T,
            d_c=neuron_d_c, d_v=neuron_d_v, v_th=v_th, weight_gain=weight_gain,
        )
        # MLPModel calls distribution.init_mlp_weights(self.mlp) on the ANN body; it
        # assumes an nn.Sequential MLP and doesn't apply to the spiking net, which
        # brings its own init. Skip it (guarded) -- the spiking net is already sane.
        if self.distribution is not None:
            try:
                self.distribution.init_mlp_weights(self.mlp)
            except (AttributeError, IndexError, TypeError):
                pass

    @property
    def firing_rate_last(self) -> float:
        """Convenience passthrough for the H2 energy proxy (L5)."""
        return getattr(self.mlp, "_last_firing_rate", float("nan"))
