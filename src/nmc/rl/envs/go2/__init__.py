"""Registers "Go2JoystickFlatTerrain" into mujoco_playground's locomotion
registry on import, so `mujoco_playground.registry.load("Go2JoystickFlatTerrain")`
works exactly like the built-in Go1/G1/Spot envs.

Playground has no built-in Go2 env (only Go1, Spot, Barkour, ... quadrupeds/
bipeds) -- this ports the Go1 joystick task to Go2's morphology, following the
documented "adding custom environments to Playground" pattern (register into
the private but stable `_envs`/`_cfgs`/`_randomizer` dicts of
`mujoco_playground._src.locomotion`) rather than editing the installed
package. Import this module (or `nmc.rl.envs.go2`) before calling
`registry.load`.
"""

import functools

from mujoco_playground._src import locomotion as _playground_locomotion

from nmc.rl.envs.go2 import joystick as go2_joystick
from nmc.rl.envs.go2 import randomize as go2_randomize

ENV_NAME = "Go2JoystickFlatTerrain"

_playground_locomotion.register_environment(
    ENV_NAME,
    functools.partial(go2_joystick.Joystick, task="flat_terrain"),
    go2_joystick.default_config,
)
_playground_locomotion._randomizer[ENV_NAME] = go2_randomize.domain_randomize
