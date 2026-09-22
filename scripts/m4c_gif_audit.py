"""Audit: did the recorded terrain-comparison GIFs (render_terrain_videos.py)
actually show a SUCCESSFUL episode, or just whatever episode 0 happened to be?

render_terrain_videos.py records `ep == 0` unconditionally (no success check),
unlike eval_mlp_go2.py's own `eval_frozen` pattern which only keeps a GIF
`if reached`. This reproduces its exact procedure (same seeds, same warm-up)
and reports the true outcome for the specific episode each GIF shows.
"""
import sys, os
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from nmc.controllers.privileged_expert import PrivilegedExpert, PrivilegedConfig
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from eval_mlp_go2 import run_episode
from pilot_m4 import make_rstdp, make_frozen_snn, make_frozen_mlp, run_block

seed0 = 7000
warmup = 15
mu_by_mode = {"ice": 0.08, "sand": 1.6}
# the actual seed/mu render_terrain_videos.py used for the GIFs on the site
overrides = {"ice": 0.28}  # snn_rstdp_ice_mu028_warm15.gif override

expert = PrivilegedExpert(PrivilegedConfig(inflate_m=0.9, fallback_inflations_m=(0.9, 0.7, 0.55)))


def make_env(terrain_mode, mu, episode_len_s=45.0):
    return Go2NavEnv(Go2NavConfig(shift_type="terrain", terrain_mode=terrain_mode,
                                  shift_time_s=0.5, episode_len_s=episode_len_s,
                                  terrain_friction={terrain_mode: mu}))


for terrain in ("ice", "sand"):
    mu = mu_by_mode[terrain]
    print(f"\n=== terrain={terrain}, mu={mu} (default script mu) ===")

    env = make_env(terrain, mu)
    mlp = make_frozen_mlp()
    info, step, frames, pl, sp = run_episode(env, mlp, seed=seed0, expert=expert, record_frames=True)
    print(f"  frozen MLP  ep0 (seed={seed0}): reached={info['reached']} fell={info['fell']} "
          f"collision={info['collision']}({info.get('collision_kind')}) steps={step} frames={len(frames)}")
    env.close()

    env = make_env(terrain, mu)
    snn = make_frozen_snn(seed=7)
    info, step, frames, pl, sp = run_episode(env, snn, seed=seed0, expert=expert, record_frames=True)
    print(f"  frozen SNN  ep0 (seed={seed0}): reached={info['reached']} fell={info['fell']} "
          f"collision={info['collision']}({info.get('collision_kind')}) steps={step} frames={len(frames)}")
    env.close()

    env = make_env(terrain, mu)
    rstdp = make_rstdp(eta=0.05, seed=7, reward_mode="td", plastic_layers=[0, -1], anchor=0.005)
    warm_seeds = list(range(seed0 + 500, seed0 + 500 + warmup))
    run_block(env, rstdp, warm_seeds, adapt=True)
    info, step, frames, pl, sp = run_episode(env, rstdp, seed=seed0 + 1000, expert=expert, record_frames=True)
    print(f"  R-STDP SNN  ep0 (seed={seed0+1000}, mu={mu}): reached={info['reached']} fell={info['fell']} "
          f"collision={info['collision']}({info.get('collision_kind')}) steps={step} frames={len(frames)}")
    env.close()

# the ACTUAL featured ice GIF used a different mu (0.28) + warmup (15) per its filename
mu = overrides["ice"]
print(f"\n=== terrain=ice, mu={mu} (the actual featured snn_rstdp_ice_mu028_warm15.gif config) ===")
env = make_env("ice", mu)
rstdp = make_rstdp(eta=0.05, seed=7, reward_mode="td", plastic_layers=[0, -1], anchor=0.005)
warm_seeds = list(range(seed0 + 500, seed0 + 500 + 15))
run_block(env, rstdp, warm_seeds, adapt=True)
info, step, frames, pl, sp = run_episode(env, rstdp, seed=seed0 + 1000, expert=expert, record_frames=True)
print(f"  R-STDP SNN  ep0 (seed={seed0+1000}, mu={mu}): reached={info['reached']} fell={info['fell']} "
      f"collision={info['collision']}({info.get('collision_kind')}) steps={step} frames={len(frames)}")
env.close()
