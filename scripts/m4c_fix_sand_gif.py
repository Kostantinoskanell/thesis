"""Fix: render_terrain_videos.py recorded episode 0 unconditionally with no
success check, and it happened to be a failure (collision with a dynamic
obstacle) for R-STDP on sand -- yet the site captioned it as a recovery demo.
Search subsequent episode seeds (same warm-up) for one R-STDP genuinely wins,
verify reached==True with no fall/collision, and save that as the real GIF.
"""
import sys, os
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
from PIL import Image

from nmc.controllers.privileged_expert import PrivilegedExpert, PrivilegedConfig
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from eval_mlp_go2 import run_episode
from pilot_m4 import make_rstdp, run_block

OUT = Path("archive/M4b_terrain_walk_compare")
seed0 = 7000
mu = 1.6

expert = PrivilegedExpert(PrivilegedConfig(inflate_m=0.9, fallback_inflations_m=(0.9, 0.7, 0.55)))
env = Go2NavEnv(Go2NavConfig(shift_type="terrain", terrain_mode="sand",
                             terrain_friction={"sand": mu}, shift_time_s=0.5, episode_len_s=45.0))
rstdp = make_rstdp(eta=0.05, seed=7, reward_mode="td", plastic_layers=[0, -1], anchor=0.005)
warm_seeds = list(range(seed0 + 500, seed0 + 500 + 15))
run_block(env, rstdp, warm_seeds, adapt=True)

found = False
for ep in range(30):
    seed = seed0 + 1000 + ep
    info, step, frames, pl, sp = run_episode(env, rstdp, seed=seed, expert=expert, record_frames=True)
    print(f"ep seed={seed}: reached={info['reached']} fell={info['fell']} "
          f"collision={info['collision']}({info.get('collision_kind')}) steps={step}")
    if info["reached"] and not info["fell"] and not info["collision"]:
        path = OUT / "snn_rstdp_sand.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(f"VERIFIED SUCCESS -- wrote {path} ({len(frames)} frames, seed={seed})")
        found = True
        break
env.close()
if not found:
    print("no clean success found in 30 episodes -- needs a different approach")
