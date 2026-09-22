"""Diagnose the M4b-3 trajectory figure: why do frozen SNN and R-STDP all crash
into the same obstacle near the start instead of reaching the goal, and is the
crash actually inside the dead-beam blind sector (the intended shift working
as designed) or something else (a bug / a policy weakness unrelated to the
shift)?"""
import sys, os
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from m6_energy import build

def shift_cfg(dropout=0.20, start=8):
    return Go2NavConfig(shift_type="sensor", shift_time_s=0.5,
                        sensor_dropout_frac=dropout, sensor_dropout_start=start,
                        episode_len_s=60.0)

def dead_beam_offsets(env):
    dead = np.where(env._dead_beams)[0]
    n = env.cfg.n_lidar_beams
    return dead, np.degrees(-np.pi + dead * (2 * np.pi / n))


def run_and_diagnose(name, seed, ep_seed, label):
    env = Go2NavEnv(shift_cfg())
    ctrl = build(name, seed)
    obs, _ = env.reset(seed=ep_seed)
    print(f"\n=== {label} (seed={seed}, ep_seed={ep_seed}) ===")
    dead = None
    offsets_deg = None
    step = 0
    while True:
        a = ctrl.act(obs)
        pos, yaw = env._robot_pose()
        nobs, r, term, trunc, info = env.step(a)
        ctrl.observe(r, nobs, term or trunc)
        obs = nobs
        step += 1
        if dead is None and env._dead_beams is not None:
            dead, offsets_deg = dead_beam_offsets(env)
            print(f"dead beams: {dead.tolist()}  -> bearing offsets {offsets_deg.round(1).tolist()} deg "
                  f"(0=fwd, +=CCW/left, -=CW/right)")
        if info["collision"]:
            obst = env.privileged_state()[3]
            robot_xy = pos[:2]
            dists = [np.linalg.norm(robot_xy - o[:2]) for o in obst]
            j = int(np.argmin(dists))
            bearing_world = np.degrees(np.arctan2(obst[j][1] - robot_xy[1], obst[j][0] - robot_xy[0]))
            rel_bearing = ((bearing_world - np.degrees(yaw) + 180) % 360) - 180
            in_blind = any(abs(((rel_bearing - o + 180) % 360) - 180) < 6 for o in offsets_deg)
            print(f"COLLISION at step {step}, t={info['t']:.1f}s, kind={info['collision_kind']}")
            print(f"  robot pos {robot_xy.round(2)}, yaw {np.degrees(yaw):.1f} deg, "
                  f"nearest obstacle at {np.array(obst[j][:2]).round(2)}, dist {dists[j]:.2f}")
            print(f"  obstacle bearing relative to heading: {rel_bearing:.1f} deg "
                  f"-> {'INSIDE dead-beam sector' if in_blind else 'OUTSIDE dead-beam sector (visible!)'}")
            break
        if term or trunc:
            print(f"episode ended without collision: reached={info['reached']} fell={info['fell']} "
                  f"trunc={trunc} at step {step}")
            break
    env.close()

for name in ["Frozen SNN"]:
    run_and_diagnose(name, 6000, 600000, f"{name}, the exact M4b-3 frozen episode")

rstdp = build("R-STDP SNN", 6000)
env = Go2NavEnv(shift_cfg())
obs, _ = env.reset(seed=600000)
while True:
    a = rstdp.act(obs)
    nobs, r, term, trunc, info = env.step(a)
    rstdp.observe(r, nobs, term or trunc)
    obs = nobs
    if term or trunc:
        break
env.close()
env2 = Go2NavEnv(shift_cfg())
for ep in range(12):
    obs, _ = env2.reset(seed=600100 + ep)
    while True:
        a = rstdp.act(obs)
        nobs, r, term, trunc, info = env2.step(a)
        rstdp.observe(r, nobs, term or trunc)
        obs = nobs
        if term or trunc:
            break
env2.close()

env3 = Go2NavEnv(shift_cfg())
obs, _ = env3.reset(seed=600000)
print(f"\n=== R-STDP after 12 adaptation eps (seed=6000, ep_seed=600000) ===")
dead = None
offsets_deg = None
step = 0
while True:
    a = rstdp.act(obs)
    pos, yaw = env3._robot_pose()
    nobs, r, term, trunc, info = env3.step(a)
    rstdp.observe(r, nobs, term or trunc)
    obs = nobs
    step += 1
    if dead is None and env3._dead_beams is not None:
        dead, offsets_deg = dead_beam_offsets(env3)
        print(f"dead beams offsets {offsets_deg.round(1).tolist()} deg")
    if info["collision"]:
        obst = env3.privileged_state()[3]
        robot_xy = pos[:2]
        dists = [np.linalg.norm(robot_xy - o[:2]) for o in obst]
        j = int(np.argmin(dists))
        bearing_world = np.degrees(np.arctan2(obst[j][1] - robot_xy[1], obst[j][0] - robot_xy[0]))
        rel_bearing = ((bearing_world - np.degrees(yaw) + 180) % 360) - 180
        in_blind = any(abs(((rel_bearing - o + 180) % 360) - 180) < 6 for o in offsets_deg)
        print(f"COLLISION at step {step}, t={info['t']:.1f}s, kind={info['collision_kind']}")
        print(f"  robot pos {robot_xy.round(2)}, yaw {np.degrees(yaw):.1f} deg, "
              f"nearest obstacle at {np.array(obst[j][:2]).round(2)}, dist {dists[j]:.2f}")
        print(f"  obstacle bearing relative to heading: {rel_bearing:.1f} deg "
              f"-> {'INSIDE dead-beam sector' if in_blind else 'OUTSIDE dead-beam sector (visible!)'}")
        break
    if term or trunc:
        print(f"episode ended: reached={info['reached']} fell={info['fell']} step={step}")
        break
env3.close()
