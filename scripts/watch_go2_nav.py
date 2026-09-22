"""Live, interactive viewer for the Go2 nav env -- opens a real MuJoCo GUI window
(not a pre-rendered video) and steps a chosen controller through the sensor-
dropout shift in real time, so you can rotate/zoom/pause and inspect details
frame-by-frame yourself.

Controllers: frozen_mlp | online_mlp | expert | frozen_snn | pure_stdp | rstdp | tmnorm
(rstdp's eta/anchor/scope/reward-mode are overridable -- defaults match the M4
recipe, i.e. the one config we've now tested six ways against frozen_snn).

Examples:
  conda run -n nmc python scripts/watch_go2_nav.py --controller frozen_snn
  conda run -n nmc python scripts/watch_go2_nav.py --controller rstdp --seed 6003
  conda run -n nmc python scripts/watch_go2_nav.py --controller rstdp --rstdp-scope all --rstdp-reward-mode rpe
  conda run -n nmc python scripts/watch_go2_nav.py --controller expert --no-shift
  conda run -n nmc python scripts/watch_go2_nav.py --controller tmnorm --episodes 3 --speed 3
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mujoco
import mujoco.viewer

from nmc.envs.go2_nav_env import Go2NavEnv, Go2NavConfig
from nmc.controllers.snn import SNNNavController
from nmc.controllers.privileged_expert import PrivilegedExpert, PrivilegedConfig
from nmc.plasticity.stdp import STDPConfig
from m5_full_comparison import (make_frozen_mlp, make_online_mlp, make_frozen_snn,
                                 make_pure_stdp, load_snn_net, readout_bounds)


def make_rstdp_custom(seed, eta, anchor, scope, reward_mode, gate):
    net, T = load_snn_net()
    w_min, w_max = readout_bounds(net)
    cfg = STDPConfig(reward_modulated=True, eta=eta, tau_e_ms=200.0, w_min=w_min, w_max=w_max)
    plastic_layers = {"input+readout": [0, -1], "all": [0, 1, 2], "readout": [-1]}[scope]
    return SNNNavController(net, n_steps=T, plasticity_enabled=True, stdp_cfg=cfg,
                            seed=seed, reward_mode=reward_mode, gate_threshold=gate,
                            plastic_layers=plastic_layers, anchor=anchor)


def build_controller(args):
    name = args.controller
    if name == "frozen_mlp": return make_frozen_mlp(), False
    if name == "online_mlp": return make_online_mlp(), True
    if name == "frozen_snn": return make_frozen_snn(args.seed), False
    if name == "pure_stdp": return make_pure_stdp(args.seed), True
    if name == "tmnorm":
        from nmc.controllers.tmnorm import TMNormController
        net, T = load_snn_net()
        return TMNormController(net, n_steps=T, seed=args.seed, ema_alpha=0.01), True
    if name == "rstdp":
        return make_rstdp_custom(args.seed, args.rstdp_eta, args.rstdp_anchor,
                                 args.rstdp_scope, args.rstdp_reward_mode,
                                 args.rstdp_gate), True
    if name == "expert":
        return PrivilegedExpert(PrivilegedConfig(inflate_m=0.9,
                                fallback_inflations_m=(0.9, 0.7, 0.55))), None
    raise ValueError(f"unknown controller {name!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", required=True,
                    choices=["frozen_mlp", "online_mlp", "expert", "frozen_snn",
                             "pure_stdp", "rstdp", "tmnorm"])
    ap.add_argument("--seed", type=int, default=6000)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.20,
                    help="sensor-dropout fraction; 0.20 is the corrected/non-floored "
                         "severity (0.30, M4's pick, turns out floored -- see "
                         "archive/M5_full_comparison/severity_screen.md)")
    ap.add_argument("--no-shift", action="store_true", help="run on the base (unshifted) distribution")
    ap.add_argument("--speed", type=float, default=1.0, help="real-time multiplier (2 = 2x speed)")
    ap.add_argument("--rstdp-eta", type=float, default=0.05)
    ap.add_argument("--rstdp-anchor", type=float, default=0.005)
    ap.add_argument("--rstdp-scope", choices=["input+readout", "all", "readout"], default="input+readout")
    ap.add_argument("--rstdp-reward-mode", choices=["td", "rpe", "raw"], default="td")
    ap.add_argument("--rstdp-gate", type=float, default=0.0)
    args = ap.parse_args()

    if args.no_shift:
        cfg = Go2NavConfig(shift_time_s=1e9, episode_len_s=60.0)
    else:
        cfg = Go2NavConfig(shift_type="sensor", shift_time_s=0.5,
                           sensor_dropout_frac=args.dropout, sensor_dropout_start=8,
                           episode_len_s=60.0)
    env = Go2NavEnv(cfg)
    ctrl, adapts = build_controller(args)
    is_expert = args.controller == "expert"

    print(f"controller={args.controller}  shift={'OFF' if args.no_shift else f'sensor dropout={args.dropout}'}  "
          f"seed={args.seed}  speed={args.speed}x")
    print("MuJoCo viewer window opening -- drag to rotate, scroll to zoom, space to pause.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(args.episodes):
            if not viewer.is_running():
                break
            obs, _ = env.reset(seed=args.seed * 100 + ep)
            step = 0
            t0 = time.perf_counter()
            while viewer.is_running():
                a = ctrl.act(obs, env) if is_expert else ctrl.act(obs)
                nobs, r, term, trunc, info = env.step(a)
                if not is_expert and adapts:
                    ctrl.observe(r, nobs, term or trunc)
                obs = nobs
                viewer.sync()
                step += 1
                if step % 25 == 0:   # ~once per sim-second at control_hz=25
                    print(f"  ep {ep} t={info['t']:5.1f}s  action={a}  "
                          f"lidar_min={min(obs[:32]):.2f}  goal_dist={env._goal_distance():.2f}  "
                          f"phase={info.get('phase', '?')}", flush=True)
                target_t = t0 + step * (1.0 / cfg.control_hz) / args.speed
                sleep_s = target_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                if term or trunc:
                    verdict = ("REACHED" if info["reached"] else "FELL" if info["fell"]
                              else f"COLLIDED({info['collision_kind']})" if info["collision"]
                              else "TIMEOUT")
                    print(f"episode {ep}: {verdict}  (t={info['t']:.1f}s, {step} steps)\n")
                    break
    env.close()


if __name__ == "__main__":
    main()
