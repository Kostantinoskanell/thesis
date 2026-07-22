"""L4 -- R-STDP gait recovery under a terrain (icy floor) shift, in Isaac Lab.

The M4c motivation: R-STDP on the NAVIGATION layer could recalibrate velocity
commands (fixed sand) but had no lever over slipping on ice, because it never
touches the legs. This script releases the LOCOMOTION policy itself (the L3
PopSAN spiking actor) to R-STDP and tests recovery on an icy floor -- the
locomotion-layer analogue of the nav-layer H1 recovery claim.

Runs entirely inside Isaac Lab/PhysX (not the Windows MuJoCo loop) -- L1-L3
never left Isaac Lab either, so this avoids any cross-simulator joint-order or
asset mismatch risk. Uses the *_Play-v0* task variant (clean: no obs
corruption noise, no random push events) for a controlled plasticity signal.

Four modes, mirroring the nav-layer M4 pilot structure:
  baseline      : frozen policy, DEFAULT terrain friction (reference reward)
  frozen_shift  : frozen policy, ICY terrain friction (does the shift hurt?)
  rstdp_shift   : R-STDP-adapting policy, ICY friction, continual across the
                  episode block (does it recover?)
  retention     : the rstdp-adapted weights, back on DEFAULT friction (did it
                  forget the base task? -- the stability half of D9)

Terrain friction: default static/dynamic = 1.0/1.0, robot foot friction fixed
0.8/0.6, combine_mode="multiply" -> effective friction = terrain x foot. Icy
sets terrain friction low (e.g. 0.05) so multiply naturally dominates it down,
no combine-mode surgery needed (unlike MuJoCo's geom_priority trick).

Run (inside the isaac conda env, via wsl_isaac_l4.sh):
  python scripts/l4_rstdp_terrain.py --mode baseline --episodes 20
  python scripts/l4_rstdp_terrain.py --mode frozen_shift --episodes 20 --friction 0.05
  python scripts/l4_rstdp_terrain.py --mode rstdp_shift --episodes 30 --friction 0.05 \
      --save-weights /home/hapos/l4_adapted.pt
  python scripts/l4_rstdp_terrain.py --mode retention --episodes 20 \
      --load-weights /home/hapos/l4_adapted.pt
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/mnt/c/Users/hapos/Desktop/thesis/src")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["baseline", "frozen_shift", "rstdp_shift", "retention"], required=True)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--friction", type=float, default=1.0, help="terrain static+dynamic friction (default 1.0)")
parser.add_argument("--eta", type=float, default=0.05, help="R-STDP reward learning rate (D13 tuning)")
parser.add_argument("--anchor", type=float, default=0.005, help="R-STDP elastic weight anchor strength")
parser.add_argument("--plastic-layers", default="0,-1", help="comma-separated fc layer indices to make plastic")
parser.add_argument("--ckpt", default="/home/hapos/IsaacLab/logs/rsl_rl/unitree_go2_flat/2026-07-22_03-21-45/model_1499.pt")
parser.add_argument("--load-weights", default=None, help="load a previously-adapted mlp state_dict instead of --ckpt's")
parser.add_argument("--save-weights", default=None, help="save the (possibly adapted) mlp state_dict here")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", action="store_true", help="record an rgb_array video of the rollout")
parser.add_argument("--video-length", type=int, default=1000, help="video length in steps")
parser.add_argument("--video-dir", default="/home/hapos/l4_videos", help="output dir for videos")
parser.add_argument("--dump-traj", default=None, help="save first-episode base pose + joint trajectory to this .npz (headless, no RTX render needed)")
parser.add_argument("--force-command", default=None, help="override the velocity command the policy sees each step, 'vx,vy,yaw' (clean walk test)")
parser.add_argument("--mlp-ckpt", default=None, help="eval the standard MLP baseline actor from this checkpoint instead of the spiking net (raw obs, no normalize, no tanh)")
parser.add_argument("--T", type=int, default=8, help="spiking net timesteps (must match the trained checkpoint; L5 energy sweep uses T=4)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
if args_cli.video:
    args_cli.enable_cameras = True  # must be set BEFORE AppLauncher

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np

import isaaclab_tasks  # noqa: F401  (registers Isaac-Velocity-Flat-Unitree-Go2-Play-v0)
from isaaclab_tasks.utils import parse_env_cfg

from nmc.locomotion.popsan_actor import PopSpikingActorNet
from nmc.locomotion.popsan_rstdp import PopSpikingRSTDPController
from nmc.plasticity.stdp import STDPConfig

TASK = "Isaac-Velocity-Flat-Unitree-Go2-Play-v0"
NET_KW = dict(obs_dim=48, act_dim=12, hidden=(128, 128, 128), in_pop=10, out_pop=10,
              T=8, decoder_tanh=False)


def load_actor(ckpt_path, weights_override=None, device="cuda:0"):
    kw = dict(NET_KW); kw["T"] = args_cli.T
    net = PopSpikingActorNet(**kw).to(device)
    if weights_override:
        # a previously-adapted checkpoint saved by --save-weights: mlp weights +
        # normalizer stats bundled together in one plain state-dict blob.
        blob = torch.load(weights_override, map_location=device, weights_only=True)
        mean, std = blob.pop("_obs_mean").to(device), blob.pop("_obs_std").to(device)
        net.load_state_dict(blob)
        return net, mean, std
    # fresh from the rsl_rl training checkpoint: strip the "mlp." prefix rsl_rl's
    # MLPModel wraps our net's own state_dict in.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_sd = ckpt["actor_state_dict"]
    mean = actor_sd["obs_normalizer._mean"].to(device)
    std = actor_sd["obs_normalizer._std"].to(device)
    mlp_sd = {k[len("mlp."):]: v for k, v in actor_sd.items() if k.startswith("mlp.")}
    net.load_state_dict(mlp_sd)
    return net, mean, std


def make_env(friction: float, device="cuda:0"):
    env_cfg = parse_env_cfg(TASK, device=device, num_envs=1)
    env_cfg.scene.terrain.physics_material.static_friction = friction
    env_cfg.scene.terrain.physics_material.dynamic_friction = friction
    import gymnasium as gym
    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        import os
        os.makedirs(args_cli.video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env, video_folder=args_cli.video_dir,
            step_trigger=lambda step: step == 0,          # record from the very first step
            video_length=args_cli.video_length,
            name_prefix=f"l4_{args_cli.mode}_fric{friction}",
            disable_logger=True)
        print(f"[L4] recording video -> {args_cli.video_dir}", flush=True)
    return env


def normalize(obs, mean, std, eps=1e-2):
    return (obs - mean) / (std + eps)


def load_mlp(ckpt_path, device="cuda:0"):
    """Rebuild the standard rsl_rl MLP baseline actor: 48->128->128->128->12 ELU,
    action = mlp(raw_obs) (GaussianDistribution deterministic output is the mean;
    no tanh; this baseline trained with obs_normalization=False -> raw obs)."""
    import torch.nn as nn
    mlp = nn.Sequential(
        nn.Linear(48, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(),
        nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 12)).to(device)
    asd = torch.load(ckpt_path, map_location=device, weights_only=False)["actor_state_dict"]
    sd = {k[len("mlp."):]: v for k, v in asd.items() if k.startswith("mlp.")}
    mlp.load_state_dict(sd)
    mlp.eval()
    return mlp


def main():
    torch.manual_seed(args_cli.seed)
    device = "cuda:0"
    mlp_mode = args_cli.mlp_ckpt is not None
    if mlp_mode:
        net = load_mlp(args_cli.mlp_ckpt, device)
        mean = std = None
    else:
        net, mean, std = load_actor(args_cli.ckpt, args_cli.load_weights, device)
        net.eval()

    adapt = args_cli.mode == "rstdp_shift" and not mlp_mode
    friction = args_cli.friction if args_cli.mode in ("frozen_shift", "rstdp_shift") else 1.0
    env = make_env(friction, device)

    ctrl = None
    if adapt:
        n_layers = len(net.fc)
        plastic = [int(x) % n_layers for x in args_cli.plastic_layers.split(",")]
        cfg = STDPConfig(reward_modulated=True, eta=args_cli.eta, tau_e_ms=200.0)
        ctrl = PopSpikingRSTDPController(net, plastic_layers=plastic,
                                         stdp_cfg=cfg, anchor=args_cli.anchor, reward_mode="td", device=device)
        print(f"[L4] R-STDP: eta={args_cli.eta} anchor={args_cli.anchor} plastic_layers={plastic}", flush=True)

    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.data.joint_names)
    traj = {"base_pos": [], "base_quat": [], "base_lin_vel": [], "joint_pos": [], "command": []}

    obs_dict, _ = env.reset()
    ep_reward, ep_len = 0.0, 0
    ep_rewards, ep_lens = [], []

    force_cmd = None
    if args_cli.force_command:
        force_cmd = torch.tensor([float(x) for x in args_cli.force_command.split(",")],
                                 device=device, dtype=torch.float32)
        print(f"[L4] forcing velocity command (obs slots 9:12) = {force_cmd.tolist()}", flush=True)

    while len(ep_rewards) < args_cli.episodes:
        raw = obs_dict["policy"]
        if force_cmd is not None:
            raw = raw.clone()
            raw[0, 9:12] = force_cmd     # velocity_commands slots the policy reads
        if mlp_mode:
            with torch.no_grad():
                action = net(raw)         # MLP: raw obs in, mean out (no normalize/tanh)
        else:
            norm = normalize(raw, mean, std)
            if adapt:
                norm_np = norm.squeeze(0).detach().cpu().numpy()
                action_np = ctrl.act(norm_np)
                action = torch.as_tensor(action_np, device=device, dtype=torch.float32).unsqueeze(0)
            else:
                with torch.no_grad():
                    action = net(norm)

        obs_dict, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated[0] or truncated[0])
        r = float(reward[0])
        ep_reward += r
        ep_len += 1

        if args_cli.dump_traj and len(ep_rewards) == 0:  # record the FIRST episode only
            traj["base_pos"].append(robot.data.root_pos_w[0].cpu().numpy().copy())
            traj["base_quat"].append(robot.data.root_quat_w[0].cpu().numpy().copy())
            traj["base_lin_vel"].append(robot.data.root_lin_vel_b[0].cpu().numpy().copy())
            traj["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
            traj["command"].append(raw[0, 9:12].cpu().numpy().copy())  # velocity_commands slots

        if adapt:
            next_norm_np = normalize(obs_dict["policy"], mean, std).squeeze(0).detach().cpu().numpy()
            ctrl.learn(r, next_obs=next_norm_np, done=done)

        if done:
            ep_rewards.append(ep_reward)
            ep_lens.append(ep_len)
            print(f"[{args_cli.mode}] episode {len(ep_rewards)}/{args_cli.episodes}: "
                  f"return={ep_reward:.2f} len={ep_len}", flush=True)
            ep_reward, ep_len = 0.0, 0
            if args_cli.dump_traj and len(ep_rewards) == 1:
                import numpy as _np
                _np.savez(args_cli.dump_traj,
                          base_pos=_np.array(traj["base_pos"]),
                          base_quat=_np.array(traj["base_quat"]),
                          base_lin_vel=_np.array(traj["base_lin_vel"]),
                          joint_pos=_np.array(traj["joint_pos"]),
                          command=_np.array(traj["command"]),
                          joint_names=_np.array(joint_names))
                print(f"[L4] dumped {len(traj['joint_pos'])}-step trajectory -> {args_cli.dump_traj}", flush=True)

    env.close()

    rewards = np.array(ep_rewards)
    n = len(rewards)
    half = n // 2
    print(f"\n=== L4 [{args_cli.mode}] friction={friction} n={n} ===")
    print(f"mean return: {rewards.mean():.3f}  std: {rewards.std():.3f}")
    if adapt and n >= 4:
        print(f"1st-half mean: {rewards[:half].mean():.3f}  2nd-half mean: {rewards[half:].mean():.3f}")

    if args_cli.save_weights:
        blob = dict(net.state_dict())
        blob["_obs_mean"] = mean.cpu()
        blob["_obs_std"] = std.cpu()
        torch.save(blob, args_cli.save_weights)
        print(f"saved adapted weights -> {args_cli.save_weights}")

    simulation_app.close()


if __name__ == "__main__":
    main()
