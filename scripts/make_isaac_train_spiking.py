# Transformer: copy Isaac Lab's rsl_rl train.py -> train_spiking.py with two edits:
#  (1) put the repo src/ on sys.path so resolve_callable can import our actor
#  (2) patch cfg_dict["actor"]["class_name"] to the spiking actor before the runner
src_path = "/home/hapos/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py"
dst_path = "/home/hapos/IsaacLab/scripts/reinforcement_learning/rsl_rl/train_spiking.py"
s = open(src_path).read()

# (1) add repo src to path right after the first 'import sys'
inject = 'import sys\nsys.path.insert(0, "/mnt/c/Users/hapos/Desktop/thesis/src")\n'
assert "import sys\n" in s
s = s.replace("import sys\n", inject, 1)

# (1b) anti-crouch reward shaping (D14): inject a base-height termination + base-height
# reward into env_cfg BEFORE gym.make, to kill the belly-flop local optimum the spiking
# policy falls into. Anchored after the num_envs override line (early in main()).
env_anchor = ("    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None "
              "else env_cfg.scene.num_envs")
env_patch = env_anchor + (
    "\n    import os as _oc\n"
    "    if _oc.environ.get(\"SPIKING_ANTICROUCH\") == \"1\":\n"
    "        from isaaclab.envs import mdp as _mdp\n"
    "        from isaaclab.managers import RewardTermCfg as _RewTerm\n"
    "        from isaaclab.managers import TerminationTermCfg as _DoneTerm\n"
    "        _minh = float(_oc.environ.get(\"SPIKING_MIN_HEIGHT\", \"0.20\"))\n"
    "        _tgth = float(_oc.environ.get(\"SPIKING_TARGET_HEIGHT\", \"0.32\"))\n"
    "        _hw = float(_oc.environ.get(\"SPIKING_HEIGHT_WEIGHT\", \"-5.0\"))\n"
    "        env_cfg.terminations.low_base = _DoneTerm(func=_mdp.root_height_below_minimum,\n"
    "                                                  params={\"minimum_height\": _minh})\n"
    "        env_cfg.rewards.base_height = _RewTerm(func=_mdp.base_height_l2, weight=_hw,\n"
    "                                               params={\"target_height\": _tgth})\n"
    "        _vw = _oc.environ.get(\"SPIKING_VEL_WEIGHT\")\n"
    "        if _vw:\n"
    "            env_cfg.rewards.track_lin_vel_xy_exp.weight = float(_vw)\n"
    "            env_cfg.rewards.track_ang_vel_z_exp.weight = float(_vw) * 0.5\n"
    "        print(f\"[SPIKING] anti-crouch: terminate<{_minh}m + base_height_l2(target={_tgth}, w={_hw}), vel_w={_vw}\")\n"
)
assert env_anchor in s, "env_cfg anchor line not found"
s = s.replace(env_anchor, env_patch, 1)

# (2) replace the OnPolicyRunner construction line with a patched one
orig = "        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)"
patch = (
    "        _sp = agent_cfg.to_dict()\n"
    "        import os as _os\n"
    "        _in_pop = int(_os.environ.get(\"SPIKING_IN_POP\", \"10\"))\n"
    "        _out_pop = int(_os.environ.get(\"SPIKING_OUT_POP\", \"10\"))\n"
    "        _T = int(_os.environ.get(\"SPIKING_T\", \"8\"))\n"
    "        _hidden_env = _os.environ.get(\"SPIKING_HIDDEN\")\n"
    "        _hidden = [int(x) for x in _hidden_env.split(\",\")] if _hidden_env else [128, 128, 128]\n"
    "        _sp[\"actor\"][\"class_name\"] = \"nmc.locomotion.rsl_rl_spiking:SpikingActorMLPModel\"\n"
    "        _sp[\"actor\"][\"hidden_dims\"] = _hidden\n"
    "        _sp[\"actor\"].update({\"in_pop\": _in_pop, \"out_pop\": _out_pop, \"spiking_T\": _T, \"obs_normalization\": True,\n"
    "                              \"enc_sigma\": 0.3872983, \"actor_lr_scale\": 1.0, \"decoder_tanh\": False})\n"
    "        if _os.environ.get(\"SPIKING_FIXED_LR\") == \"1\":\n"
    "            _sp[\"algorithm\"][\"schedule\"] = \"fixed\"\n"
    "            _sp[\"algorithm\"][\"learning_rate\"] = float(_os.environ.get(\"SPIKING_LR\", \"1.0e-3\"))\n"
    "        _ent = _os.environ.get(\"SPIKING_ENTROPY\")\n"
    "        if _ent:\n"
    "            _sp[\"algorithm\"][\"entropy_coef\"] = float(_ent)\n"
    "        import pprint; print(\"[SPIKING] actor cfg ->\"); pprint.pprint(_sp[\"actor\"])\n"
    "        print(\"[SPIKING] algorithm schedule ->\", _sp[\"algorithm\"].get(\"schedule\"),\n"
    "              \"lr ->\", _sp[\"algorithm\"].get(\"learning_rate\"))\n"
    "        runner = OnPolicyRunner(env, _sp, log_dir=log_dir, device=agent_cfg.device)\n"
    "        _iw = _os.environ.get(\"SPIKING_INIT_WEIGHTS\")\n"
    "        if _iw:\n"
    "            import torch as _t\n"
    "            _blob = _t.load(_iw, map_location=agent_cfg.device, weights_only=True)\n"
    "            _m = _blob.pop(\"_obs_mean\"); _sd = _blob.pop(\"_obs_std\")\n"
    "            _actor = runner.alg.get_policy()\n"
    "            _actor.mlp.load_state_dict(_blob)   # BC-distilled PopSpikingActorNet weights\n"
    "            _nz = getattr(_actor, \"obs_normalizer\", None)\n"
    "            if _nz is not None and hasattr(_nz, \"_mean\"):\n"
    "                _nz._mean.copy_(_m.to(agent_cfg.device)); _nz._std.copy_(_sd.to(agent_cfg.device))\n"
    "                if hasattr(_nz, \"_var\"): _nz._var.copy_((_sd**2).to(agent_cfg.device))\n"
    "                if hasattr(_nz, \"count\"): _nz.count.fill_(10_000_000)  # freeze at distill stats (avoid early drift off the BC init)\n"
    "            print(f\"[SPIKING] loaded BC-distilled init weights from {_iw} (normalizer frozen)\")\n"
)
assert orig in s, "anchor line not found"
s = s.replace(orig, patch, 1)

open(dst_path, "w").write(s)
print("WROTE", dst_path)
print("has spiking patch:", "SpikingActorMLPModel" in open(dst_path).read())
