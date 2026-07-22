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
    "        print(f\"[SPIKING] anti-crouch: terminate<{_minh}m + base_height_l2(target={_tgth}, w={_hw})\")\n"
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
    "        import pprint; print(\"[SPIKING] actor cfg ->\"); pprint.pprint(_sp[\"actor\"])\n"
    "        print(\"[SPIKING] algorithm schedule ->\", _sp[\"algorithm\"].get(\"schedule\"),\n"
    "              \"lr ->\", _sp[\"algorithm\"].get(\"learning_rate\"))\n"
    "        runner = OnPolicyRunner(env, _sp, log_dir=log_dir, device=agent_cfg.device)"
)
assert orig in s, "anchor line not found"
s = s.replace(orig, patch, 1)

open(dst_path, "w").write(s)
print("WROTE", dst_path)
print("has spiking patch:", "SpikingActorMLPModel" in open(dst_path).read())
