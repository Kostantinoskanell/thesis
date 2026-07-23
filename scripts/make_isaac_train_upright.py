# Transformer: copy Isaac Lab's rsl_rl train.py -> train_upright.py with TWO edits:
# (1) inject an anti-crouch base-height reward + termination into env_cfg (same
# machinery as make_isaac_train_spiking.py's step 1b), but keep the STANDARD MLP actor.
# This retrains the MLP *teacher* to walk upright (~0.30 m) instead of the crouched 0.18 m
# the default flat-velocity reward converges to -- so the distilled spiking student
# (which can only imitate the teacher) can also stand tall. Gated on UPRIGHT_ANTICROUCH=1.
# (2) inject a feet_slide penalty (Isaac Lab's mdp.feet_slide: penalizes a foot's linear
# velocity while it's in ground contact -- the direct "dragging leg" signature the user
# spotted in the rendered gait video). Gated on UPRIGHT_FEET_SLIDE=1.
src_path = "/home/hapos/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py"
dst_path = "/home/hapos/IsaacLab/scripts/reinforcement_learning/rsl_rl/train_upright.py"
s = open(src_path).read()

env_anchor = ("    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None "
              "else env_cfg.scene.num_envs")
env_patch = env_anchor + (
    "\n    import os as _oc\n"
    "    if _oc.environ.get(\"UPRIGHT_ANTICROUCH\") == \"1\":\n"
    "        from isaaclab.envs import mdp as _mdp\n"
    "        from isaaclab.managers import RewardTermCfg as _RewTerm\n"
    "        from isaaclab.managers import TerminationTermCfg as _DoneTerm\n"
    "        _minh = float(_oc.environ.get(\"UPRIGHT_MIN_HEIGHT\", \"0.18\"))\n"
    "        _tgth = float(_oc.environ.get(\"UPRIGHT_TARGET_HEIGHT\", \"0.30\"))\n"
    "        _hw = float(_oc.environ.get(\"UPRIGHT_HEIGHT_WEIGHT\", \"-10.0\"))\n"
    "        env_cfg.terminations.low_base = _DoneTerm(func=_mdp.root_height_below_minimum,\n"
    "                                                  params={\"minimum_height\": _minh})\n"
    "        env_cfg.rewards.base_height = _RewTerm(func=_mdp.base_height_l2, weight=_hw,\n"
    "                                               params={\"target_height\": _tgth})\n"
    "        print(f\"[UPRIGHT] anti-crouch: terminate<{_minh}m + base_height_l2(target={_tgth}, w={_hw})\")\n"
    "    if _oc.environ.get(\"UPRIGHT_FEET_SLIDE\") == \"1\":\n"
    "        from isaaclab.managers import RewardTermCfg as _RewTerm2, SceneEntityCfg as _SceneEntityCfg\n"
    "        import isaaclab_tasks.manager_based.locomotion.velocity.mdp as _loco_mdp\n"
    "        _fsw = float(_oc.environ.get(\"UPRIGHT_FEET_SLIDE_WEIGHT\", \"-1.0\"))\n"
    "        env_cfg.rewards.feet_slide = _RewTerm2(func=_loco_mdp.feet_slide, weight=_fsw,\n"
    "            params={\"sensor_cfg\": _SceneEntityCfg(\"contact_forces\", body_names=\".*_foot\"),\n"
    "                    \"asset_cfg\": _SceneEntityCfg(\"robot\", body_names=\".*_foot\")})\n"
    "        print(f\"[UPRIGHT] feet_slide penalty added, weight={_fsw}\")\n"
)
assert env_anchor in s, "env_cfg anchor line not found"
s = s.replace(env_anchor, env_patch, 1)

open(dst_path, "w").write(s)
print("WROTE", dst_path)
print("has upright patch:", "UPRIGHT_ANTICROUCH" in open(dst_path).read())
