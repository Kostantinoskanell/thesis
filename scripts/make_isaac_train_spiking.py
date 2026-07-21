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

# (2) replace the OnPolicyRunner construction line with a patched one
orig = "        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)"
patch = (
    "        _sp = agent_cfg.to_dict()\n"
    "        _sp[\"actor\"][\"class_name\"] = \"nmc.locomotion.rsl_rl_spiking:SpikingActorMLPModel\"\n"
    "        _sp[\"actor\"].update({\"in_pop\": 10, \"out_pop\": 10, \"spiking_T\": 8, \"obs_normalization\": True,\n"
    "                              \"enc_sigma\": 0.3872983, \"actor_lr_scale\": 1.0, \"decoder_tanh\": False})\n"
    "        import pprint; print(\"[SPIKING] actor cfg ->\"); pprint.pprint(_sp[\"actor\"])\n"
    "        runner = OnPolicyRunner(env, _sp, log_dir=log_dir, device=agent_cfg.device)"
)
assert orig in s, "anchor line not found"
s = s.replace(orig, patch, 1)

open(dst_path, "w").write(s)
print("WROTE", dst_path)
print("has spiking patch:", "SpikingActorMLPModel" in open(dst_path).read())
