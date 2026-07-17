#!/usr/bin/env bash
# Diagnose the installed MuJoCo Playground registry API + find the Go2 env name.
cd ~
source nmc-rl/bin/activate
python - <<'PY'
import mujoco_playground as mp
print("mujoco_playground:", getattr(mp, "__version__", "?"), mp.__file__)
from mujoco_playground import registry
print("registry attrs:", [a for a in dir(registry) if not a.startswith("_")])
found = {}
for cat in ["locomotion", "manipulation", "dm_control_suite"]:
    sub = getattr(registry, cat, None)
    if sub is None:
        continue
    envs = None
    for attr in ("ALL_ENVS", "_envs", "ALL"):
        envs = getattr(sub, attr, None)
        if envs:
            break
    try:
        names = list(envs.keys()) if hasattr(envs, "keys") else list(envs)
    except Exception:
        names = []
    found[cat] = names
    go2 = [n for n in names if "go2" in n.lower()]
    if go2:
        print(f"{cat} Go2 envs:", go2)
print("locomotion count:", len(found.get("locomotion", [])))
print("sample locomotion:", found.get("locomotion", [])[:12])
PY
