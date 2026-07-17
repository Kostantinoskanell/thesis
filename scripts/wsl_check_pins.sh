#!/usr/bin/env bash
# Show the exact mujoco/jax/brax version constraints playground 0.0.5 declares,
# to find a coherent set (or confirm a deadlock).
cd ~
source nmc-rl2/bin/activate
python - <<'PY'
import urllib.request, json
d = json.load(urllib.request.urlopen("https://pypi.org/pypi/playground/0.0.5/json"))
reqs = d["info"].get("requires_dist") or []
print("playground 0.0.5 requires:")
for r in reqs:
    if any(x in r.lower() for x in ["mujoco", "jax", "brax"]):
        print("  ", r)
# When did mujoco-mjx Data get `_impl`? check a few versions' metadata for jax pin
print("\nmujoco-mjx jax requirement by version:")
for v in ["3.3.1", "3.3.3", "3.3.5", "3.3.7", "3.4.0"]:
    try:
        m = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/mujoco-mjx/{v}/json"))
        jr = [r for r in (m["info"].get("requires_dist") or []) if "jax" in r.lower()]
        print(f"  mjx {v}: {jr}")
    except Exception as e:
        print(f"  mjx {v}: ERR {e}")
PY
