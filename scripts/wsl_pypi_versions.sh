#!/usr/bin/env bash
# Query PyPI JSON for real version lists (pip's version listing is unreliable here).
cd ~
source nmc-rl/bin/activate
python - <<'PY'
import urllib.request, json
def vers(pkg):
    d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json"))
    return sorted(d["releases"].keys())
for pkg in ["playground", "mujoco", "mujoco-mjx", "brax"]:
    try:
        vs = vers(pkg)
        print(f"{pkg}: {vs[-14:]}")
    except Exception as e:
        print(pkg, "ERR", e)
# Which playground releases DON'T require warp-lang? Check each release's deps.
print("--- playground releases requiring warp-lang ---")
d = json.load(urllib.request.urlopen("https://pypi.org/pypi/playground/json"))
import urllib.request as u
for v in sorted(d["releases"].keys()):
    try:
        meta = json.load(u.urlopen(f"https://pypi.org/pypi/playground/{v}/json"))
        reqs = meta["info"].get("requires_dist") or []
        warp = any("warp" in r.lower() for r in reqs)
        print(f"  {v}: warp={'YES' if warp else 'no'}")
    except Exception as e:
        print(f"  {v}: ERR {e}")
PY
