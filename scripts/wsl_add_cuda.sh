#!/usr/bin/env bash
# Add the CUDA jaxlib/plugin MATCHING the jax version the pinned stack pulled
# (do NOT let it upgrade jax to a warp-requiring version). Confirm GPU visibility.
set -e
cd ~
source nmc-rl2/bin/activate
JV=$(python -c "import jax; print(jax.__version__)")
echo "pinned jax version: $JV -> installing matching CUDA plugin"
pip install -q "jax[cuda12]==$JV"
python - <<'PY'
import jax
print("jax", jax.__version__, "devices:", jax.devices())
print("GPU_OK" if any(d.platform == "gpu" for d in jax.devices()) else "GPU_MISSING")
PY
echo "CUDA_ADD_DONE"
