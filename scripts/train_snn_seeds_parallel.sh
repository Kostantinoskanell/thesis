#!/usr/bin/env bash
# Train the M3 SNN seeds in parallel. A single small-net BPTT training does not
# saturate this 20-core box (small matmuls scale poorly past ~5 threads), so we
# run 4 trainings concurrently, each capped to 5 threads -> all finish in roughly
# ONE seed's wall-time instead of 4x. 3 ALIF seeds (for CIs) + 1 vanilla-LIF
# reference (for the LIF->ALIF ablation).
#
# Run:  conda run -n nmc bash scripts/train_snn_seeds_parallel.sh
set -u
cd /mnt/c/Users/hapos/Desktop/thesis 2>/dev/null || cd "C:/Users/hapos/Desktop/thesis"
export KMP_DUPLICATE_LIB_OK=TRUE

train() {  # threads, extra-args...
  local th="$1"; shift
  OMP_NUM_THREADS="$th" MKL_NUM_THREADS="$th" NMC_TORCH_THREADS="$th" \
    python scripts/train_snn_go2.py --epochs 22 --tsteps 20 "$@"
}

train 5 --seed 0 --neuron alif --out assets/snn_seeds/snn_seed0.pt \
      --figdir archive/M3_snn_pretrain_go2 &
train 5 --seed 1 --neuron alif --out assets/snn_seeds/snn_seed1.pt \
      --figdir archive/M3_snn_pretrain_go2 &
train 5 --seed 2 --neuron alif --out assets/snn_seeds/snn_seed2.pt \
      --figdir archive/M3_snn_pretrain_go2 &
train 5 --seed 0 --neuron lif  --out assets/snn_seeds_lif/snn_lif0.pt \
      --figdir archive/M3_snn_pretrain_go2/lif &
wait
echo "ALL SNN TRAINING DONE"
