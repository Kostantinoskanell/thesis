# FPGA Synaptic-Update Co-processor (Phase 4)

**Do not start here.** Build only after the pilot (M4) shows R-STDP recovery in
simulation. This directory is scaffolded so the interface is fixed early.

## Scope (honest framing)

The accelerator is **memristor-inspired**, resource-constrained, and
time-multiplexed — *not* a true O(1) analog crossbar and *not* a device-level
memristor emulator. A Zynq-7020 has only 220 DSP slices, so a 512×512 layer
(~2.6e5 MACs) must reuse a bank of processing elements across synapse tiles.

The H3 claim applies to the **inner weight-update kernel only**, reported per
network size against both a CPU loop and a vectorized GPU baseline. The
deliverable is *"find the crossover network size,"* not *"beat the GPU."*

## Datapath (mirrors `src/nmc/plasticity/stdp.py`)

- `spike_time_regs`   — one 16-bit fixed-point Δt register per neuron (0.1 ms res).
- `dt_subtractor`     — tiled parallel `Δt_ij = t_post_i − t_pre_j`.
- `stdp_lut`          — BRAM, 64 bins, **logarithmic** quantization of |Δt|
                        (kernel is steep near 0, flat far out). Validate against
                        `nmc.plasticity.stdp.kernel_reference` to < 2% rel. error
                        *before synthesis*.
- `elig_update`       — eligibility trace register + decay-multiply per synapse (R-STDP).
- `weight_update`     — saturating MAC, clamp to [w_min, w_max].
- `lfsr_noise`        — optional stochasticity unit (memristive property).
- `axi_dma`           — AXI4-Stream burst transfer of spike/weight vectors.

## Layout

```
hdl/    Verilog/SystemVerilog sources + testbenches
pynq/   Python host: AXI-DMA bridge, golden-reference cross-check, timing sweep
```

## Verification gate

`pynq/validate_lut.py` (to be written) must confirm the synthesized LUT
reproduces `kernel_reference()` within tolerance, and a cosim testbench must
match `STDPLearner.step()` on a random spike sequence, before any hardware-in-loop
timing is trusted.
