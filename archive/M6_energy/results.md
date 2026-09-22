# M6 — nav-layer energy (H2), 45 nm Horowitz / SynOps method

Measured on closed-loop rollouts, 3 seeds x 10 eps, sensor-dropout=0.20 (the corrected severity from M5).

Per-op: AC 0.9 pJ, MUL 3.7 pJ, MAC 4.6 pJ. SNN T=20.


## Per-decision energy breakdown (nJ)

| controller | firing rate | inference | neurons | encoder | learning | **total** | vs frozen MLP |
|---|---|---|---|---|---|---|---|
| Frozen MLP | — | 1321.7 | 0.0 | 0.0 | 0.0 | **1321.7** | 1.00x COSTLIER |
| Online MLP | — | 1321.7 | 0.0 | 0.0 | 7586.7 | **8908.4** | 0.15x COSTLIER |
| Frozen SNN | 1.50% | 125.6 | 420.0 | 4.2 | 0.0 | **549.7** | 2.40x cheaper |
| Pure-STDP SNN | 0.36% | 59.0 | 420.0 | 4.2 | 15538.5 | **16021.6** | 0.08x COSTLIER |
| ↳ Pure-STDP SNN, event-driven update (FPGA/crossbar target) | 0.36% | 59.0 | 420.0 | 4.2 | 587.5 | **1070.6** | 1.23x cheaper |
| R-STDP SNN | 1.49% | 125.1 | 420.0 | 4.2 | 15538.5 | **16087.7** | 0.08x COSTLIER |
| ↳ R-STDP SNN, event-driven update (FPGA/crossbar target) | 1.49% | 125.1 | 420.0 | 4.2 | 747.3 | **1296.5** | 1.02x cheaper |
| TM-NORM SNN | 3.09% | 223.9 | 420.0 | 4.2 | 133.7 | **781.8** | 1.69x cheaper |

## Literature-standard view (SynOps vs MACs only, no neuron/encoder/learning)

| controller | SynOps or MACs | energy (nJ) | vs frozen MLP |
|---|---|---|---|
| Frozen MLP | 283,648 MACs | 1304.8 | 1.00x |
| Frozen SNN | 139,463 SynOps | 125.5 | 10.4x cheaper |
| Pure-STDP SNN | 65,457 SynOps | 58.9 | 22.1x cheaper |
| R-STDP SNN | 138,929 SynOps | 125.0 | 10.4x cheaper |
| TM-NORM SNN | 248,750 SynOps | 223.9 | 5.8x cheaper |

## Energy per SUCCESSFUL navigation (deployment-relevant)

energy/decision x decisions/episode / success-rate — a controller that is cheap but fails is not efficient.

| controller | nJ/decision | decisions/ep | success (M5, dropout 0.20) | **µJ per success** |
|---|---|---|---|---|
| Frozen MLP | 1321.7 | 586 | 34.0% | **2279.8** |
| Online MLP | 8908.4 | 663 | 24.0% | **24624.2** |
| Frozen SNN | 549.7 | 882 | 21.3% | **2276.4** |
| Pure-STDP SNN | 16021.6 | 461 | 8.7% | **84810.1** |
| R-STDP SNN | 16087.7 | 770 | 19.3% | **64203.6** |
| TM-NORM SNN | 781.8 | 331 | 1.7% | **15232.5** |

## Firing-rate distribution shift (M4b item 6)

Wasserstein distance of the per-decision firing-rate distribution from the clean (unshifted) frozen-SNN reference — does the shift move spike statistics, and does R-STDP move them back?

| distribution | mean rate | Wasserstein vs clean |
|---|---|---|
| frozen SNN, clean | 1.52% | 0 (reference) |
| frozen SNN, shifted | 1.51% | 2.45e-04 |
| R-STDP, shifted (late episodes) | 1.49% | 3.06e-04 |

**Renormalization: NO** — R-STDP's firing statistics are further from the clean reference than the frozen network's (3.06e-04 vs 2.45e-04).