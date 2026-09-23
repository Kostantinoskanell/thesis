# e-prop guardrail: base-distribution stability across eta

Frozen SNN reference (no plasticity): 42% over 24 eps.

| eta | mean success (5 seeds) | all weights finite? | verdict |
|---|---|---|---|
| 0.0005 | 44% | True | OK |
| 0.001 | 45% | True | OK |
| 0.002 | 43% | True | OK |
| 0.005 | 38% | True | OK |
| 0.01 | 30% | True | DEGRADED |
| 0.02 | 24% | True | DEGRADED |
| 0.05 | 13% | True | DEGRADED |
| 0.1 | 5% | True | DEGRADED |
| 0.2 | 2% | True | DEGRADED |

**Recommended eta for the rigor test: 0.005** (largest that stays within 10pts of frozen SNN, weights finite)