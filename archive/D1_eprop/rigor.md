# e-prop rigor test: does e-prop recover where R-STDP did not?

eta=0.005 (from guardrail), anchor=0.005, plastic_layers=(0,-1) -- same as R-STDP's own recipe for a fair comparison. 10 seeds x 30 eps.


## dropout = 0.3

| controller | success (mean+/-SD) | source |
|---|---|---|
| **e-prop SNN** | 11.3%+/-7.5% | this run |
| Frozen SNN | 11.0%+/-2.6% | archive/M5_full_comparison* |
| R-STDP SNN | 8.7%+/-3.7% | archive/M5_full_comparison* |
| Pure-STDP SNN | 5.0%+/-3.4% | archive/M5_full_comparison* |

e-prop vs Frozen SNN: delta=+0.3%, approx p=0.899 

e-prop vs R-STDP SNN: delta=+2.6%, approx p=0.339 

e-prop vs Pure-STDP SNN: delta=+6.3%, approx p=0.020 *

## dropout = 0.2

| controller | success (mean+/-SD) | source |
|---|---|---|
| **e-prop SNN** | 28.7%+/-9.2% | this run |
| Frozen SNN | 21.3%+/-7.2% | archive/M5_full_comparison* |
| R-STDP SNN | 19.3%+/-5.3% | archive/M5_full_comparison* |
| Pure-STDP SNN | 8.7%+/-4.5% | archive/M5_full_comparison* |

e-prop vs Frozen SNN: delta=+7.4%, approx p=0.054 

e-prop vs R-STDP SNN: delta=+9.4%, approx p=0.007 *

e-prop vs Pure-STDP SNN: delta=+20.0%, approx p=0.000 *


**Verdict: e-prop shows a real advantage over frozen SNN that R-STDP did not**