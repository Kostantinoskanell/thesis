# M5 plasticity-scope / third-factor follow-up (dropout=0.20, 10 seeds x 30 eps)

| variant | mean success | sd |
|---|---|---|
| frozen SNN | 21.3% | 7.2% |
| R-STDP (input+readout, td) [baseline] | 19.3% | 5.3% |
| all-layers + td | 20.7% | 2.0% |
| input+readout + rpe | 21.7% | 7.2% |
| all-layers + rpe | 21.7% | 6.2% |