# Custom map test: forced detour through the dead-beam sector + one mover

5 controller-seeds x 20 eps = 100 episodes/controller, dropout=0.2, fixed layout (see scripts/custom_map.py).

| controller | success | collision(static) | collision(dynamic) | fell | timeout |
|---|---|---|---|---|---|
| Frozen MLP | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| Online MLP | 0.0% | 0.0% | 0.0% | 5.0% | 95.0% |
| Frozen SNN | 84.0% | 0.0% | 1.0% | 6.0% | 9.0% |
| Pure-STDP SNN | 1.0% | 82.0% | 0.0% | 2.0% | 15.0% |
| R-STDP SNN | 72.0% | 0.0% | 2.0% | 8.0% | 18.0% |
| TM-NORM SNN | 0.0% | 98.0% | 0.0% | 2.0% | 0.0% |

R-STDP vs Frozen SNN (per-seed means, Welch t-test): 72.0% vs 84.0%, p=0.112