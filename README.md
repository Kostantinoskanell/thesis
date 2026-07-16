# Memristor-Inspired Neuromorphic Control for Robotics

Undergraduate thesis (ECE, University of Patras). A hardware–software co-design
comparing a spiking neural network (SNN) with reward-modulated STDP (R-STDP)
against DNN and RL baselines on an obstacle-avoidance task that undergoes a
**mid-episode distribution shift**, with an FPGA co-processor that is
*structurally analogous* to a memristive crossbar for the synaptic weight update.

See [`proposal/proposal.tex`](proposal/proposal.tex) for the full proposal and
[`ROADMAP.md`](ROADMAP.md) for the milestone plan and locked scoping decisions.

## Layout

```
src/nmc/
  envs/nav_env.py         PyBullet nav env + mid-episode shift protocol
  encoding/               rate + TTFS spike encoders
  controllers/mlp.py      frozen + online-TD MLP baselines
  controllers/snn.py      LIF-SNN controller (+ frozen-SNN ablation)
  plasticity/stdp.py      online STDP / R-STDP  (golden reference for the FPGA)
  eval/metrics.py         SynOps, recovery time, robustness, latency
fpga/                     HDL + PyNQ host (Phase 4)
tests/                    unit tests (plasticity golden reference)
docs/references/          curated, topic-grouped citations + study plan + SOTA log
docs/debug-log/           dated war-stories: symptom -> cause -> fix -> lesson
archive/<milestone>/      milestone figures + proof logs (the thesis "journey")
```

**Documentation is first-class here** (see [docs/references/](docs/references/) and
[docs/debug-log/](docs/debug-log/)): every useful source and every non-trivial bug is
recorded as we go, for the write-up and defense.

## Setup — conda env `nmc` (Python 3.11)

Python 3.13 has no PyBullet wheel and the sdist needs MSVC. Use the conda env:

```powershell
conda create -n nmc python=3.11 -y
conda install -n nmc -c conda-forge pybullet numpy matplotlib -y   # prebuilt, no compiler
conda run -n nmc python -m pip install snntorch stable-baselines3 gymnasium wandb
# install the CUDA torch build matching your GPU from pytorch.org
```

Run env-dependent scripts with `conda run -n nmc python ...`.

## Run the tests (numpy-only, no GPU/board needed)

```bash
$env:PYTHONPATH="src"; pytest -q
```

## Status

- **M0** done — scaffold + verified R-STDP golden reference (`archive/M0_scaffold/`).
- **M1** done — PyBullet nav env runs a 60 s episode with the shift at t=30 s
  (`archive/M1_env_shift/`, smoke test PASS).

Next: **M1b** — scripted gap-follower expert + imitation dataset, then **M2** MLP
baselines. See [ROADMAP.md](ROADMAP.md).
