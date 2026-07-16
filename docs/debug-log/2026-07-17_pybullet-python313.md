# PyBullet won't install on Python 3.13 (Windows)

_2026-07-17 · severity: blocker (environment setup)_

## Symptom
`pip install pybullet` failed two ways:
1. `--only-binary` → "No matching distribution found for pybullet" (no cp313 wheel).
2. Source build → `error: Microsoft Visual C++ 14.0 or greater is required`.

The base interpreter is miniconda **Python 3.13**; PyBullet ships no 3.13 wheel and the
sdist needs an MSVC toolchain that isn't installed.

## Cause
- PyBullet's newest release predates cp313 Windows wheels.
- torch / snntorch are also unreliable on 3.13.

## Fix
Create a dedicated **conda env on Python 3.11** and install PyBullet from **conda-forge**
(prebuilt binary — no compiler needed):

```powershell
conda create -n nmc python=3.11 -y
conda install -n nmc -c conda-forge pybullet numpy matplotlib -y
```

Verified: `conda run -n nmc python -c "import pybullet as p; p.connect(p.DIRECT)"` → id 0.

## Lesson
On Windows, prefer **conda-forge** for C/C++-extension scientific packages to avoid the
MSVC build-tools rabbit hole. Pin the whole project to Python 3.11 until the torch/
pybullet/snntorch stack has stable 3.13 wheels. Run everything via `conda run -n nmc`.
