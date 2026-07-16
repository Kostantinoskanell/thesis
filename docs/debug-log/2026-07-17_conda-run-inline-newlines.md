# `conda run ... python -c` silently fails on multi-line scripts

_2026-07-17 · severity: gotcha (tooling)_

## Symptom
`conda run -n nmc python -c "<multi-line script>"` produced **no output** (and no error
in some shells). A quick capability check appeared to "hang" or do nothing.

## Cause
`conda run` does not support inline `-c` scripts whose arguments contain **newlines**:
> NotImplementedError: Support for scripts where arguments contain newlines not implemented.

Single-line `-c` works; script *files* work. Only inline multi-line `-c` breaks.

## Fix
Always run **script files** in the conda env, never inline multi-line `-c`:
```powershell
conda run -n nmc python scripts/whatever.py
```
For throwaway checks, write a tiny temp `.py` file and run that.

## Lesson
This wasted a couple of iterations early (checks that "printed nothing"). Standard
practice now: every env-dependent snippet goes in a file under `scripts/`.
