# Isaac Lab on WSL2 (16GB host / 8GB VRAM) — bring-up war story

**Milestone:** L-track kickoff (extending the SNN + plasticity into the locomotion layer).
**Goal:** run Isaac Lab's Go2 flat-velocity RL task on the lab laptop, as the training
substrate for a *spiking* locomotion policy (the "second thesis" the M4c ice result
motivated — plasticity can't fix slipping from the nav layer, so gait itself must adapt).

## The headline finding
**The hardware was never the blocker.** WSL sees only 7.6 GB RAM (WSL2 caps at ~50% of the
16 GB host) and the GPU has 8 GB VRAM — both roughly *half* Isaac Lab's stated minimums
(32 GB / 16 GB). Every failure along the way was environment plumbing, not capacity: Isaac
Sim booted headless, built 32 Go2 environments, and ran PPO training iterations on the 8 GB
GPU (`TRAIN_EXIT=0`, iteration time ~0.77 s). The stated minimums are soft for a small
env-count locomotion task.

## The four plumbing blockers (in the order they surfaced)
1. **Headless UI-extension segfault.** A raw `SimulationApp({"headless": True})` with the
   `isaacsim[all]` package segfaulted (signal 11) in the URDF-importer *UI* extension at
   only 1.45 GB used — not an OOM. Fix: launch through **Isaac Lab's `AppLauncher`**
   (its headless app profile doesn't load the crashing GUI extensions).
2. **`libhdx.so: cannot open shared object file`.** The physx native plugins couldn't find
   their sibling libs when python is launched directly (no isaacsim launcher env). Fix:
   prepend **every `bin/` dir under the isaacsim pip package** to `LD_LIBRARY_PATH`
   (184 dirs; build it with `find "$ISAAC_PKG" -type d -name bin`).
3. **`libSM.so.6: cannot open shared object file`.** Minimal Ubuntu 24.04 WSL lacks the
   X11/GL system libs Isaac Sim needs. Fix: `apt-get install` the standard set (libsm6,
   libice6, libxext6, libxt6, libgl1, libegl1, libglu1-mesa, libxrandr2, …). Ran it as
   **`wsl -d Ubuntu -u root`** — WSL grants host-side root with no password, sidestepping
   the sudo-password prompt entirely (the user's Windows PIN is *not* the WSL sudo password).
4. **`munmap_chunk(): invalid pointer` (SIGABRT) at "Starting the simulation".** Got all the
   way to 32 envs created, then aborted. Cause: PhysX's GPU pipeline couldn't load
   `libcuda.so`, which in WSL lives at the nonstandard **`/usr/lib/wsl/lib`**. Fix: prepend
   `/usr/lib/wsl/lib` to `LD_LIBRARY_PATH`.

## Two host-side gotchas that wasted time (both about the Windows→WSL boundary)
- **Git Bash path mangling.** Running `wsl … bash /home/hapos/smoke.sh` from the Bash tool,
  Git Bash rewrote `/home/hapos/…` to `C:/Program Files/Git/home/hapos/…` (MSYS auto path
  conversion). Prefix commands with **`MSYS_NO_PATHCONV=1`**. This also silently emptied
  several `$VAR` expansions earlier, causing confusing "file not found / empty variable"
  symptoms.
- **CRLF line endings.** Scripts written from Windows tools get `\r\n`; bash chokes (paths
  gain a trailing `\r`, so `: > "$LOG"` writes to a bogus name and the script no-ops at
  exit 0). Strip with `tr -d '\015' < winfile > /home/hapos/smoke.sh` before running.

## Environment (reproducible)
- Ubuntu 24.04.3 (WSL2), Miniconda env **`isaac`** = Python **3.10** (isaacsim wheels don't
  support the system's 3.12; conda-forge env needed `--override-channels -c conda-forge` to
  dodge the defaults-channel ToS block on conda 26.x).
- **Isaac Sim 4.5.0** (`pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url
  https://pypi.nvidia.com`, ~15 GB, `OMNI_KIT_ACCEPT_EULA=YES`).
- **Isaac Lab 2.3.2** source packages installed editable directly (bypassing `isaaclab.sh
  --install`, whose `sudo apt` cmake step hangs on the password prompt under a no-stdin
  background run); cmake supplied via conda-forge; `rsl-rl-lib` 5.4.2.
- Launch recipe captured in `scripts/wsl_isaac_go2_smoke.sh` (task
  `Isaac-Velocity-Flat-Unitree-Go2-v0`, `--headless --num_envs 32`).

## Still open / next
- **Peak VRAM at a real env count** — the 32-env smoke freed VRAM before it could be
  measured; need a monitored run to find the max `num_envs` that fits 8 GB (stock config
  uses 4096; expect ~1024–2048 here).
- `.wslconfig` (12 GB RAM + 32 GB swap) is staged but **was not needed** for this smoke;
  keep it for larger env counts, which may push RAM.
- The `libcuda.so` PhysX *warning* still prints even though CUDA binds to `cuda:0` and
  training runs — cosmetic, left as-is.

## Lesson
"Requirements: 16 GB VRAM" scared us off, but the real gate was a stack of missing libs and
Windows↔WSL quoting/path/line-ending friction. Peel the actual error each time
(signal 11 ≠ OOM; check `Maximum resident set size` and `dmesg` for real OOMs) instead of
trusting the spec sheet — the box runs it.
