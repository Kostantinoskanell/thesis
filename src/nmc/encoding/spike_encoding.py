"""Sensor -> spike-train encoders (proposal Sec. 3.2, step 1).

Two schemes:

* rate coding      -- firing probability proportional to stimulus intensity;
                      used for the distance channels.
* time-to-first-spike (TTFS) -- higher intensity fires earlier within the
                      encoding window; used for "urgency" (near-obstacle)
                      channels where reaction latency matters.

Inputs are assumed already normalized to [0, 1] (e.g. LiDAR range / max_range,
inverted so that *near* = high intensity for the urgency channels).
"""

from __future__ import annotations

import numpy as np


def rate_encode(x: np.ndarray, n_steps: int, max_rate_hz: float = 200.0,
                dt_ms: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Poisson rate code.

    Parameters
    ----------
    x        : (F,) intensities in [0, 1]
    n_steps  : encoding window length in timesteps
    Returns  : (n_steps, F) binary spike raster
    """
    rng = rng if rng is not None else np.random.default_rng()
    x = np.clip(x, 0.0, 1.0)
    p_spike = x * (max_rate_hz * dt_ms / 1000.0)  # per-step spike probability
    draws = rng.random((n_steps, x.shape[0]))
    return (draws < p_spike[None, :]).astype(np.float32)


def ttfs_encode(x: np.ndarray, n_steps: int) -> np.ndarray:
    """Time-to-first-spike code: intensity 1 fires at t=0, intensity ->0 fires late.

    Each feature emits exactly one spike (or none if intensity is 0).
    Returns (n_steps, F) binary raster.
    """
    x = np.clip(x, 0.0, 1.0)
    raster = np.zeros((n_steps, x.shape[0]), dtype=np.float32)
    active = x > 0.0
    # latency in [0, n_steps-1]; higher intensity -> earlier (smaller index)
    latency = np.round((1.0 - x) * (n_steps - 1)).astype(int)
    for f in np.nonzero(active)[0]:
        raster[latency[f], f] = 1.0
    return raster


def encode_observation(lidar: np.ndarray, vel: np.ndarray, n_steps: int,
                       max_range: float = 8.0, rng: np.random.Generator | None = None
                       ) -> np.ndarray:
    """Combine LiDAR + velocity into one spike raster.

    * LiDAR distances -> rate code on (1 - d/max_range)  (near = fires more)
    * A few nearest beams -> TTFS urgency channels
    * velocity magnitude -> rate code

    Returns (n_steps, F_total) raster ready to feed the LIF-SNN.
    """
    rng = rng if rng is not None else np.random.default_rng()
    d_norm = np.clip(lidar / max_range, 0.0, 1.0)
    nearness = 1.0 - d_norm

    rate_dist = rate_encode(nearness, n_steps, rng=rng)                 # (T, n_beams)
    urgency = ttfs_encode(nearness, n_steps)                           # (T, n_beams)
    v_norm = np.clip(np.linalg.norm(vel) / 2.0, 0.0, 1.0)
    rate_vel = rate_encode(np.array([v_norm]), n_steps, rng=rng)       # (T, 1)

    return np.concatenate([rate_dist, urgency, rate_vel], axis=1)


# -- full navigation-observation encoder (M3) ------------------------------
# The Go2NavEnv observation is the flat 37-vector the MLP consumes:
#   [ lidar(32) already normalized to [0,1], goal_dx, goal_dy, heading_err, v, omega ]
# encode_observation() above only handled lidar+velocity and DROPPED the goal
# and heading -- essential for navigation. encode_nav_obs encodes the whole
# vector so the SNN sees exactly the same information as the MLP (fair M3
# comparison). Signed features (goal vector, heading error, yaw rate) are split
# into on/off channels (positive part, negative part), the standard neuromorphic
# way to rate-code a signed value with non-negative firing rates.

_ARENA_HALF_M = 5.0      # goal_dx/dy normalization (arena is 10 m)
_V_MAX = 0.8             # forward-speed normalization
_OMEGA_MAX = 1.5         # yaw-rate normalization


def _on_off(x: float) -> tuple[float, float]:
    """Split a signed value in [-1, 1] into (positive part, negative part)."""
    return max(x, 0.0), max(-x, 0.0)


def encode_nav_obs(obs: np.ndarray, n_steps: int, n_lidar: int = 32,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Encode the full 37-dim nav observation into a spike raster (n_steps, F).

    Channel layout (F = 2*n_lidar + 9 = 73 for n_lidar=32):
      rate(nearness)  n_lidar     distance channels (near = fires more)
      ttfs(nearness)  n_lidar     urgency channels (near = fires earlier)
      rate goal_dx+   1           goal direction, on/off split, arena-normalized
      rate goal_dx-   1
      rate goal_dy+   1
      rate goal_dy-   1
      rate heading+   1           heading error / pi, on/off split
      rate heading-   1
      rate v          1           forward speed, normalized
      rate omega+     1           yaw rate / omega_max, on/off split
      rate omega-     1
    """
    rng = rng if rng is not None else np.random.default_rng()
    obs = np.asarray(obs, dtype=np.float64)
    lidar = obs[:n_lidar]                       # already in [0,1] (dist/max_range)
    goal_dx, goal_dy, heading_err, v, omega = obs[n_lidar:n_lidar + 5]

    nearness = np.clip(1.0 - lidar, 0.0, 1.0)
    rate_dist = rate_encode(nearness, n_steps, rng=rng)
    urgency = ttfs_encode(nearness, n_steps)

    gx_on, gx_off = _on_off(np.clip(goal_dx / _ARENA_HALF_M, -1, 1))
    gy_on, gy_off = _on_off(np.clip(goal_dy / _ARENA_HALF_M, -1, 1))
    h_on, h_off = _on_off(np.clip(heading_err / np.pi, -1, 1))
    w_on, w_off = _on_off(np.clip(omega / _OMEGA_MAX, -1, 1))
    v_n = np.clip(v / _V_MAX, 0.0, 1.0)
    scalars = np.array([gx_on, gx_off, gy_on, gy_off, h_on, h_off, v_n, w_on, w_off])
    rate_scalars = rate_encode(scalars, n_steps, rng=rng)

    return np.concatenate([rate_dist, urgency, rate_scalars], axis=1).astype(np.float32)


def encode_nav_obs_dim(n_lidar: int = 32) -> int:
    """Input dimension produced by encode_nav_obs (for building the LIF-SNN)."""
    return 2 * n_lidar + 9
