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
