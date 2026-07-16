"""Evaluation metrics (proposal Sec. 4.3).

All metrics take per-timestep episode logs and return scalars comparable across
controllers, matching the definitions table in the proposal.
"""

from __future__ import annotations

import numpy as np


def synops(spike_raster: np.ndarray, weight_mask: np.ndarray | None = None) -> float:
    """SynOps = sum_t sum_ij s_i(t) * 1[w_ij != 0]  (proposal Eq. in Sec. 4.3).

    spike_raster : (T, N_pre) presynaptic spikes
    weight_mask  : (N_post, N_pre) nonzero-weight mask; if None assume dense.
    Returns average synaptic operations per timestep.
    """
    fan_out = weight_mask.astype(bool).sum(0) if weight_mask is not None else None
    per_step = spike_raster @ fan_out if fan_out is not None else spike_raster.sum(1) * 1.0
    return float(np.mean(per_step))


def recovery_time(success_series: np.ndarray, times: np.ndarray, shift_time: float,
                  baseline_window_s: float = 10.0, tol: float = 0.10) -> float:
    """Seconds after the shift until success rate returns within `tol` of the
    pre-shift baseline (proposal Sec. 4.2, step 3).

    Returns np.inf if the controller never recovers (the frozen baselines).
    """
    pre = success_series[(times < shift_time) & (times >= shift_time - baseline_window_s)]
    if pre.size == 0:
        return float("nan")
    baseline = pre.mean()
    threshold = baseline * (1.0 - tol)
    post_mask = times >= shift_time
    post_t, post_s = times[post_mask], success_series[post_mask]
    recovered = post_s >= threshold
    if not recovered.any():
        return float("inf")
    return float(post_t[np.argmax(recovered)] - shift_time)


def response_latency_ms(t_action: np.ndarray, t_sensor: np.ndarray) -> float:
    """Mean T_action - T_sensor in ms."""
    return float(np.mean(t_action - t_sensor) * 1000.0)


def graceful_degradation(success_by_noise: dict[float, float]) -> np.ndarray:
    """Return success rates ordered by noise sigma, for the robustness curve (H4)."""
    sigmas = sorted(success_by_noise)
    return np.array([success_by_noise[s] for s in sigmas])
