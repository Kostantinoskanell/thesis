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


# -- navigation-standard metrics (SOTA eval suite) -------------------------

def path_length(positions: np.ndarray) -> float:
    """Total Euclidean distance travelled along an (T, 2) position trace."""
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def spl(successes, agent_path_lengths, shortest_path_lengths) -> float:
    """Success weighted by (normalized inverse) Path Length -- the standard
    embodied-navigation metric (Anderson et al. 2018, "On Evaluation of Embodied
    Navigation Agents", arXiv:1807.06757):

        SPL = (1/N) sum_i  S_i * l_i / max(p_i, l_i)

    S_i success indicator, l_i shortest-path length, p_i the agent's path length.
    Rewards *efficient* success: a controller that reaches the goal via a long
    detour scores below one that takes the near-optimal route. Success rate alone
    can't see that difference. Range [0, 1]."""
    S = np.asarray(successes, dtype=np.float64)
    p = np.asarray(agent_path_lengths, dtype=np.float64)
    l = np.asarray(shortest_path_lengths, dtype=np.float64)
    denom = np.maximum(p, l)
    ratio = np.where((denom > 0) & (S > 0), l / np.maximum(denom, 1e-9), 0.0)
    return float(np.mean(S * ratio))


def collision_rate(collision_flags) -> float:
    """Fraction of episodes ending in a collision."""
    return float(np.mean(np.asarray(collision_flags, dtype=np.float64)))


def mean_ci95(values):
    """(mean, half-width) of a 95% normal-approx confidence interval.

    For a small number of seeds uses the t-distribution critical value so the
    interval isn't over-confident; falls back gracefully for n<2."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    mean = float(v.mean()) if n else float("nan")
    if n < 2:
        return mean, float("nan")
    sem = v.std(ddof=1) / np.sqrt(n)
    # t critical values for 95% two-sided, small n (df = n-1); ~1.96 for large n.
    t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
               6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    t = t_table.get(n - 1, 1.96)
    return mean, float(t * sem)


def wilson_ci95(k: int, n: int):
    """(rate, low, high) Wilson score interval for a binomial success count --
    the right CI for a single controller's success rate over n episodes
    (better than normal-approx for rates near 0/1 and small n)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return float(p), float(max(0.0, centre - half)), float(min(1.0, centre + half))
