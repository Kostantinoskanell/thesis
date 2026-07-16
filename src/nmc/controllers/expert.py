"""Scripted navigation expert (M1b).

A reactive gap-follower with goal attraction, operating directly on the env
observation [lidar(32 normalized), goal_dx, goal_dy, heading_err, v, omega].
It provides:
  * expert (obs, action) demonstrations to train the frozen MLP-DNN by imitation
    learning (proposal Sec. 3.1, controller 1), and
  * a sanity reference for how hard the task is.

It is intentionally simple and hand-tuned, not learned. Beams are ordered so
that index i maps to relative bearing theta_i = -pi + i*(2*pi/N); straight ahead
is theta=0 (index N/2), +theta is left, -theta is right (matches NavEnv._raycast).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExpertConfig:
    n_beams: int = 32
    front_arc_deg: float = 100.0     # candidate steering arc
    bubble_trigger: float = 0.22     # if nearest normalized dist < this, bubble it
    bubble_half_deg: float = 30.0    # angular half-width zeroed around nearest obstacle
    gap_thresh: float = 0.25         # normalized dist above which a beam is "open"
    goal_bias: float = 0.30          # weight pulling the chosen gap toward the goal
    steer_deadband_deg: float = 10.0 # |desired heading| below this -> go straight
    hard_stop: float = 0.10          # forward-cone dist that forces an evasive turn
    hard_stop_arc_deg: float = 20.0  # half-arc of the forward safety cone
    erode_k: int = 2                 # min-filter half-width (beams) for body clearance


class ScriptedExpert:
    """Goal-biased Follow-The-Gap (FTG) navigator.

    1. Put a safety bubble around the nearest obstacle (zero those beams so we
       never steer toward the closest hazard even if neighbours look open).
    2. Among front beams that remain open, pick the one maximizing depth while
       staying close to the goal bearing.
    3. Reduce to the 4 discrete NavEnv actions.

    FTG has no local-minima trap (unlike potential fields) and is collision-averse
    thanks to the bubble -- the standard reactive method in F1TENTH-style racing.
    """

    def __init__(self, cfg: ExpertConfig | None = None):
        self.cfg = cfg or ExpertConfig()
        n = self.cfg.n_beams
        self.angles = -np.pi + np.arange(n) * (2 * np.pi / n)
        self._front = np.abs(self.angles) <= np.deg2rad(self.cfg.front_arc_deg)
        self._safety = np.abs(self.angles) <= np.deg2rad(self.cfg.hard_stop_arc_deg)

    def act(self, obs: np.ndarray, env=None) -> int:
        cfg = self.cfg
        n = cfg.n_beams
        raw = obs[:n]
        lidar = raw.copy()
        heading_err = float(obs[n + 2])                 # goal bearing, robot frame

        # 0. Hard safety override on the RAW scan: if anything is close in the
        #    forward cone, turn toward the more open side before anything else.
        if float(raw[self._safety].min()) < cfg.hard_stop:
            left = raw[self.angles > 0].mean()
            right = raw[self.angles < 0].mean()
            return 1 if left >= right else 2

        # 1. Erode the scan by the robot's angular half-width so a direction only
        #    counts as open if the whole BODY fits (point-robot FTG clips corners).
        k = cfg.erode_k
        if k > 0:
            padded = np.concatenate([lidar[-k:], lidar, lidar[:k]])   # wrap-around
            lidar = np.array([padded[i:i + 2 * k + 1].min() for i in range(n)])

        # 2. Safety bubble around the closest obstacle.
        i_min = int(np.argmin(lidar))
        if lidar[i_min] < cfg.bubble_trigger:
            bubble = np.abs(self.angles - self.angles[i_min]) < np.deg2rad(cfg.bubble_half_deg)
            lidar[bubble] = 0.0

        # 3. Choose an open front beam, biased toward the goal bearing.
        open_beam = self._front & (lidar > cfg.gap_thresh)
        if open_beam.any():
            score = lidar - cfg.goal_bias * np.abs(self.angles - heading_err)
            score[~open_beam] = -1e9
            theta = float(self.angles[int(np.argmax(score))])
        else:
            # Fully blocked ahead: rotate toward the more open side.
            left = lidar[self.angles > 0].mean()
            right = lidar[self.angles < 0].mean()
            return 1 if left >= right else 2

        db = np.deg2rad(cfg.steer_deadband_deg)
        if theta > db:
            return 1   # left
        if theta < -db:
            return 2   # right
        return 0       # forward
