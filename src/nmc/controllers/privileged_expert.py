"""Privileged A* teacher for imitation-learning demonstrations (M1b).

Unlike the reactive ScriptedExpert, this teacher sees ground-truth obstacle
positions (via env.privileged_state) and plans a collision-free path with A* on
an inflated occupancy grid, then follows it with pure pursuit. It replans every
step, so moving obstacles are handled too.

This is the standard teacher-student setup: the teacher may use privileged info
to produce near-optimal demonstrations; the student (frozen MLP-DNN / SNN) later
imitates the mapping (LiDAR observation -> action) without that info.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

# 8-connected neighbourhood with Euclidean step costs.
_NBRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]


@dataclass
class PrivilegedConfig:
    arena_size_m: float = 10.0
    res_m: float = 0.2               # grid cell size
    inflate_m: float = 0.75          # static obstacle inflation: robot radius + margin
    dyn_extra_inflate_m: float = 0.15  # extra clearance for moving obstacles
    lookahead_m: float = 0.6         # short lookahead -> tracks corners tightly
    steer_deadband_deg: float = 8.0
    predict_times_s: tuple = (0.0, 0.3, 0.6)   # moving-obstacle sweep in A*
    # Defensive braking: yield to a moving obstacle predicted to pass close ahead.
    brake_margin_m: float = 0.95     # closest-approach distance that triggers a yield
    brake_horizon_s: float = 1.0     # how far ahead to look for a crossing
    brake_bearing_deg: float = 110.0 # only yield to obstacles roughly ahead


class PrivilegedExpert:
    def __init__(self, cfg: PrivilegedConfig | None = None):
        self.cfg = cfg or PrivilegedConfig()
        self.s = self.cfg.arena_size_m / 2.0
        self.G = int(np.ceil(self.cfg.arena_size_m / self.cfg.res_m))

    # -- world <-> grid --------------------------------------------------
    def _to_cell(self, xy):
        c = int((xy[0] + self.s) / self.cfg.res_m)
        r = int((xy[1] + self.s) / self.cfg.res_m)
        return (int(np.clip(r, 0, self.G - 1)), int(np.clip(c, 0, self.G - 1)))

    def _to_world(self, rc):
        r, c = rc
        return np.array([(c + 0.5) * self.cfg.res_m - self.s,
                         (r + 0.5) * self.cfg.res_m - self.s])

    def _occupancy(self, obst):
        occ = np.zeros((self.G, self.G), dtype=bool)
        rr, cc = np.meshgrid(np.arange(self.G), np.arange(self.G), indexing="ij")
        cx = (cc + 0.5) * self.cfg.res_m - self.s
        cy = (rr + 0.5) * self.cfg.res_m - self.s
        for (ox, oy, orad, vx, vy) in obst:
            moving = (vx * vx + vy * vy) > 1e-6
            infl = self.cfg.inflate_m + (self.cfg.dyn_extra_inflate_m if moving else 0.0)
            rad2 = (orad + infl) ** 2
            times = self.cfg.predict_times_s if moving else (0.0,)
            for t in times:                       # block the swept future positions
                px, py = ox + vx * t, oy + vy * t
                occ |= (cx - px) ** 2 + (cy - py) ** 2 < rad2
        return occ

    # -- A* --------------------------------------------------------------
    def _astar(self, occ, start, goal):
        if occ[goal]:
            goal = self._nearest_free(occ, goal)
            if goal is None:
                return None
        G = self.G

        def h(a, b):
            return np.hypot(a[0] - b[0], a[1] - b[1])

        open_heap = [(h(start, goal), 0.0, start)]
        came, gscore = {}, {start: 0.0}
        while open_heap:
            _, gc, cur = heapq.heappop(open_heap)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return path[::-1]
            if gc > gscore.get(cur, 1e18):
                continue
            for dr, dc, cost in _NBRS:
                nr, nc = cur[0] + dr, cur[1] + dc
                if not (0 <= nr < G and 0 <= nc < G) or occ[nr, nc]:
                    continue
                ng = gc + cost
                if ng < gscore.get((nr, nc), 1e18):
                    gscore[(nr, nc)] = ng
                    came[(nr, nc)] = cur
                    heapq.heappush(open_heap, (ng + h((nr, nc), goal), ng, (nr, nc)))
        return None

    def _nearest_free(self, occ, cell):
        # BFS ring outward to find the closest unblocked cell to `cell`.
        from collections import deque
        seen = {cell}
        q = deque([cell])
        while q:
            r, c = q.popleft()
            if not occ[r, c]:
                return (r, c)
            for dr, dc, _ in _NBRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.G and 0 <= nc < self.G and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        return None

    # -- policy ----------------------------------------------------------
    def _dynamic_threat(self, pos, yaw, obst) -> bool:
        """True if a moving obstacle is predicted to pass within brake_margin of the
        robot's current position, roughly ahead of it -> yield (brake) to let it pass."""
        cfg = self.cfg
        for (ox, oy, orad, vx, vy) in obst:
            if vx * vx + vy * vy < 1e-6:
                continue
            for t in np.linspace(0.0, cfg.brake_horizon_s, 6):
                px, py = ox + vx * t, oy + vy * t
                if np.hypot(px - pos[0], py - pos[1]) < cfg.brake_margin_m:
                    bearing = np.arctan2(py - pos[1], px - pos[0]) - yaw
                    bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
                    if abs(bearing) < np.deg2rad(cfg.brake_bearing_deg):
                        return True
        return False

    def act(self, obs, env) -> int:
        cfg = self.cfg
        pos, yaw, goal, obst = env.privileged_state()

        # Defensive yield: brake to let a crossing obstacle pass before planning.
        if self._dynamic_threat(pos, yaw, obst):
            return 3

        occ = self._occupancy(obst)
        start = self._to_cell(pos)
        occ[start] = False  # never block the robot's own cell
        path = self._astar(occ, start, self._to_cell(goal))

        if not path:
            return 3       # fully blocked (e.g. by a crossing obstacle): brake and wait
        if len(path) < 2:
            target = goal  # essentially at the goal cell already
        else:
            pts = [self._to_world(rc) for rc in path]
            target = pts[-1]
            for pt in pts:                       # first path point beyond lookahead
                if np.linalg.norm(pt - pos) >= cfg.lookahead_m:
                    target = pt
                    break

        vec = target - pos
        desired = np.arctan2(vec[1], vec[0])
        err = np.arctan2(np.sin(desired - yaw), np.cos(desired - yaw))
        db = np.deg2rad(cfg.steer_deadband_deg)
        if err > db:
            return 1   # left
        if err < -db:
            return 2   # right
        return 0       # forward
