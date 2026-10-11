"""
waypoints.py  —  intermediate-stop tracking for the ECML 2026 challenge
======================================================================

In the competition every train has a line: start -> intermediate stops -> target.
A stop only counts as served if the train is STOPPED on one of the stop's cells
(ECML2026Rewards: -50 per stop not served). This module keeps track of which stop
each train should head for next and gives shortest-path distances to it, so the
observation, the expert action mask and the reward shaping can all aim for the
next stop instead of the final target.

One WaypointTracker lives on the observation builder (env.obs_builder.wp), so the
same state is shared by everything that runs in that process.

Switch: Config.serve_intermediate_stops. In Flatland 4.2.6 ECML2026Rewards charges
the collision penalty (250 x speed before the step) on EVERY MOVING -> STOPPED
transition, including a deliberate stop. So stopping at a stop costs 250 while
skipping it costs 50, and the switch is off by default: trains then head straight
for their final target. Turn it on if the penalty is changed
(https://github.com/flatland-association/flatland-rl/issues/280).
"""
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.step_utils.states import TrainState

from config import Config
from shortestpath import rail_transitions

Pos = Tuple[int, int]


class WaypointTracker:
    def __init__(self, serve_stops: Optional[bool] = None):
        self.serve_stops = (getattr(Config, "serve_intermediate_stops", False)
                            if serve_stops is None else serve_stops)
        self.env = None
        self.next_idx: Dict[int, int] = {}          # handle -> index into agent.waypoints of the next stop
        self._dist_cache: Dict[Pos, np.ndarray] = {} # target cell -> dist[row, col, dir]
        self._rail_key: Optional[int] = None         # detects a changed rail layout

    # ------------------------------------------------------------------
    def reset(self, env) -> None:
        """Call at the start of every episode (after the env has its agents)."""
        self.env = env
        self.next_idx = {a.handle: 1 for a in env.agents}
        # The competition map is fixed, so BFS maps stay valid across episodes.
        # Only drop them if the rail layout actually changed.
        rail_key = hash(env.rail.grid.tobytes())
        if rail_key != self._rail_key:
            self._dist_cache.clear()
            self._rail_key = rail_key

    def update(self) -> None:
        """Advance trains that are standing still on their next intermediate stop."""
        for a in self.env.agents:
            wps = self._waypoints(a)
            i = self.next_idx.get(a.handle, 1)
            if i >= len(wps) - 1:            # heading for the final target already
                continue
            if a.position is None or a.state != TrainState.STOPPED:
                continue
            if any(w.position == a.position for w in wps[i]):
                self.next_idx[a.handle] = i + 1

    # ------------------------------------------------------------------
    def at_unserved_stop(self, handle: int) -> bool:
        """True if the train is on its next intermediate stop and hasn't stopped there yet."""
        a = self.env.agents[handle]
        wps = self._waypoints(a)
        i = self.next_idx.get(handle, 1)
        return (a.position is not None and i < len(wps) - 1
                and any(w.position == a.position for w in wps[i]))

    def target(self, handle: int) -> Pos:
        """Cell the train should currently head for (next stop, or the final target)."""
        a = self.env.agents[handle]
        wps = self._waypoints(a)
        i = self.next_idx.get(handle, 1)
        if i < len(wps) - 1:
            return tuple(wps[i][0].position)
        return tuple(a.target)

    def dist_grid(self, target: Pos) -> np.ndarray:
        """dist[row, col, dir] = shortest number of cells to reach `target` (inf if unreachable)."""
        target = tuple(target)
        if target not in self._dist_cache:
            self._dist_cache[target] = self._bfs(target)
        return self._dist_cache[target]

    def distance(self, handle: int, target: Optional[Pos] = None) -> float:
        """Distance from the train's current (or starting) cell to `target` (default: its current target)."""
        a = self.env.agents[handle]
        if a.state == TrainState.DONE:
            return 0.0
        pos = a.position if a.position is not None else a.initial_position
        d = a.direction if a.direction is not None else a.initial_direction
        return float(self.dist_grid(target or self.target(handle))[pos[0], pos[1], d])

    def stops_served_fraction(self, handle: int) -> float:
        """Share of this train's intermediate stops it has stopped at (0 if stop-serving is off)."""
        if not self.serve_stops:
            return 0.0
        wps = self._waypoints(self.env.agents[handle])
        n_inter = len(wps) - 2
        if n_inter <= 0:
            return 1.0
        return min(self.next_idx.get(handle, 1) - 1, n_inter) / n_inter

    # ------------------------------------------------------------------
    def _waypoints(self, agent):
        # Older Flatland versions have no intermediate stops, and with
        # serve_stops off we ignore them: treat as start -> target.
        wps = getattr(agent, "waypoints", None) if self.serve_stops else None
        return wps if wps else [[None], [None]]

    def _bfs(self, target: Pos) -> np.ndarray:
        """Backward BFS over the directed rail graph from every direction at `target`."""
        rail = self.env.rail
        h, w = self.env.height, self.env.width
        dist = np.full((h, w, 4), np.inf)
        q = deque()
        for d in range(4):
            if any(rail_transitions(rail, target, d)):
                dist[target[0], target[1], d] = 0
                q.append((target, d, 0))
        while q:
            pos, d, c = q.popleft()
            prev = get_new_position(pos, (d + 2) % 4)       # cell we came from, travelling in direction d
            if not (0 <= prev[0] < h and 0 <= prev[1] < w):
                continue
            for pd in range(4):                              # heading while standing on `prev`
                if rail_transitions(rail, prev, pd)[d] and c + 1 < dist[prev[0], prev[1], pd]:
                    dist[prev[0], prev[1], pd] = c + 1
                    q.append((prev, pd, c + 1))
        return dist
