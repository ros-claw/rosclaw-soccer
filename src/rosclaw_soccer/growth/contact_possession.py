"""Bounded, contact-grounded possession evidence; never a motion authority."""

from __future__ import annotations

import math


class ContactPossession:
    """Require fresh foot evidence and continuous slow, nearby, safe ball control.

    Losing control requires a new physical foot contact to reacquire ownership.
    Clock/data faults latch closed. This is evidence bookkeeping, not learning.
    """

    def __init__(self) -> None:
        self.tracking_agent_id: str | None = None
        self.faulted = False
        self._last_time: float | None = None
        self._contact_time = -math.inf
        self._controlled_since: float | None = None

    def _clear(self) -> None:
        self.tracking_agent_id = None
        self._contact_time = -math.inf
        self._controlled_since = None

    def observe(
        self,
        *,
        time_sec: float,
        foot_agents: frozenset[str],
        interrupted: bool,
        ball_speed_mps: float,
        nearest_foot_m: float,
        body_safe: bool,
    ) -> None:
        """Observe one physics step; caller measures contacts above 1 N."""
        if self.faulted:
            return
        values = (time_sec, ball_speed_mps, nearest_foot_m)
        if (
            any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)
            or type(interrupted) is not bool
            or type(body_safe) is not bool
            or not isinstance(foot_agents, frozenset)
            or any(not isinstance(a, str) or not a or len(a) > 128 for a in foot_agents)
            or (self._last_time is not None and time_sec <= self._last_time)
        ):
            self._clear()
            self.faulted = True
            return
        if self._last_time is not None and time_sec - self._last_time > 0.0041:
            self._clear()
        self._last_time = time_sec
        if interrupted or len(foot_agents) > 1:
            self._clear()
            return
        if foot_agents:
            agent = next(iter(foot_agents))
            if agent != self.tracking_agent_id:
                self._clear()
                self.tracking_agent_id = agent
            self._contact_time = time_sec
        if self.tracking_agent_id is None:
            return
        if (
            not body_safe
            or ball_speed_mps > 0.5
            or nearest_foot_m > 0.35
            or time_sec - self._contact_time > 3.0
        ):
            self._clear()
            return
        if self._controlled_since is None:
            self._controlled_since = time_sec

    def owner_at(self, time_sec: float) -> str | None:
        if (
            self.faulted
            or type(time_sec) not in (int, float)
            or not math.isfinite(time_sec)
            or self._last_time is None
            or not 0 <= time_sec - self._last_time <= 0.0041
            or self._controlled_since is None
            or time_sec - self._controlled_since < 0.3 - 1e-9
            or time_sec - self._contact_time > 3.0
        ):
            return None
        return self.tracking_agent_id
