"""Explicit SIM training assistant returns, not player actions or official restarts."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rosclaw_soccer.world.match_boundary import ball_exit_reason


@dataclass(frozen=True)
class TrainingBallReturnConfig:
    delay_sec: float = 1.0
    maximum_returns: int = 3
    inward_speed_mps: float = 2.0
    upward_speed_mps: float = 1.8
    release_height_m: float = 0.8
    player_clearance_m: float = 0.75

    def __post_init__(self) -> None:
        for value, lower, upper in (
            (self.delay_sec, 0.5, 5.0),
            (self.inward_speed_mps, 0.5, 3.0),
            (self.upward_speed_mps, 0.0, 3.0),
            (self.release_height_m, 0.3, 1.2),
            (self.player_clearance_m, 0.65, 2.0),
        ):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError("bounded training return parameters required")
        if type(self.maximum_returns) is not int or not 1 <= self.maximum_returns <= 10:
            raise ValueError("bounded explicit return budget required")


@dataclass(frozen=True)
class TrainingBallReturnEvent:
    # 1: whole-ball exit; 2: external release; 3: measured re-entry.
    code: int
    time_sec: float
    return_count: int
    exit_reason: str
    release_position_m: tuple[float, float, float] | None = None
    release_velocity_mps: tuple[float, float, float] | None = None


class TrainingBallReturnReferee:
    """Deterministic, bounded coach throw-in proposals from measured state.

    No simulator or control handle is retained. The simulation owner alone
    applies a release and logs the pre/post ball state. All players retain their
    physical state and clock. Re-entry must be observed before another exit;
    a throw still outside after three seconds is a failed return, not live play.
    """

    def __init__(
        self,
        config: TrainingBallReturnConfig,
        *,
        left_x: float,
        right_x: float,
        radius: float,
        goal_width: float,
        goal_height: float,
    ):
        if not isinstance(config, TrainingBallReturnConfig):
            raise ValueError("explicit training return config required")
        values = (left_x, right_x, radius, goal_width, goal_height)
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in values) or not (
            3 <= right_x - left_x <= 120
            and 0.05 <= radius <= 0.2
            and 0.5 <= goal_width <= 8
            and 0.5 <= goal_height <= 3
            and config.release_height_m > radius
        ):
            raise ValueError("bounded training pitch geometry required")
        self.config = config
        self.geometry = dict(
            left_x=left_x,
            right_x=right_x,
            radius=radius,
            goal_width=goal_width,
            goal_height=goal_height,
        )
        self.return_count = 0
        self._phase = "live"
        self._last_time = -1.0
        self._pending_time = 0.0
        self._exit_position = (0.0, 0.0, 0.0)
        self._exit_reason = ""
        self._release_time = 0.0

    def observe(
        self,
        *,
        time_sec: float,
        ball_position_m: tuple[float, float, float],
        player_positions_xy_m: tuple[tuple[float, float], ...],
    ) -> TrainingBallReturnEvent | None:
        if (
            type(time_sec) not in (int, float)
            or not math.isfinite(time_sec)
            or time_sec < 0
            or time_sec <= self._last_time
            or len(ball_position_m) != 3
            or not player_positions_xy_m
            or any(len(p) != 2 for p in player_positions_xy_m)
        ):
            raise ValueError("monotonic measured clock and explicit player positions required")
        values = (*ball_position_m, *(v for p in player_positions_xy_m for v in p))
        if any(
            type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 10000 for v in values
        ):
            raise ValueError("finite bounded positions required")
        reason = ball_exit_reason(ball_position_m, **self.geometry)
        self._last_time = time_sec
        x, y, _ = ball_position_m
        inside = (
            self.geometry["left_x"] + self.geometry["radius"]
            <= x
            <= self.geometry["right_x"] - self.geometry["radius"]
            and abs(y) <= 3.0 - self.geometry["radius"]
        )
        if self._phase == "released":
            if inside:
                self._phase = "live"
                return TrainingBallReturnEvent(3, time_sec, self.return_count, self._exit_reason)
            if time_sec - self._release_time < 3.0:
                return None
            self._phase = "live"  # Failed outside return can consume another bounded attempt.
        if self._phase == "live":
            if reason is None:
                return None
            self._phase = "pending"
            self._pending_time, self._exit_position, self._exit_reason = (
                time_sec,
                ball_position_m,
                reason,
            )
            return TrainingBallReturnEvent(1, time_sec, self.return_count, reason)
        if (
            self.return_count >= self.config.maximum_returns
            or time_sec - self._pending_time < self.config.delay_sec
        ):
            return None
        origin_x = min(
            self.geometry["right_x"] - 0.75,
            max(self.geometry["left_x"] + 0.75, self._exit_position[0]),
        )
        side = 1.0 if self._exit_position[1] >= 0 else -1.0
        # Same-side candidates first; no role/seed/future trajectory selection.
        for lateral_side in (side, -side):
            for dx in (0.0, 1.0, -1.0, 2.0, -2.0):
                px = min(
                    self.geometry["right_x"] - 0.75,
                    max(self.geometry["left_x"] + 0.75, origin_x + dx),
                )
                py = lateral_side * (3.0 + self.geometry["radius"] + 0.25)
                if any(
                    math.hypot(px - p[0], py - p[1]) < self.config.player_clearance_m
                    for p in player_positions_xy_m
                ):
                    continue
                self.return_count += 1
                self._phase, self._release_time = "released", time_sec
                return TrainingBallReturnEvent(
                    2,
                    time_sec,
                    self.return_count,
                    self._exit_reason,
                    (px, py, self.config.release_height_m),
                    (
                        0.0,
                        -lateral_side * self.config.inward_speed_mps,
                        self.config.upward_speed_mps,
                    ),
                )
        return None  # No clear release location: do not spawn inside a player.
