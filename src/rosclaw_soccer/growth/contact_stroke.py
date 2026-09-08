"""Finite, smooth ankle stroke; an action clock, not a learned policy."""

from __future__ import annotations

import math
from dataclasses import dataclass


def stroke_blend(progress: float) -> tuple[float, float]:
    """Quintic position blend and derivative on a finite action interval."""
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError("stroke progress must be finite and inside [0, 1]")
    return (
        10 * progress**3 - 15 * progress**4 + 6 * progress**5,
        30 * progress**2 * (1 - progress) ** 2,
    )


@dataclass
class ContactStroke:
    """Lock one foot for an attempt; completion never automatically rearms."""

    start_sec: float | None = None
    last_sec: float | None = None
    use_left: bool = False
    direction_xy: tuple[float, float] = (1.0, 0.0)
    lateral_sign: float = 1.0

    def reset(self) -> None:
        self.start_sec = None
        self.last_sec = None

    def step(
        self,
        *,
        time_sec: float,
        duration_sec: float,
        use_left: bool,
        direction_xy: tuple[float, float] = (1.0, 0.0),
        lateral_sign: float = 1.0,
    ) -> float:
        if (
            not math.isfinite(time_sec)
            or time_sec < 0
            or not math.isfinite(duration_sec)
            or not 0.12 <= duration_sec <= 0.60
            or len(direction_xy) != 2
            or not all(math.isfinite(x) for x in direction_xy)
            or math.hypot(*direction_xy) <= 1e-9
            or lateral_sign not in (-1.0, 1.0)
        ):
            raise ValueError("invalid finite stroke clock")
        if self.last_sec is not None and time_sec < self.last_sec:
            raise ValueError("stroke clock moved backwards")
        self.last_sec = time_sec
        if self.start_sec is None:
            self.start_sec = time_sec
            self.use_left = use_left
            norm = math.hypot(*direction_xy)
            self.direction_xy = (direction_xy[0] / norm, direction_xy[1] / norm)
            self.lateral_sign = lateral_sign
        if time_sec < self.start_sec:
            raise ValueError("stroke clock moved backwards")
        return min(1.0, (time_sec - self.start_sec) / duration_sec)
