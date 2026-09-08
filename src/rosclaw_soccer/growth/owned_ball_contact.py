"""Bounded, learnable stance parameters; no authority over poses or torques."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class OwnedBallContactPolicy:
    depth_m: float = 0.24
    lateral_m: float = 0.12
    pass_speed_mps: float = 1.5

    def __post_init__(self) -> None:
        if (
            not all(math.isfinite(v) for v in (self.depth_m, self.lateral_m, self.pass_speed_mps))
            or not 0.18 <= self.depth_m <= 0.40
            or not -0.18 <= self.lateral_m <= 0.18
            or not 0.5 <= self.pass_speed_mps <= 2.5
        ):
            raise ValueError("owned contact policy exceeds the bounded SIM search space")

    @property
    def policy_hash(self) -> str:
        return str(hash_json(asdict(self)))

    def stance(
        self, ball: tuple[float, float], destination: tuple[float, float]
    ) -> tuple[tuple[float, float], float]:
        if not all(math.isfinite(v) for v in (*ball, *destination)):
            raise ValueError("contact observations must be finite")
        dx, dy = destination[0] - ball[0], destination[1] - ball[1]
        distance = math.hypot(dx, dy)
        if distance < 1.0e-6:
            raise ValueError("pass destination must differ from ball")
        dx, dy = dx / distance, dy / distance
        return (
            (
                ball[0] - self.depth_m * dx - self.lateral_m * dy,
                ball[1] - self.depth_m * dy + self.lateral_m * dx,
            ),
            math.atan2(dy, dx),
        )
