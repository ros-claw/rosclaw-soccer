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

    def approach_waypoint(
        self,
        current: tuple[float, float],
        ball: tuple[float, float],
        destination: tuple[float, float],
        *,
        lateral_clearance_m: float,
    ) -> tuple[float, float]:
        """Stage a wrong-side approach around the ball, not through it.

        This is a position objective, not collision-free physical evidence.
        The world still applies velocity, player-clearance and actuator guards.
        No waypoint changes the ball or grants a contact/possession lease.
        """
        if (
            any(len(v) != 2 for v in (current, ball, destination))
            or not all(math.isfinite(v) for v in (*current, *ball, *destination))
            or type(lateral_clearance_m) not in (int, float)
            or not math.isfinite(lateral_clearance_m)
            or not 0.35 <= lateral_clearance_m <= 0.90
        ):
            raise ValueError("finite planar contact approach and bounded clearance required")
        stance, yaw = self.stance(ball, destination)
        dx, dy = math.cos(yaw), math.sin(yaw)
        lx, ly = -dy, dx
        bx, by = ball[0] - current[0], ball[1] - current[1]
        depth = bx * dx + by * dy
        lateral = bx * lx + by * ly
        if depth < self.depth_m * 0.5:
            if abs(lateral) < lateral_clearance_m:
                side = -1.0 if lateral >= -1e-10 else 1.0
                return (
                    current[0] + side * lateral_clearance_m * lx,
                    current[1] + side * lateral_clearance_m * ly,
                )
            return (
                ball[0] - (self.depth_m + 0.2) * dx - lateral * lx,
                ball[1] - (self.depth_m + 0.2) * dy - lateral * ly,
            )
        return stance
