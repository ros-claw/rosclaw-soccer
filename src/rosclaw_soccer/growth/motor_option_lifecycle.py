"""SIM motor-option rearming proposals, independent of policy or physics writes.

An observed ball departure permits a new encounter after recovery. A missed
option can retry at most twice without departure. Readiness never bypasses
the world's possession, task, stance, yaw or exclusive-output admission.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MotorOptionLifecycle:
    activation_frame: int | None = None
    completion_frame: int | None = None
    departed: bool = False
    consecutive_misses: int = 0
    activation_count: int = 0

    def start(self, frame: int) -> None:
        if (
            type(frame) is not int
            or frame < 0
            or (self.activation_frame is not None and self.completion_frame is None)
            or (self.completion_frame is not None and frame <= self.completion_frame)
        ):
            raise ValueError("motor lifecycle requires a fresh admitted activation")
        if self.departed:
            self.consecutive_misses = 0
        self.activation_frame = frame
        self.completion_frame = None
        self.departed = False
        self.activation_count += 1

    def finish(self, frame: int, *, contact_observed: bool) -> None:
        if (
            type(frame) is not int
            or type(contact_observed) is not bool
            or self.activation_frame is None
            or self.completion_frame is not None
            or frame < self.activation_frame
        ):
            raise ValueError("motor lifecycle completion must belong to an active option")
        self.completion_frame = frame
        self.consecutive_misses = 0 if contact_observed else self.consecutive_misses + 1

    def ready(
        self, frame: int, *, ball_distance_m: float, body_ready: bool, incoming_receive: bool
    ) -> bool:
        if (
            type(frame) is not int
            or frame < 0
            or not math.isfinite(ball_distance_m)
            or ball_distance_m < 0
            or type(body_ready) is not bool
            or type(incoming_receive) is not bool
        ):
            raise ValueError("finite measured rearm context required")
        if self.completion_frame is None:
            return False
        if frame < self.completion_frame:
            raise ValueError("rearm observation predates completion")
        self.departed |= ball_distance_m >= 1.5
        if not body_ready or incoming_receive or ball_distance_m > 1.2:
            return False
        elapsed = frame - self.completion_frame
        return bool(
            self.departed and elapsed >= 50 or 0 < self.consecutive_misses < 3 and elapsed >= 100
        )
