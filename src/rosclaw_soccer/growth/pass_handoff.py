"""Bounded evidence state for a football pass-to-receive task handoff.

A promise to pass, shin contact, or unrelated ball movement is not evidence
that the promised foot pass was launched. This object grants no motion access.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PassHandoff:
    source: str
    receiver: str
    created_sec: float
    source_foot_contact_sec: float | None = None
    interrupted: bool = False
    lifetime_sec: float = 3.0
    flight_window_sec: float = 3.0

    def __post_init__(self) -> None:
        if (
            not all(
                re.fullmatch(r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*", a)
                for a in (self.source, self.receiver)
            )
            or self.source == self.receiver
            or self.source.split(".")[0] != self.receiver.split(".")[0]
            or not math.isfinite(self.created_sec)
            or self.created_sec < 0
            or not 0.5 <= self.lifetime_sec <= 4.0
            or not 0.5 <= self.flight_window_sec <= 4.0
            or type(self.interrupted) is not bool
            or (
                self.source_foot_contact_sec is not None
                and (
                    not math.isfinite(self.source_foot_contact_sec)
                    or self.source_foot_contact_sec < self.created_sec
                    or self.source_foot_contact_sec >= self.created_sec + self.lifetime_sec
                )
            )
        ):
            raise ValueError("invalid bounded pass handoff")

    def expired(self, time_sec: float, *, receiver_stable: bool = True) -> bool:
        if not math.isfinite(time_sec) or time_sec < self.created_sec:
            raise ValueError("handoff time must be finite and monotonic from creation")
        return (
            self.interrupted
            or not receiver_stable
            or time_sec
            >= (
                self.created_sec + self.lifetime_sec
                if self.source_foot_contact_sec is None
                else self.source_foot_contact_sec + self.flight_window_sec
            )
        )

    def observe_contact(
        self, *, agent_id: str, foot: bool, force_n: float, time_sec: float
    ) -> PassHandoff:
        if not math.isfinite(force_n) or force_n < 0 or type(foot) is not bool:
            raise ValueError("handoff contact must be finite positive physical evidence")
        if self.expired(time_sec) or force_n <= 1e-6:
            return self
        if agent_id.split(".")[0] != self.source.split(".")[0]:
            return replace(self, interrupted=True)
        if agent_id == self.source and foot and self.source_foot_contact_sec is None:
            return replace(self, source_foot_contact_sec=time_sec)
        return self

    def can_activate(self, time_sec: float, *, progressed: bool) -> bool:
        if type(progressed) is not bool:
            raise ValueError("handoff progress must be a measured boolean predicate")
        return bool(
            not self.expired(time_sec)
            and self.source_foot_contact_sec is not None
            and time_sec >= self.source_foot_contact_sec
            and progressed
        )
