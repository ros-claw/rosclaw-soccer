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
    launch_target_xy: tuple[float, float] | None = None

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
                self.launch_target_xy is not None
                and (
                    type(self.launch_target_xy) is not tuple
                    or len(self.launch_target_xy) != 2
                    or any(
                        type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1000
                        for x in self.launch_target_xy
                    )
                )
            )
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

    def withdraw_unlaunched(self, *, source_still_committed: bool) -> PassHandoff:
        """Withdraw an unlaunched promise when its source changes the option.

        A physical launch survives later tactical changes: the receiver must
        still track the ball. This only interrupts task evidence, never motion
        authority. Motor readiness alone is not a withdrawal of intention.
        """
        if type(source_still_committed) is not bool:
            raise ValueError("source commitment must be an observed boolean predicate")
        if not source_still_committed and self.source_foot_contact_sec is None:
            return replace(self, interrupted=True)
        return self

    def observe_contact(
        self, *, agent_id: str, foot: bool, force_n: float, time_sec: float
    ) -> PassHandoff:
        if not math.isfinite(force_n) or force_n < 0 or type(foot) is not bool:
            raise ValueError("handoff contact must be finite positive physical evidence")
        if self.expired(time_sec) or force_n <= 1e-6:
            return self
        if agent_id.split(".")[0] != self.source.split(".")[0]:
            return replace(self, interrupted=True)
        if (
            agent_id == self.source
            and foot
            and self.source_foot_contact_sec is None
            and self.launch_target_xy is None
        ):
            return replace(self, source_foot_contact_sec=time_sec)
        return self

    def observe_directed_launch(
        self,
        *,
        agent_id: str,
        foot: bool,
        force_n: float,
        time_sec: float,
        ball_xy: tuple[float, float],
        ball_velocity_xy: tuple[float, float],
    ) -> PassHandoff:
        """Start flight on a physical source-foot outlet, not a preparation tap.

        The destination is immutable from handshake creation. No repeated touch
        renews an existing flight window; no proximity-only launch is possible.
        This event does not confirm reception or authorize any motion.
        """
        if self.launch_target_xy is None:
            raise ValueError("directed launch requires a handshake-bound target")
        if any(
            type(v) is not tuple
            or len(v) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1000 for x in v)
            for v in (ball_xy, ball_velocity_xy)
        ):
            raise ValueError("finite measured planar ball state required")
        state = self.observe_contact(
            agent_id=agent_id, foot=foot, force_n=force_n, time_sec=time_sec
        )
        if (
            state.expired(time_sec)
            or agent_id != self.source
            or not foot
            or force_n <= 1.0
            or state.source_foot_contact_sec is not None
        ):
            return state
        dx = self.launch_target_xy[0] - ball_xy[0]
        dy = self.launch_target_xy[1] - ball_xy[1]
        distance = math.hypot(dx, dy)
        if distance > 1e-6 and (
            ball_velocity_xy[0] * (dx / distance) + ball_velocity_xy[1] * (dy / distance) >= 0.4
        ):
            return replace(state, source_foot_contact_sec=time_sec)
        return state

    def can_activate(self, time_sec: float, *, progressed: bool) -> bool:
        if type(progressed) is not bool:
            raise ValueError("handoff progress must be a measured boolean predicate")
        return bool(
            not self.expired(time_sec)
            and self.source_foot_contact_sec is not None
            and time_sec >= self.source_foot_contact_sec
            and progressed
        )

    def can_track_incoming_ball(
        self,
        time_sec: float,
        *,
        ball_xy: tuple[float, float],
        ball_velocity_xy: tuple[float, float],
        receiver_xy: tuple[float, float],
    ) -> bool:
        """Causal anticipation only: source foot contact and measured approach.

        This does not activate a lease, transfer possession or confirm reception.
        It can precede expiry of the source's remembered contact-owner label.
        """
        vectors = (ball_xy, ball_velocity_xy, receiver_xy)
        if any(
            type(v) is not tuple
            or len(v) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
            for v in vectors
        ):
            raise ValueError("finite planar flight observations required")
        if (
            self.expired(time_sec)
            or self.source_foot_contact_sec is None
            or time_sec < self.source_foot_contact_sec
        ):
            return False
        dx, dy = receiver_xy[0] - ball_xy[0], receiver_xy[1] - ball_xy[1]
        distance = math.hypot(dx, dy)
        return distance > 1e-6 and (
            (ball_velocity_xy[0] * dx + ball_velocity_xy[1] * dy) / distance > 0.10
        )
