"""Private read-only physics consumers, independent of motor ownership.

These slots receive immutable values only. A consumer fault invalidates its
evidence stream; it must neither register a motor nor modify a motor target.
This is an in-process trusted-provider boundary, not a Python security sandbox.
"""

import re
from collections.abc import Callable
from typing import Protocol, cast, runtime_checkable

from rosclaw_soccer.skills.team.motor_option import TeamMotorPhysicsObservation


@runtime_checkable
class PhysicsEvidenceConsumer(Protocol):
    agent_id: str
    contract_hash: str

    def observe_physics(self, observation: TeamMotorPhysicsObservation) -> None: ...


class PhysicsEvidenceSlot:
    """One ordered 500 Hz stream, with immutable registration and fault latch."""

    def __init__(self, consumer: PhysicsEvidenceConsumer) -> None:
        if (
            not isinstance(consumer, PhysicsEvidenceConsumer)
            or not callable(consumer.observe_physics)
            or type(consumer.agent_id) is not str
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", consumer.agent_id) is None
            or type(consumer.contract_hash) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", consumer.contract_hash) is None
        ):
            raise ValueError("private content-bound physics evidence consumer required")
        self.consumer = consumer
        self._agent_id = consumer.agent_id
        self._contract_hash = consumer.contract_hash
        self.faulted = False
        self.fault_reason: str | None = None
        self._time: float | None = None

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def contract_hash(self) -> str:
        return self._contract_hash

    def _check_identity(self) -> None:
        if (
            self.consumer.agent_id != self.agent_id
            or self.consumer.contract_hash != self.contract_hash
        ):
            raise ValueError("physics evidence consumer changed its registration")

    def observe_physics(self, observation: TeamMotorPhysicsObservation) -> None:
        if self.faulted:
            return
        try:
            self._check_identity()
            if not isinstance(observation, TeamMotorPhysicsObservation):
                raise ValueError("typed immutable physics evidence required")
            observation.__post_init__()
            if (
                not observation.contacts_complete
                or observation.observer_agent_id != self.agent_id
                or self._time is None
                and abs(observation.time_sec - 0.002) > 1e-6
                or self._time is not None
                and abs(observation.time_sec - self._time - 0.002) > 1e-6
            ):
                raise ValueError("complete consecutive same-player physics evidence required")
            callback = cast(
                Callable[[TeamMotorPhysicsObservation], object], self.consumer.observe_physics
            )
            if callback(observation) is not None:
                raise ValueError("an evidence consumer cannot return an action proposal")
            self._check_identity()
            self._time = observation.time_sec
        except Exception as exc:
            # Provider bugs invalidate evidence, not unrelated motor commands.
            self.faulted = True
            self.fault_reason = type(exc).__name__
