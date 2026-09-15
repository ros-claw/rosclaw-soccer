"""Explicit fresh-lease renewal of a normally retired simulation motor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rosclaw_soccer.skills.team.motor_option import TeamMotorOption, TeamReceiveCommitment
from rosclaw_soccer.skills.team.motor_retirement import TeamMotorRetirement


@dataclass(frozen=True)
class TeamMotorRearmContext:
    agent_id: str
    frame: int
    time_sec: float
    retirement_frame: int
    contract_hash: str
    native_motor_active: bool
    commitment: TeamReceiveCommitment

    def __post_init__(self) -> None:
        TeamMotorRetirement(self.agent_id, self.frame, self.time_sec, self.contract_hash)
        if (
            type(self.retirement_frame) is not int
            or not 0 <= self.retirement_frame < self.frame
            or type(self.native_motor_active) is not bool
            or not isinstance(self.commitment, TeamReceiveCommitment)
        ):
            raise ValueError("explicit retired motor and typed fresh commitment required")
        self.commitment.__post_init__()
        if (
            self.commitment.receiver_agent_id != self.agent_id
            or self.commitment.accepted_frame > self.frame
            or self.commitment.accepted_time_sec > self.time_sec
        ):
            raise ValueError("foreign or future receive commitment")

    @property
    def eligible(self) -> bool:
        self.__post_init__()
        return (
            not self.native_motor_active and self.commitment.accepted_frame > self.retirement_frame
        )


@runtime_checkable
class TeamMotorRearmProvider(Protocol):
    def successor(self, context: TeamMotorRearmContext) -> TeamMotorOption | None: ...


def validate_motor_successor(
    context: TeamMotorRearmContext, *, predecessor: TeamMotorOption, successor: TeamMotorOption
) -> None:
    if (
        not isinstance(context, TeamMotorRearmContext)
        or not context.eligible
        or predecessor is successor
        or predecessor.contract_hash != context.contract_hash
        or successor.contract_hash != context.contract_hash
        or not callable(getattr(successor, "propose", None))
    ):
        raise ValueError("fresh, idle, same-contract motor successor required")
