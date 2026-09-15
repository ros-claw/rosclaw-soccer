"""One-way release of SIM motor ownership, not a success or promotion claim."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class TeamMotorRetirement:
    agent_id: str
    frame: int
    time_sec: float
    contract_hash: str
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.agent_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", self.agent_id) is None
            or type(self.frame) is not int
            or self.frame < 0
            or type(self.time_sec) not in (int, float)
            or not math.isfinite(self.time_sec)
            or self.time_sec < 0
            or not isinstance(self.contract_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.contract_hash) is None
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("same-frame content-bound SIM motor retirement required")


@runtime_checkable
class TeamMotorRetirementProvider(Protocol):
    def retirement_request(self, *, frame: int, time_sec: float) -> TeamMotorRetirement | None: ...


def validate_motor_retirement(
    request: TeamMotorRetirement,
    *,
    agent_id: str,
    frame: int,
    time_sec: float,
    contract_hash: str,
    proposed_target: bool,
) -> None:
    if not isinstance(request, TeamMotorRetirement):
        raise ValueError("typed retirement request required")
    # Reconstruct values so frozen-dataclass tampering cannot bypass validation.
    checked = TeamMotorRetirement(
        request.agent_id,
        request.frame,
        request.time_sec,
        request.contract_hash,
        request.activation_ceiling,
    )
    if (
        type(proposed_target) is not bool
        or proposed_target
        or type(frame) is not int
        or frame < 0
        or checked.agent_id != agent_id
        or checked.frame != frame
        or type(time_sec) not in (int, float)
        or not math.isfinite(time_sec)
        or abs(checked.time_sec - time_sec) > 1e-9
        or checked.contract_hash != contract_hash
    ):
        raise ValueError("a proposing, stale or foreign motor cannot retire")
