"""Explicit removal-only contact-teacher ablation for simulation comparisons.

This cannot grant motor ownership, inject torque, disable safety projection,
or permit a contact teacher to compete with a registered whole-body motor.
"""

import re
from dataclasses import asdict, dataclass

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class ContactTeacherSuppression:
    agent_id: str
    start_frame: int

    def __post_init__(self) -> None:
        if (
            type(self.agent_id) is not str
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", self.agent_id) is None
            or type(self.start_frame) is not int
            or not 0 <= self.start_frame <= 1000
        ):
            raise ValueError("one explicit player and bounded suppression boundary required")

    @property
    def contract_hash(self) -> str:
        return str(hash_json({"schema": "soccer.contact_teacher_suppression.v1", **asdict(self)}))

    def suppressed_agent(self, frame: int) -> str | None:
        if type(frame) is not int or frame < 0:
            raise ValueError("nonnegative control frame required")
        return self.agent_id if frame >= self.start_frame else None
