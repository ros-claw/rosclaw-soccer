"""Explicit simulation-only end of A0 residual authority at foundation entry."""

import re
from dataclasses import asdict, dataclass, field
from typing import Literal

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


@dataclass(frozen=True)
class ReceivingFoundationHandoff:
    agent_id: str
    entry_frame: int
    schedule_hash: str
    motor_contract_hash: str
    activation_ceiling: Literal["SIM_ONLY"] = field(default="SIM_ONLY", init=False)

    def __post_init__(self) -> None:
        if (
            type(self.agent_id) is not str
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", self.agent_id) is None
            or type(self.entry_frame) is not int
            or not 1 <= self.entry_frame < 1000
            or self.activation_ceiling != "SIM_ONLY"
            or any(
                type(v) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", v) is None
                for v in (self.schedule_hash, self.motor_contract_hash)
            )
        ):
            raise ValueError("content-bound measured foundation handoff required")

    @property
    def contract_hash(self) -> str:
        return str(hash_json(asdict(self)))

    def validate_binding(
        self,
        schedule: ReceivingOracleSchedule,
        *,
        motor_agent_id: str,
        motor_contract_hash: str,
        motor_entry_frame: int,
        idle_residual_fallback: bool,
    ) -> None:
        self.__post_init__()
        if not isinstance(schedule, ReceivingOracleSchedule):
            raise ValueError("typed predecessor schedule required")
        schedule.__post_init__()
        if (
            schedule.substrate != "A0_leg12"
            or schedule.agent_id != self.agent_id
            or schedule.contract_hash != self.schedule_hash
            or schedule.start_frame >= self.entry_frame
            or motor_agent_id != self.agent_id
            or motor_contract_hash != self.motor_contract_hash
            or type(motor_entry_frame) is not int
            or motor_entry_frame != self.entry_frame
            or idle_residual_fallback is not True
        ):
            raise ValueError("handoff differs from predecessor, successor or entry boundary")

    def predecessor_retired(self, frame: int) -> bool:
        """Scheduled retirement never depends on successor success or a caller permit."""
        if type(frame) is not int or not 0 <= frame < 1000:
            raise ValueError("bounded measured control frame required")
        return frame >= self.entry_frame
