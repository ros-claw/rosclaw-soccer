"""Explicit experimental navigation limits, never robot motion authorization."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationNavigationEnvelope:
    """Bind a larger simulation command range to one private motor contract.

    This is an experiment declaration, not qualification, a safety guarantee,
    a hardware permit, or permission to rescale post-clearance commands.
    """

    agent_id: str
    motor_contract_hash: str
    maximum_speed_mps: float
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if (
            type(self.agent_id) is not str
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", self.agent_id) is None
            or type(self.motor_contract_hash) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.motor_contract_hash) is None
            or type(self.maximum_speed_mps) not in (float, int)
            or not math.isfinite(self.maximum_speed_mps)
            or not 0.7 < self.maximum_speed_mps <= 1.5
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("player-bound SIM_ONLY experimental navigation envelope required")
