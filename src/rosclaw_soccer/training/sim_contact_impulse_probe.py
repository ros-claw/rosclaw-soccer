"""Bounded subframe torque intervention for simulation research only."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SimContactImpulseProbe:
    """Immutable predeclared action (at most 20ms), not a learned policy."""

    agent_id: str
    frame: int
    joint: str
    delta_nm: float
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    duration_substeps: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.agent_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", self.agent_id) is None
            or type(self.frame) is not int
            or not 0 <= self.frame < 500
            or self.joint
            not in (
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            )
            or type(self.delta_nm) is not float
            or self.delta_nm not in (-5.0, 5.0)
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized is not False
            or type(self.duration_substeps) is not int
            or self.duration_substeps not in (1, 5, 10)
        ):
            raise ValueError("invalid bounded SIM-only contact impulse probe")
