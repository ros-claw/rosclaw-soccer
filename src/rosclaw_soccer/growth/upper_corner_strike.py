"""Lane-conditioned, bounded contact policy for upper-corner strikes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from rosclaw_soccer.growth.ballistic_contact_torque_residual import (
    G1BallisticContactTorqueResidualConfig,
)
from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class UpperCornerLaneAction:
    lane_id: str
    foot_yaw_offset_rad: float
    schema_version: str = "rosclaw_soccer.upper_corner_lane_action.v1"

    def __post_init__(self) -> None:
        if self.lane_id not in {"left-post", "right-post"}:
            raise ValueError("upper-corner action has an unknown regulation lane")
        if (
            not math.isfinite(self.foot_yaw_offset_rad)
            or not -0.12 <= self.foot_yaw_offset_rad <= 0.12
        ):
            raise ValueError("upper-corner foot-yaw action is outside the bounded adapter")


@dataclass(frozen=True)
class UpperCornerStrikePolicy:
    """One common torque muscle memory plus a small lane-conditioned aim head."""

    actions: tuple[UpperCornerLaneAction, ...] = (
        UpperCornerLaneAction("left-post", 0.0050),
        UpperCornerLaneAction("right-post", -0.0162),
    )
    right_leg_residual_nm: tuple[float, ...] = (5.0, 0.0, 0.0, 0.0, -4.0, 5.0)
    counterbalance_residual_nm: tuple[float, ...] = (3.6, 0.0, 0.0, 5.0, 0.0, 0.0)
    contact_policy_frame: int = 256
    lead_duration_sec: float = 0.08
    trail_duration_sec: float = 0.065
    maximum_joint_residual_nm: float = 12.0
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.upper_corner_strike_policy.v1"

    def __post_init__(self) -> None:
        if len(self.actions) != 2 or {item.lane_id for item in self.actions} != {
            "left-post",
            "right-post",
        }:
            raise ValueError("upper-corner policy requires both regulation lanes")
        # Reuse the runtime contract as the single source of truth for torque bounds.
        self.torque_config()
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("upper-corner policy must remain SIM_ONLY")

    @property
    def artifact_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def action(self, lane_id: str) -> UpperCornerLaneAction:
        try:
            return next(item for item in self.actions if item.lane_id == lane_id)
        except StopIteration as exc:
            raise ValueError("upper-corner lane is not represented by this policy") from exc

    def torque_config(self) -> G1BallisticContactTorqueResidualConfig:
        return G1BallisticContactTorqueResidualConfig(
            right_leg_residual_nm=self.right_leg_residual_nm,
            counterbalance_residual_nm=self.counterbalance_residual_nm,
            contact_policy_frame=self.contact_policy_frame,
            lead_duration_sec=self.lead_duration_sec,
            trail_duration_sec=self.trail_duration_sec,
            maximum_joint_residual_nm=self.maximum_joint_residual_nm,
        )


__all__ = ["UpperCornerLaneAction", "UpperCornerStrikePolicy"]
