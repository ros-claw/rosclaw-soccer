"""Causal bridge from an independent tactical commitment to one physical option.

The bridge is deliberately small: it grants no pose, joint, torque, ROS, or
hardware authority.  It only proves that the agent, observation, Champion
policy, team coordination frame, option policy, target, and execution window
belong to the same immutable request.  A physics runner must later close that
request with measured contacts and safety state.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rosclaw_soccer.growth.independent_agent_cell import (
    AgentCellDecision,
    RosclawSoccerAgentCell,
)
from rosclaw_soccer.growth.role_self_model import (
    TacticalIntent,
    TeamCoordinationFrame,
)
from rosclaw_soccer.sim.contracts import hash_json

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")


class PhysicalSoccerOption(StrEnum):
    PASS = "pass"
    SHOOT = "shoot"
    SAVE = "save"
    DISTRIBUTE = "distribute"


class PhysicalOptionTerminal(StrEnum):
    COMPLETE = "complete"
    ABORTED = "aborted"


_OPTION_BY_INTENT = {
    TacticalIntent.PASS: PhysicalSoccerOption.PASS,
    TacticalIntent.SHOOT: PhysicalSoccerOption.SHOOT,
    TacticalIntent.SAVE: PhysicalSoccerOption.SAVE,
    TacticalIntent.DISTRIBUTE: PhysicalSoccerOption.DISTRIBUTE,
}


@dataclass(frozen=True)
class PhysicalOptionRequest:
    request_id: str
    agent_id: str
    option: PhysicalSoccerOption
    tactical_intent: TacticalIntent
    target_position_m: tuple[float, float, float]
    target_agent_id: str | None
    agent_cell_hash: str
    agent_policy_hash: str
    decision_hash: str
    coordination_frame_hash: str
    option_policy_hash: str
    ball_state_hash: str
    requested_at_sec: float
    expires_at_sec: float
    pass_handshake_hash: str | None = None
    exclusive_ball_lease: bool = True
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    schema_version: str = "rosclaw_soccer.physical_option_request.v1"

    def __post_init__(self) -> None:
        hashes = (
            self.agent_cell_hash,
            self.agent_policy_hash,
            self.decision_hash,
            self.coordination_frame_hash,
            self.option_policy_hash,
            self.ball_state_hash,
        )
        if (
            not _IDENTIFIER.fullmatch(self.request_id)
            or not _IDENTIFIER.fullmatch(self.agent_id)
            or not isinstance(self.option, PhysicalSoccerOption)
            or not isinstance(self.tactical_intent, TacticalIntent)
            or _OPTION_BY_INTENT.get(self.tactical_intent) is not self.option
            or len(self.target_position_m) != 3
            or any(not math.isfinite(value) for value in self.target_position_m)
            or (
                self.target_agent_id is not None and not _IDENTIFIER.fullmatch(self.target_agent_id)
            )
            or any(not _HASH.fullmatch(value) for value in hashes)
            or not math.isfinite(self.requested_at_sec)
            or not math.isfinite(self.expires_at_sec)
            or self.requested_at_sec < 0.0
            or self.expires_at_sec <= self.requested_at_sec
            or not isinstance(self.exclusive_ball_lease, bool)
            or not self.exclusive_ball_lease
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.direct_joint_torque_output
        ):
            raise ValueError("physical option request contract is invalid")
        if self.option is PhysicalSoccerOption.PASS:
            if (
                self.target_agent_id is None
                or self.pass_handshake_hash is None
                or not _HASH.fullmatch(self.pass_handshake_hash)
            ):
                raise ValueError("physical PASS requires a content-bound receiver handshake")
        elif self.pass_handshake_hash is not None:
            raise ValueError("non-PASS option cannot carry a pass handshake")

    @property
    def request_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["option"] = self.option.value
        value["tactical_intent"] = self.tactical_intent.value
        return value


@dataclass(frozen=True)
class PhysicalOptionOutcome:
    request_hash: str
    terminal: PhysicalOptionTerminal
    contact_observed: bool
    contact_time_sec: float | None
    contact_link: str | None
    pre_contact_ball_speed_mps: float
    post_contact_peak_ball_speed_mps: float
    target_delivery_distance_m: float | None
    target_agent_contact_observed: bool
    goal_crossed: bool
    goalkeeper_contact_observed: bool
    finite_state: bool
    minimum_pelvis_height_m: float
    maximum_tilt_rad: float
    required_minimum_pelvis_height_m: float
    allowed_maximum_tilt_rad: float
    joint_limit_violation: bool
    torque_limit_violation: bool
    robot_robot_contact_count: int
    option_started_from_current_commitment: bool
    recovery_handoff_completed: bool
    root_pose_write_after_start: bool = False
    ball_state_write_after_start: bool = False
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    pixels_used_for_scoring: bool = False
    schema_version: str = "rosclaw_soccer.physical_option_outcome.v1"

    def __post_init__(self) -> None:
        values = (
            self.pre_contact_ball_speed_mps,
            self.post_contact_peak_ball_speed_mps,
            self.minimum_pelvis_height_m,
            self.maximum_tilt_rad,
            self.required_minimum_pelvis_height_m,
            self.allowed_maximum_tilt_rad,
        )
        flags = (
            self.contact_observed,
            self.target_agent_contact_observed,
            self.goal_crossed,
            self.goalkeeper_contact_observed,
            self.finite_state,
            self.joint_limit_violation,
            self.torque_limit_violation,
            self.option_started_from_current_commitment,
            self.recovery_handoff_completed,
            self.root_pose_write_after_start,
            self.ball_state_write_after_start,
            self.hardware_command_sent,
            self.pixels_used_for_scoring,
        )
        if (
            not _HASH.fullmatch(self.request_hash)
            or not isinstance(self.terminal, PhysicalOptionTerminal)
            or any(not isinstance(value, bool) for value in flags)
            or any(not math.isfinite(value) or value < 0.0 for value in values)
            or (
                self.contact_time_sec is not None
                and (not math.isfinite(self.contact_time_sec) or self.contact_time_sec < 0.0)
            )
            or (self.contact_observed != (self.contact_time_sec is not None))
            or (self.contact_observed != (self.contact_link is not None))
            or (
                self.contact_link is not None
                and self.contact_link
                not in {"left_foot", "right_foot", "left_glove", "right_glove"}
            )
            or (
                self.target_delivery_distance_m is not None
                and (
                    not math.isfinite(self.target_delivery_distance_m)
                    or self.target_delivery_distance_m < 0.0
                )
            )
            or isinstance(self.robot_robot_contact_count, bool)
            or self.robot_robot_contact_count < 0
            or self.activation_ceiling != "SIM_ONLY"
            or self.required_minimum_pelvis_height_m <= 0.0
            or self.allowed_maximum_tilt_rad <= 0.0
        ):
            raise ValueError("physical option outcome contract is invalid")

    @property
    def safe(self) -> bool:
        return bool(
            self.finite_state
            and self.minimum_pelvis_height_m >= self.required_minimum_pelvis_height_m
            and self.maximum_tilt_rad <= self.allowed_maximum_tilt_rad
            and not self.joint_limit_violation
            and not self.torque_limit_violation
            and self.robot_robot_contact_count == 0
            and not self.root_pose_write_after_start
            and not self.ball_state_write_after_start
            and not self.hardware_command_sent
            and not self.pixels_used_for_scoring
        )

    @property
    def physical_contact_success(self) -> bool:
        return bool(
            self.terminal is PhysicalOptionTerminal.COMPLETE
            and self.contact_observed
            and self.post_contact_peak_ball_speed_mps
            >= max(0.50, self.pre_contact_ball_speed_mps + 0.35)
            and self.option_started_from_current_commitment
            and self.recovery_handoff_completed
            and self.safe
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["terminal"] = self.terminal.value
        value["safe"] = self.safe
        value["physical_contact_success"] = self.physical_contact_success
        return value


def build_physical_option_request(
    *,
    cell: RosclawSoccerAgentCell,
    decision: AgentCellDecision,
    coordination: TeamCoordinationFrame,
    option_policy_hash: str,
    timeout_sec: float,
) -> PhysicalOptionRequest:
    """Bind one current independent decision to a SIM-only physical option."""

    if not math.isfinite(timeout_sec) or not 0.5 <= timeout_sec <= 10.0:
        raise ValueError("physical option timeout is invalid")
    try:
        commitment = next(
            value for value in coordination.intents if value.agent_id == cell.agent_id
        )
        observation = next(
            value for value in coordination.observations if value.observer_agent_id == cell.agent_id
        )
    except StopIteration as exc:
        raise ValueError("physical option agent is absent from the coordination frame") from exc
    option = _OPTION_BY_INTENT.get(decision.intent)
    if option is None:
        raise ValueError("tactical intent has no physical option route")
    if (
        decision.agent_id != cell.agent_id
        or decision.policy_artifact_hash != cell.self_model.policy_artifact_hash
        or commitment.policy_artifact_hash != cell.self_model.policy_artifact_hash
        or commitment.intent is not decision.intent
        or commitment.skill is not decision.skill
        or commitment.target_agent_id != decision.target_agent_id
        or not math.isclose(commitment.confidence, decision.confidence, abs_tol=1.0e-12)
        or coordination.roster.agent(cell.agent_id).self_model_hash
        != cell.self_model.self_model_hash
    ):
        raise ValueError("physical option decision is not the current agent commitment")
    handshake_hash: str | None = None
    if option is PhysicalSoccerOption.PASS:
        matches = tuple(
            value
            for value in coordination.pass_receive_handshakes
            if value.passer_agent_id == cell.agent_id
            and value.receiver_agent_id == decision.target_agent_id
        )
        if len(matches) != 1 or not matches[0].accepted_by_receiver:
            raise ValueError("physical PASS lacks exactly one accepted handshake")
        handshake_hash = matches[0].handshake_hash
    return PhysicalOptionRequest(
        request_id=(
            f"option.{option.value}.{cell.agent_id}."
            f"{coordination.frame_hash.removeprefix('sha256:')[:12]}"
        ),
        agent_id=cell.agent_id,
        option=option,
        tactical_intent=decision.intent,
        target_position_m=decision.target_position_m,
        target_agent_id=decision.target_agent_id,
        agent_cell_hash=cell.cell_hash,
        agent_policy_hash=cell.self_model.policy_artifact_hash,
        decision_hash=decision.decision_hash,
        coordination_frame_hash=coordination.frame_hash,
        option_policy_hash=option_policy_hash,
        ball_state_hash=observation.ball_state_hash,
        requested_at_sec=observation.time_sec,
        expires_at_sec=observation.time_sec + timeout_sec,
        pass_handshake_hash=handshake_hash,
    )


__all__ = [
    "PhysicalOptionOutcome",
    "PhysicalOptionRequest",
    "PhysicalOptionTerminal",
    "PhysicalSoccerOption",
    "build_physical_option_request",
]
