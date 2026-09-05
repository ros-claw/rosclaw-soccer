"""Fail-closed credit assignment for an independent soccer option chain.

The module is intentionally above the whole-body controllers.  It binds the
PASS -> RECEIVE -> SHOOT -> SAVE sequence to one physical ball lineage and
assigns the *earliest* causal failure to exactly one role-owned ROSClaw cell.
It grants no pose, joint, torque, ROS, or hardware authority.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rosclaw.continual.plasticity_lease import PlasticityLease

from rosclaw_soccer.growth.independent_agent_cell import (
    RosclawSoccerAgentCell,
    build_agent_plasticity_lease,
)
from rosclaw_soccer.growth.role_self_model import MatchRole
from rosclaw_soccer.sim.contracts import hash_json

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")


class ContinuousChainPhase(StrEnum):
    PASS = "pass"
    RECEIVE = "receive"
    SHOOT = "shoot"
    SAVE = "save"


class ContinuousChainFailure(StrEnum):
    BALL_LINEAGE_BROKEN = "ball_lineage_broken"
    NONDETERMINISTIC_REPLAY = "nondeterministic_replay"
    UNSAFE_MOTION = "unsafe_motion"
    PASS_NO_CONTACT = "pass_no_contact"
    PASS_TOO_SLOW = "pass_too_slow"
    PASS_INACCURATE = "pass_inaccurate"
    RECEIVE_NOT_READY = "receive_not_ready"
    RECEIVE_INACCURATE = "receive_inaccurate"
    SHOOT_NO_CONTACT = "shoot_no_contact"
    SHOT_TOO_SLOW = "shot_too_slow"
    SHOT_INACCURATE = "shot_inaccurate"
    SAVE_NO_CONTACT = "save_no_contact"
    SAVE_WEAK_DEFLECTION = "save_weak_deflection"
    CHAIN_INCOMPLETE = "chain_incomplete"


_PHASES = (
    ContinuousChainPhase.PASS,
    ContinuousChainPhase.RECEIVE,
    ContinuousChainPhase.SHOOT,
    ContinuousChainPhase.SAVE,
)
_ROLE_BY_PHASE = {
    ContinuousChainPhase.PASS: MatchRole.PLAYMAKER,
    ContinuousChainPhase.RECEIVE: MatchRole.FINISHER,
    ContinuousChainPhase.SHOOT: MatchRole.FINISHER,
    ContinuousChainPhase.SAVE: MatchRole.GOALKEEPER,
}


@dataclass(frozen=True)
class ChainRoleBinding:
    agent_id: str
    role: MatchRole
    cell_hash: str
    champion_policy_hash: str
    parent_policy_hash: str
    schema_version: str = "rosclaw_soccer.chain_role_binding.v1"

    def __post_init__(self) -> None:
        if (
            not _IDENTIFIER.fullmatch(self.agent_id)
            or self.role not in {MatchRole.PLAYMAKER, MatchRole.FINISHER, MatchRole.GOALKEEPER}
            or any(
                not _HASH.fullmatch(value)
                for value in (
                    self.cell_hash,
                    self.champion_policy_hash,
                    self.parent_policy_hash,
                )
            )
        ):
            raise ValueError("continuous-chain role binding is invalid")

    @property
    def binding_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        return value


@dataclass(frozen=True)
class ContinuousOptionChainRequest:
    chain_id: str
    roster_hash: str
    ball_id: str
    initial_ball_state_hash: str
    role_bindings: tuple[ChainRoleBinding, ...]
    goal_target_m: tuple[float, float, float]
    maximum_pass_error_m: float = 0.25
    maximum_receive_error_m: float = 0.20
    maximum_shot_error_m: float = 0.10
    minimum_pass_speed_mps: float = 0.50
    minimum_shot_speed_mps: float = 4.0
    minimum_save_deflection_mps: float = 1.0
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.continuous_option_chain_request.v1"

    def __post_init__(self) -> None:
        bindings = tuple(self.role_bindings)
        values = (
            *self.goal_target_m,
            self.maximum_pass_error_m,
            self.maximum_receive_error_m,
            self.maximum_shot_error_m,
            self.minimum_pass_speed_mps,
            self.minimum_shot_speed_mps,
            self.minimum_save_deflection_mps,
        )
        roles = {binding.role for binding in bindings}
        if (
            not _IDENTIFIER.fullmatch(self.chain_id)
            or not _IDENTIFIER.fullmatch(self.ball_id)
            or not _HASH.fullmatch(self.roster_hash)
            or not _HASH.fullmatch(self.initial_ball_state_hash)
            or len(bindings) != 3
            or roles != {MatchRole.PLAYMAKER, MatchRole.FINISHER, MatchRole.GOALKEEPER}
            or len({binding.agent_id for binding in bindings}) != 3
            or len(self.goal_target_m) != 3
            or any(not math.isfinite(value) for value in values)
            or not 0.02 <= self.maximum_pass_error_m <= 0.50
            or not 0.02 <= self.maximum_receive_error_m <= 0.50
            or not 0.02 <= self.maximum_shot_error_m <= 0.25
            or not 0.25 <= self.minimum_pass_speed_mps <= 5.0
            or not 1.0 <= self.minimum_shot_speed_mps <= 20.0
            or not 0.25 <= self.minimum_save_deflection_mps <= 10.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("continuous option-chain request is invalid")
        object.__setattr__(self, "role_bindings", bindings)

    def agent_for(self, phase: ContinuousChainPhase) -> str:
        role = _ROLE_BY_PHASE[phase]
        return next(binding.agent_id for binding in self.role_bindings if binding.role is role)

    @property
    def request_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "chain_id": self.chain_id,
            "roster_hash": self.roster_hash,
            "ball_id": self.ball_id,
            "initial_ball_state_hash": self.initial_ball_state_hash,
            "role_bindings": [value.to_dict() for value in self.role_bindings],
            "goal_target_m": list(self.goal_target_m),
            "maximum_pass_error_m": self.maximum_pass_error_m,
            "maximum_receive_error_m": self.maximum_receive_error_m,
            "maximum_shot_error_m": self.maximum_shot_error_m,
            "minimum_pass_speed_mps": self.minimum_pass_speed_mps,
            "minimum_shot_speed_mps": self.minimum_shot_speed_mps,
            "minimum_save_deflection_mps": self.minimum_save_deflection_mps,
            "activation_ceiling": self.activation_ceiling,
            "hardware_authorized": self.hardware_authorized,
        }


@dataclass(frozen=True)
class ContinuousChainEvent:
    phase: ContinuousChainPhase
    agent_id: str
    ball_id: str
    input_ball_state_hash: str
    output_ball_state_hash: str
    decision_hash: str
    option_request_hash: str
    started_at_sec: float
    ended_at_sec: float
    contact_observed: bool
    safe: bool
    phase_ready: bool
    target_error_m: float | None
    post_contact_ball_speed_mps: float
    exact_replay: bool
    safety_failure_agent_ids: tuple[str, ...] = ()
    root_pose_write_after_start: bool = False
    ball_state_write_after_start: bool = False
    pixels_used_for_scoring: bool = False
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.continuous_chain_event.v1"

    def __post_init__(self) -> None:
        flags = (
            self.contact_observed,
            self.safe,
            self.phase_ready,
            self.exact_replay,
            self.root_pose_write_after_start,
            self.ball_state_write_after_start,
            self.pixels_used_for_scoring,
            self.hardware_command_sent,
        )
        safety_failures = tuple(self.safety_failure_agent_ids)
        if (
            not isinstance(self.phase, ContinuousChainPhase)
            or not _IDENTIFIER.fullmatch(self.agent_id)
            or not _IDENTIFIER.fullmatch(self.ball_id)
            or any(
                not _HASH.fullmatch(value)
                for value in (
                    self.input_ball_state_hash,
                    self.output_ball_state_hash,
                    self.decision_hash,
                    self.option_request_hash,
                )
            )
            or any(not isinstance(value, bool) for value in flags)
            or not math.isfinite(self.started_at_sec)
            or not math.isfinite(self.ended_at_sec)
            or self.started_at_sec < 0.0
            or self.ended_at_sec <= self.started_at_sec
            or (
                self.target_error_m is not None
                and (not math.isfinite(self.target_error_m) or self.target_error_m < 0.0)
            )
            or not math.isfinite(self.post_contact_ball_speed_mps)
            or self.post_contact_ball_speed_mps < 0.0
            or len(set(safety_failures)) != len(safety_failures)
            or any(not _IDENTIFIER.fullmatch(value) for value in safety_failures)
            or (self.safe and safety_failures)
        ):
            raise ValueError("continuous-chain event is invalid")
        object.__setattr__(self, "safety_failure_agent_ids", safety_failures)

    @property
    def event_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phase"] = self.phase.value
        value["safety_failure_agent_ids"] = list(self.safety_failure_agent_ids)
        return value


@dataclass(frozen=True)
class ContinuousChainAssessment:
    request_hash: str
    event_hashes: tuple[str, ...]
    passed: bool
    completed_phase_count: int
    earliest_failure: ContinuousChainFailure | None
    failure_phase: ContinuousChainPhase | None
    focal_agent_id: str | None
    ball_lineage_verified: bool
    safe: bool
    exact_replay: bool
    downstream_credit_blocked: bool
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.continuous_chain_assessment.v1"

    def __post_init__(self) -> None:
        if (
            not _HASH.fullmatch(self.request_hash)
            or any(not _HASH.fullmatch(value) for value in self.event_hashes)
            or not 0 <= self.completed_phase_count <= len(_PHASES)
            or self.passed != (self.completed_phase_count == len(_PHASES))
            or self.passed != (self.earliest_failure is None)
            or ((self.earliest_failure is None) != (self.failure_phase is None))
            or (self.focal_agent_id is not None and not _IDENTIFIER.fullmatch(self.focal_agent_id))
            or not all(
                isinstance(value, bool)
                for value in (
                    self.passed,
                    self.ball_lineage_verified,
                    self.safe,
                    self.exact_replay,
                    self.downstream_credit_blocked,
                    self.hardware_command_sent,
                )
            )
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_command_sent
        ):
            raise ValueError("continuous-chain assessment is invalid")

    @property
    def assessment_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["earliest_failure"] = (
            None if self.earliest_failure is None else self.earliest_failure.value
        )
        value["failure_phase"] = None if self.failure_phase is None else self.failure_phase.value
        return value


def assess_continuous_option_chain(
    request: ContinuousOptionChainRequest,
    events: tuple[ContinuousChainEvent, ...],
) -> ContinuousChainAssessment:
    """Select the earliest causal failure and block downstream blame."""

    sequence = tuple(events)
    if not 1 <= len(sequence) <= len(_PHASES):
        raise ValueError("continuous chain must contain a non-empty phase prefix")
    for index, event in enumerate(sequence):
        if (
            event.phase is not _PHASES[index]
            or event.agent_id != request.agent_for(event.phase)
            or event.ball_id != request.ball_id
            or (index > 0 and event.started_at_sec < sequence[index - 1].ended_at_sec)
        ):
            raise ValueError("continuous chain phase, owner, ball, or time order changed")

    lineage = sequence[0].input_ball_state_hash == request.initial_ball_state_hash and all(
        current.input_ball_state_hash == previous.output_ball_state_hash
        for previous, current in zip(sequence, sequence[1:], strict=False)
    )
    if not lineage:
        return _assessment(
            request,
            sequence,
            completed=0,
            failure=ContinuousChainFailure.BALL_LINEAGE_BROKEN,
            phase=sequence[0].phase,
            focal_agent_id=None,
            lineage=False,
        )

    completed = 0
    for event in sequence:
        failure = _event_failure(request, event)
        if failure is not None:
            focal = (
                None
                if failure is ContinuousChainFailure.NONDETERMINISTIC_REPLAY
                else event.agent_id
            )
            if failure is ContinuousChainFailure.UNSAFE_MOTION:
                focal = (
                    event.safety_failure_agent_ids[0]
                    if len(event.safety_failure_agent_ids) == 1
                    else None
                )
            return _assessment(
                request,
                sequence,
                completed=completed,
                failure=failure,
                phase=event.phase,
                focal_agent_id=focal,
                lineage=True,
            )
        completed += 1

    if len(sequence) != len(_PHASES):
        phase = _PHASES[len(sequence)]
        return _assessment(
            request,
            sequence,
            completed=completed,
            failure=ContinuousChainFailure.CHAIN_INCOMPLETE,
            phase=phase,
            focal_agent_id=request.agent_for(phase),
            lineage=True,
        )
    return _assessment(
        request,
        sequence,
        completed=completed,
        failure=None,
        phase=None,
        focal_agent_id=None,
        lineage=True,
    )


def build_chain_repair_lease(
    *,
    cells: tuple[RosclawSoccerAgentCell, ...],
    assessment: ContinuousChainAssessment,
    dataset_manifest_hash: str,
    scenario_contract_hash: str,
    maximum_optimizer_steps: int,
) -> PlasticityLease:
    """Open plasticity only for the role blamed by verified physical evidence."""

    if assessment.passed or assessment.focal_agent_id is None:
        raise ValueError("chain has no role-local failure eligible for plasticity")
    return build_agent_plasticity_lease(
        cells=cells,
        focal_agent_id=assessment.focal_agent_id,
        dataset_manifest_hash=dataset_manifest_hash,
        scenario_contract_hash=scenario_contract_hash,
        maximum_optimizer_steps=maximum_optimizer_steps,
    )


def _event_failure(
    request: ContinuousOptionChainRequest,
    event: ContinuousChainEvent,
) -> ContinuousChainFailure | None:
    if not event.exact_replay:
        return ContinuousChainFailure.NONDETERMINISTIC_REPLAY
    if (
        not event.safe
        or event.root_pose_write_after_start
        or event.ball_state_write_after_start
        or event.pixels_used_for_scoring
        or event.hardware_command_sent
    ):
        return ContinuousChainFailure.UNSAFE_MOTION
    if event.phase is ContinuousChainPhase.RECEIVE:
        if not event.phase_ready:
            return ContinuousChainFailure.RECEIVE_NOT_READY
        if event.target_error_m is None or event.target_error_m > request.maximum_receive_error_m:
            return ContinuousChainFailure.RECEIVE_INACCURATE
        return None
    if not event.contact_observed:
        return {
            ContinuousChainPhase.PASS: ContinuousChainFailure.PASS_NO_CONTACT,
            ContinuousChainPhase.SHOOT: ContinuousChainFailure.SHOOT_NO_CONTACT,
            ContinuousChainPhase.SAVE: ContinuousChainFailure.SAVE_NO_CONTACT,
        }[event.phase]
    if event.phase is ContinuousChainPhase.PASS:
        if event.post_contact_ball_speed_mps < request.minimum_pass_speed_mps:
            return ContinuousChainFailure.PASS_TOO_SLOW
        if event.target_error_m is None or event.target_error_m > request.maximum_pass_error_m:
            return ContinuousChainFailure.PASS_INACCURATE
    elif event.phase is ContinuousChainPhase.SHOOT:
        if event.post_contact_ball_speed_mps < request.minimum_shot_speed_mps:
            return ContinuousChainFailure.SHOT_TOO_SLOW
        if event.target_error_m is None or event.target_error_m > request.maximum_shot_error_m:
            return ContinuousChainFailure.SHOT_INACCURATE
    elif event.post_contact_ball_speed_mps < request.minimum_save_deflection_mps:
        return ContinuousChainFailure.SAVE_WEAK_DEFLECTION
    return None


def _assessment(
    request: ContinuousOptionChainRequest,
    events: tuple[ContinuousChainEvent, ...],
    *,
    completed: int,
    failure: ContinuousChainFailure | None,
    phase: ContinuousChainPhase | None,
    focal_agent_id: str | None,
    lineage: bool,
) -> ContinuousChainAssessment:
    return ContinuousChainAssessment(
        request_hash=request.request_hash,
        event_hashes=tuple(event.event_hash for event in events),
        passed=failure is None,
        completed_phase_count=completed,
        earliest_failure=failure,
        failure_phase=phase,
        focal_agent_id=focal_agent_id,
        ball_lineage_verified=lineage,
        safe=all(
            event.safe
            and not event.root_pose_write_after_start
            and not event.ball_state_write_after_start
            and not event.hardware_command_sent
            for event in events
        ),
        exact_replay=all(event.exact_replay for event in events),
        downstream_credit_blocked=failure is not None,
    )


__all__ = [
    "ChainRoleBinding",
    "ContinuousChainAssessment",
    "ContinuousChainEvent",
    "ContinuousChainFailure",
    "ContinuousChainPhase",
    "ContinuousOptionChainRequest",
    "assess_continuous_option_chain",
    "build_chain_repair_lease",
]
