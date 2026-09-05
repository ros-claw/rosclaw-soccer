"""Role-qualified selection of physical option backends.

Tactical autonomy is not authority to run an arbitrary whole-body prior.  A
playmaker PASS, finisher SHOOT, and goalkeeper SAVE each require a backend
whose evidence covers that role and option.  This router is high-level and
SIM-only; it never executes joints or torques.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rosclaw_soccer.growth.independent_agent_cell import RosclawSoccerAgentCell
from rosclaw_soccer.growth.physical_option_router import PhysicalSoccerOption
from rosclaw_soccer.growth.role_self_model import MatchRole, SoccerSkill
from rosclaw_soccer.sim.contracts import hash_json

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


class RoleOptionBackend(StrEnum):
    GENERIC_FREEKICK = "generic_freekick"
    PASS_AIM_RESIDUAL = "pass_aim_residual"
    DYNAMIC_LEAD_PASS = "dynamic_lead_pass"
    RUNTIME_FINISH_PLAN = "runtime_finish_plan"
    CONTEXTUAL_FINISH_TARGET = "contextual_finish_target"
    VISIBLE_BALL_GOALKEEPER = "visible_ball_goalkeeper"


_AUTHORITY = {
    PhysicalSoccerOption.PASS: (MatchRole.PLAYMAKER, SoccerSkill.LEAD_PASS),
    PhysicalSoccerOption.SHOOT: (MatchRole.FINISHER, SoccerSkill.FINISHING),
    PhysicalSoccerOption.SAVE: (MatchRole.GOALKEEPER, SoccerSkill.SAVE),
    PhysicalSoccerOption.DISTRIBUTE: (MatchRole.GOALKEEPER, SoccerSkill.DISTRIBUTION),
}

_BACKENDS = {
    PhysicalSoccerOption.PASS: {
        RoleOptionBackend.PASS_AIM_RESIDUAL,
        RoleOptionBackend.DYNAMIC_LEAD_PASS,
    },
    PhysicalSoccerOption.SHOOT: {
        RoleOptionBackend.RUNTIME_FINISH_PLAN,
        RoleOptionBackend.CONTEXTUAL_FINISH_TARGET,
    },
    PhysicalSoccerOption.SAVE: {RoleOptionBackend.VISIBLE_BALL_GOALKEEPER},
    PhysicalSoccerOption.DISTRIBUTE: {RoleOptionBackend.DYNAMIC_LEAD_PASS},
}


@dataclass(frozen=True)
class RoleOptionBackendCandidate:
    backend: RoleOptionBackend
    option: PhysicalSoccerOption
    artifact_hash: str
    evidence_hash: str
    distinct_context_count: int
    distinct_trajectory_count: int
    strict_replay: bool
    holdout_passed: bool
    parent_retention_passed: bool
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    schema_version: str = "rosclaw_soccer.role_option_backend_candidate.v1"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.backend, RoleOptionBackend)
            or not isinstance(self.option, PhysicalSoccerOption)
            or not _HASH.fullmatch(self.artifact_hash)
            or not _HASH.fullmatch(self.evidence_hash)
            or isinstance(self.distinct_context_count, bool)
            or isinstance(self.distinct_trajectory_count, bool)
            or not 0 <= self.distinct_context_count <= 1_000_000
            or not 0 <= self.distinct_trajectory_count <= 1_000_000
            or not all(
                isinstance(value, bool)
                for value in (
                    self.strict_replay,
                    self.holdout_passed,
                    self.parent_retention_passed,
                )
            )
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.direct_joint_torque_output
        ):
            raise ValueError("role option backend candidate is invalid")

    @property
    def evidence_ready(self) -> bool:
        minimum = 6 if self.backend is RoleOptionBackend.DYNAMIC_LEAD_PASS else 4
        return bool(
            self.backend in _BACKENDS[self.option]
            and self.distinct_context_count >= minimum
            and self.distinct_trajectory_count >= minimum
            and self.strict_replay
            and self.holdout_passed
            and self.parent_retention_passed
        )

    @property
    def candidate_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["backend"] = self.backend.value
        value["option"] = self.option.value
        value["evidence_ready"] = self.evidence_ready
        return value


@dataclass(frozen=True)
class RoleOptionBackendRoute:
    agent_id: str
    cell_hash: str
    option: PhysicalSoccerOption
    selected_backend: RoleOptionBackend | None
    candidate_hash: str | None
    accepted: bool
    reason: str
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.role_option_backend_route.v1"

    def __post_init__(self) -> None:
        if (
            not self.agent_id
            or not _HASH.fullmatch(self.cell_hash)
            or not isinstance(self.option, PhysicalSoccerOption)
            or (self.candidate_hash is not None and not _HASH.fullmatch(self.candidate_hash))
            or self.accepted != (self.selected_backend is not None)
            or self.accepted != (self.candidate_hash is not None)
            or not self.reason
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("role option backend route is invalid")

    @property
    def route_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["option"] = self.option.value
        value["selected_backend"] = (
            None if self.selected_backend is None else self.selected_backend.value
        )
        return value


def select_role_option_backend(
    *,
    cell: RosclawSoccerAgentCell,
    option: PhysicalSoccerOption,
    candidates: tuple[RoleOptionBackendCandidate, ...],
) -> RoleOptionBackendRoute:
    """Choose a role/evidence-qualified backend, otherwise reject the option."""

    role, skill = _AUTHORITY[option]
    if cell.self_model.primary_role is not role or not any(
        binding.skill is skill for binding in cell.self_model.skills
    ):
        raise ValueError("agent cell does not own the requested physical option")
    matching = tuple(candidate for candidate in candidates if candidate.option is option)
    ready = tuple(candidate for candidate in matching if candidate.evidence_ready)
    preference = {
        RoleOptionBackend.DYNAMIC_LEAD_PASS: 5,
        RoleOptionBackend.RUNTIME_FINISH_PLAN: 5,
        RoleOptionBackend.CONTEXTUAL_FINISH_TARGET: 5,
        RoleOptionBackend.VISIBLE_BALL_GOALKEEPER: 5,
        RoleOptionBackend.PASS_AIM_RESIDUAL: 4,
        RoleOptionBackend.GENERIC_FREEKICK: 0,
    }
    if not ready:
        return RoleOptionBackendRoute(
            agent_id=cell.agent_id,
            cell_hash=cell.cell_hash,
            option=option,
            selected_backend=None,
            candidate_hash=None,
            accepted=False,
            reason="NO_ROLE_QUALIFIED_BACKEND",
        )
    selected = max(
        ready,
        key=lambda value: (
            preference[value.backend],
            value.distinct_context_count,
            value.distinct_trajectory_count,
            value.candidate_hash,
        ),
    )
    return RoleOptionBackendRoute(
        agent_id=cell.agent_id,
        cell_hash=cell.cell_hash,
        option=option,
        selected_backend=selected.backend,
        candidate_hash=selected.candidate_hash,
        accepted=True,
        reason="ROLE_AND_EVIDENCE_QUALIFIED",
    )


__all__ = [
    "RoleOptionBackend",
    "RoleOptionBackendCandidate",
    "RoleOptionBackendRoute",
    "select_role_option_backend",
]
