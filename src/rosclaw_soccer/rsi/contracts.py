"""Minimal, content-bound Physical RSI contracts; no action execution authority.

The public records hold references to raw episode files stored outside Git.
No contract here can issue ROS, hardware, or real-robot motion requests.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Protocol

from rosclaw_soccer.sim.contracts import hash_json

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.:-]{0,127}\Z")


def _hash(value: str) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _id(value: str) -> bool:
    return type(value) is str and _IDENTIFIER.fullmatch(value) is not None


def _scalar(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _vector(value: tuple[float, ...], *, minimum: int = 1) -> bool:
    return type(value) is tuple and len(value) >= minimum and all(_scalar(item) for item in value)


class EpisodePartition(StrEnum):
    DISCOVERY = "DISCOVERY"
    TRAIN = "TRAIN"
    CONSUMED_DEV = "CONSUMED_DEV"
    FRESH_HOLDOUT = "FRESH_HOLDOUT"
    SEALED = "SEALED"


@dataclass(frozen=True)
class AthleticIntent:
    """Desired root velocity [m/s], heading [rad], and contact preference."""

    velocity_xy_mps: tuple[float, float]
    heading_rad: float
    yaw_rate_rad_s: float
    body_height_m: float
    contact_intent: str
    sport_context: str
    future_target_xy_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if (
            not _vector(self.velocity_xy_mps, minimum=2)
            or len(self.velocity_xy_mps) != 2
            or not all(
                _scalar(v) for v in (self.heading_rad, self.yaw_rate_rad_s, self.body_height_m)
            )
            or self.body_height_m <= 0
            or not _id(self.contact_intent)
            or not _id(self.sport_context)
            or self.future_target_xy_m is not None
            and (
                not _vector(self.future_target_xy_m, minimum=2) or len(self.future_target_xy_m) != 2
            )
        ):
            raise ValueError(
                "finite body intent with an explicit contact and sport context required"
            )


@dataclass(frozen=True)
class AthleteObservation:
    """One proprioceptive simulator state; joint vectors are in artifact order."""

    body_id: str
    body_hash: str
    frame: int
    root_position_m: tuple[float, float, float]
    root_velocity_mps: tuple[float, float, float]
    root_quaternion_wxyz: tuple[float, float, float, float]
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    contact_flags: tuple[bool, ...]

    def __post_init__(self) -> None:
        if (
            not _id(self.body_id)
            or not _hash(self.body_hash)
            or type(self.frame) is not int
            or self.frame < 0
            or not _vector(self.root_position_m, minimum=3)
            or len(self.root_position_m) != 3
            or not _vector(self.root_velocity_mps, minimum=3)
            or len(self.root_velocity_mps) != 3
            or not _vector(self.root_quaternion_wxyz, minimum=4)
            or len(self.root_quaternion_wxyz) != 4
            or abs(sum(v * v for v in self.root_quaternion_wxyz) - 1.0) > 1.0e-3
            or not _vector(self.joint_position)
            or not _vector(self.joint_velocity)
            or len(self.joint_position) != len(self.joint_velocity)
            or type(self.contact_flags) is not tuple
            or not self.contact_flags
            or any(type(flag) is not bool for flag in self.contact_flags)
        ):
            raise ValueError("finite, complete athlete proprioception required")


@dataclass(frozen=True)
class MotorAction:
    """One simulated target in joint or latent coordinates, never raw authority."""

    body_hash: str
    policy_hash: str
    frame: int
    joint_target: tuple[float, ...] | None = None
    motor_latent: tuple[float, ...] | None = None
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        target = self.joint_target if self.joint_target is not None else self.motor_latent
        if (
            not _hash(self.body_hash)
            or not _hash(self.policy_hash)
            or type(self.frame) is not int
            or self.frame < 0
            or (self.joint_target is None) == (self.motor_latent is None)
            or target is None
            or not _vector(target)
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("single finite, simulation-only motor proposal required")


@dataclass(frozen=True)
class PolicyArtifact:
    """Reproducible model/body/controller binding without a promotion claim."""

    artifact_id: str
    backend_id: str
    code_hash: str
    weights_hash: str
    body_hash: str
    observation_hash: str
    action_hash: str
    joint_map_hash: str
    physics_hash: str
    gain_hash: str
    parent_hash: str | None = None
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if (
            not _id(self.artifact_id)
            or not _id(self.backend_id)
            or any(
                not _hash(value)
                for value in (
                    self.code_hash,
                    self.weights_hash,
                    self.body_hash,
                    self.observation_hash,
                    self.action_hash,
                    self.joint_map_hash,
                    self.physics_hash,
                    self.gain_hash,
                )
            )
            or self.parent_hash is not None
            and not _hash(self.parent_hash)
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError(
                "policy artifact needs complete source, model, body and physics binding"
            )

    @property
    def contract_hash(self) -> str:
        return str(hash_json(asdict(self)))


class AthletePolicy(Protocol):
    @property
    def artifact(self) -> PolicyArtifact: ...

    def step(self, observation: AthleteObservation, intent: AthleticIntent) -> MotorAction: ...


class FootballSkill(Protocol):
    @property
    def contract_hash(self) -> str: ...

    def propose(self, observation: AthleteObservation, intent: AthleticIntent) -> MotorAction: ...


class PlayerPolicy(Protocol):
    @property
    def contract_hash(self) -> str: ...

    def choose_skill(self, observation_hash: str) -> str: ...


class TeamPolicy(Protocol):
    @property
    def contract_hash(self) -> str: ...

    def assign_intents(self, observation_hash: str) -> tuple[tuple[str, str], ...]: ...


@dataclass(frozen=True)
class PhysicalEpisode:
    """Hashed external trajectory and event record for one physical rollout."""

    episode_id: str
    partition: EpisodePartition
    course_hash: str
    initial_state_hash: str
    policy_hash: str
    code_hash: str
    environment_hash: str
    physics_hash: str
    trajectory_hash: str
    event_hash: str
    safe: bool
    task_success: bool
    teacher_active: bool
    backend: str = "MUJOCO_CPU"
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if (
            not _id(self.episode_id)
            or not isinstance(self.partition, EpisodePartition)
            or any(
                not _hash(value)
                for value in (
                    self.course_hash,
                    self.initial_state_hash,
                    self.policy_hash,
                    self.code_hash,
                    self.environment_hash,
                    self.physics_hash,
                    self.trajectory_hash,
                    self.event_hash,
                )
            )
            or any(
                type(value) is not bool
                for value in (self.safe, self.task_success, self.teacher_active)
            )
            or self.task_success
            and not self.safe
            or self.partition in (EpisodePartition.FRESH_HOLDOUT, EpisodePartition.SEALED)
            and self.teacher_active
            or type(self.backend) is not str
            or not _id(self.backend.lower())
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError(
                "physical episode requires safe, source-bound, teacher-free exam evidence"
            )

    @property
    def contract_hash(self) -> str:
        return str(hash_json(asdict(self)))


def validate_athlete_proposal(
    artifact: PolicyArtifact,
    observation: AthleteObservation,
    action: MotorAction,
) -> None:
    """Reject stale/foreign proposals before a simulator adapter may consume one."""
    if (
        action.body_hash != artifact.body_hash
        or observation.body_hash != artifact.body_hash
        or action.policy_hash != artifact.contract_hash
        or action.frame != observation.frame
        or action.joint_target is not None
        and len(action.joint_target) != len(observation.joint_position)
    ):
        raise ValueError("athlete proposal differs from bound body, model, clock or joint map")
