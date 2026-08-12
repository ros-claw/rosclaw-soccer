"""Evidence-driven joint policy search for the three Soccer role agents.

The search layer turns paired SIM_ONLY success/failure probes into bounded
role-local parameter updates.  It proposes a generation; it never promotes
one.  Promotion remains the responsibility of the disjoint discovery/holdout
gate in :mod:`rosclaw_soccer.growth.role_learning`.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.role_learning import SoccerRole
from rosclaw_soccer.sim.contracts import hash_json

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_PARAMETER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class RolePolicySearchSpace:
    role: SoccerRole
    parameter_names: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    maximum_updates: tuple[float, ...]
    schema_version: str = "rosclaw_soccer.role_policy_search_space.v1"

    def __post_init__(self) -> None:
        size = len(self.parameter_names)
        if not 2 <= size <= 64 or len(set(self.parameter_names)) != size:
            raise ValueError("role search parameters must contain 2-64 unique names")
        if any(not _PARAMETER.fullmatch(name) for name in self.parameter_names):
            raise ValueError("role search parameter name is invalid")
        if not all(len(values) == size for values in self.bounds):
            raise ValueError("role search bounds do not match the parameter count")
        for lower, upper, update in zip(*self.bounds, strict=True):
            if not all(math.isfinite(value) for value in (lower, upper, update)):
                raise ValueError("role search bounds must be finite")
            if not lower < upper or not 0.0 < update <= upper - lower:
                raise ValueError("role search bounds or maximum update are invalid")

    @property
    def bounds(self) -> tuple[tuple[float, ...], ...]:
        return self.lower_bounds, self.upper_bounds, self.maximum_updates

    @property
    def space_hash(self) -> str:
        value = asdict(self)
        value["role"] = self.role.value
        return str(hash_json(value))


@dataclass(frozen=True)
class RolePolicyVector:
    role: SoccerRole
    generation: int
    parameter_names: tuple[str, ...]
    values: tuple[float, ...]
    schema_version: str = "rosclaw_soccer.role_policy_vector.v1"

    def __post_init__(self) -> None:
        if not 0 <= self.generation <= 1_000_000:
            raise ValueError("role policy vector generation is invalid")
        if len(self.parameter_names) != len(self.values) or not self.values:
            raise ValueError("role policy vector shape is invalid")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("role policy vector parameter names must be unique")
        if any(not _PARAMETER.fullmatch(name) for name in self.parameter_names):
            raise ValueError("role policy vector parameter name is invalid")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("role policy vector values must be finite")

    @property
    def artifact_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        return value


@dataclass(frozen=True)
class MirroredRoleProbe:
    """One matched +epsilon/-epsilon policy experiment for one role."""

    role: SoccerRole
    seed: int
    scenario_hash: str
    environment_hash: str
    parent_artifact_hash: str
    perturbation: tuple[float, ...]
    positive_growth_score: float
    negative_growth_score: float
    positive_safety_cost: float
    negative_safety_cost: float
    positive_action_trace_hash: str
    negative_action_trace_hash: str
    strict_replay: bool = True
    rolling_authenticity_passed: bool = True
    activation_ceiling: str = "SIM_ONLY"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.mirrored_role_probe.v1"

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("mirrored role probe seed is invalid")
        for label in (
            "scenario_hash",
            "environment_hash",
            "parent_artifact_hash",
            "positive_action_trace_hash",
            "negative_action_trace_hash",
        ):
            if not _HASH.fullmatch(getattr(self, label)):
                raise ValueError(f"{label} must be a sha256 content hash")
        values = (
            self.positive_growth_score,
            self.negative_growth_score,
            self.positive_safety_cost,
            self.negative_safety_cost,
            *self.perturbation,
        )
        if not self.perturbation or not all(math.isfinite(value) for value in values):
            raise ValueError("mirrored role probe values must be finite")
        if any(abs(value) > 3.0 for value in self.perturbation):
            raise ValueError("mirrored role perturbation is outside [-3, 3]")
        if not all(0.0 <= value <= 1.0 for value in self.safety_costs):
            raise ValueError("mirrored role safety costs must be in [0, 1]")
        if self.activation_ceiling != "SIM_ONLY" or self.physics_authority != "CPU_MUJOCO":
            raise ValueError("mirrored role probe must remain SIM_ONLY CPU MuJoCo")
        if self.hardware_command_sent:
            raise ValueError("mirrored role probe contains a hardware command claim")

    @property
    def safety_costs(self) -> tuple[float, float]:
        return self.positive_safety_cost, self.negative_safety_cost

    @property
    def eligible(self) -> bool:
        return bool(
            self.strict_replay
            and self.rolling_authenticity_passed
            and self.positive_safety_cost == 0.0
            and self.negative_safety_cost == 0.0
        )


@dataclass(frozen=True)
class JointPolicySearchConfig:
    perturbation_sigma: float = 0.08
    learning_rate: float = 0.04
    gradient_norm_limit: float = 5.0
    minimum_safe_probes_per_role: int = 4
    minimum_parameter_change: float = 1e-8
    schema_version: str = "rosclaw_soccer.joint_policy_search_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.perturbation_sigma,
            self.learning_rate,
            self.gradient_norm_limit,
            self.minimum_parameter_change,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("joint policy search configuration must be finite and positive")
        if not 0.005 <= self.perturbation_sigma <= 0.50:
            raise ValueError("joint policy search sigma must be in [0.005, 0.50]")
        if not 0.001 <= self.learning_rate <= 0.25:
            raise ValueError("joint policy search learning rate must be in [0.001, 0.25]")
        if not 1.0 <= self.gradient_norm_limit <= 20.0:
            raise ValueError("joint policy search gradient norm limit must be in [1, 20]")
        if not 2 <= self.minimum_safe_probes_per_role <= 1000:
            raise ValueError("joint policy search safe probe count must be in [2, 1000]")


@dataclass(frozen=True)
class RolePolicySearchUpdate:
    role: SoccerRole
    parent: RolePolicyVector
    candidate: RolePolicyVector | None
    gradient: tuple[float, ...]
    safe_probe_seeds: tuple[int, ...]
    rejected_probe_seeds: tuple[int, ...]
    passed: bool
    reasons: tuple[str, ...]
    schema_version: str = "rosclaw_soccer.role_policy_search_update.v1"


@dataclass(frozen=True)
class JointPolicySearchDecision:
    passed: bool
    updates: tuple[RolePolicySearchUpdate, ...]
    candidates: tuple[RolePolicyVector, ...]
    reasons: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.joint_policy_search_decision.v1"


def learn_joint_policy_generation(
    *,
    parents: tuple[RolePolicyVector, ...],
    spaces: tuple[RolePolicySearchSpace, ...],
    probes: tuple[MirroredRoleProbe, ...],
    config: JointPolicySearchConfig | None = None,
) -> JointPolicySearchDecision:
    """Estimate one bounded proposal for every role from aligned SIM probes."""

    active = config or JointPolicySearchConfig()
    parent_by_role = _unique_by_role(parents, "parent")
    space_by_role = _unique_by_role(spaces, "search space")
    probe_by_role = {
        role: tuple(item for item in probes if item.role is role) for role in SoccerRole
    }
    _validate_aligned_probes(probe_by_role)
    updates = tuple(
        _learn_role_update(
            parent=parent_by_role[role],
            space=space_by_role[role],
            probes=probe_by_role[role],
            config=active,
        )
        for role in SoccerRole
    )
    reasons = tuple(f"{item.role.value}_search_failed" for item in updates if not item.passed)
    candidates = tuple(
        item.candidate for item in updates if item.passed and item.candidate is not None
    )
    passed = not reasons and len(candidates) == len(SoccerRole)
    return JointPolicySearchDecision(
        passed=passed,
        updates=updates,
        candidates=candidates if passed else (),
        reasons=reasons,
    )


def default_three_role_search_spaces() -> tuple[RolePolicySearchSpace, ...]:
    """Bounded adapters for the current pass, finish and goalkeeper policies."""

    return (
        RolePolicySearchSpace(
            role=SoccerRole.PASSER,
            parameter_names=(
                "stance_offset_y",
                "pelvis_yaw_offset",
                "com_shift_y",
                "swing_amplitude",
                "swing_speed_scale",
                "recovery_step_length",
            ),
            lower_bounds=(-0.12, -0.20, -0.08, 0.40, 0.80, 0.00),
            upper_bounds=(0.12, 0.20, 0.08, 1.15, 1.50, 0.15),
            maximum_updates=(0.01, 0.015, 0.01, 0.04, 0.04, 0.01),
        ),
        RolePolicySearchSpace(
            role=SoccerRole.SHOOTER,
            parameter_names=(
                "policy_start_sec",
                "foot_yaw_offset",
                "foot_pitch_offset",
                "com_shift_y",
                "swing_speed_scale",
                "recovery_step_length",
            ),
            lower_bounds=(1.80, -0.12, -0.18, -0.08, 0.80, 0.00),
            upper_bounds=(2.80, 0.12, 0.18, 0.08, 1.50, 0.15),
            maximum_updates=(0.04, 0.01, 0.01, 0.01, 0.04, 0.01),
        ),
        RolePolicySearchSpace(
            role=SoccerRole.GOALKEEPER,
            parameter_names=(
                "reaction_delay_sec",
                "lateral_position_gain",
                "maximum_lateral_speed_mps",
                "ready_shuffle_speed_mps",
                "arm_spread_rad",
                "maximum_waist_lean_rad",
            ),
            lower_bounds=(0.08, 0.50, 0.20, 0.00, 0.10, 0.00),
            upper_bounds=(0.35, 2.50, 0.40, 0.20, 0.45, 0.15),
            maximum_updates=(0.015, 0.08, 0.015, 0.01, 0.02, 0.01),
        ),
    )


def _learn_role_update(
    *,
    parent: RolePolicyVector,
    space: RolePolicySearchSpace,
    probes: tuple[MirroredRoleProbe, ...],
    config: JointPolicySearchConfig,
) -> RolePolicySearchUpdate:
    if parent.role is not space.role or parent.parameter_names != space.parameter_names:
        raise ValueError("role policy vector does not match its search space")
    values = np.asarray(parent.values, dtype=np.float64)
    lower = np.asarray(space.lower_bounds, dtype=np.float64)
    upper = np.asarray(space.upper_bounds, dtype=np.float64)
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError("parent role policy lies outside its search space")
    if any(item.parent_artifact_hash != parent.artifact_hash for item in probes):
        raise ValueError("mirrored role probe is not bound to its parent policy")
    if any(len(item.perturbation) != len(values) for item in probes):
        raise ValueError("mirrored role perturbation shape does not match the policy")
    safe = tuple(item for item in probes if item.eligible)
    rejected = tuple(item for item in probes if not item.eligible)
    reasons: list[str] = []
    if len(safe) < config.minimum_safe_probes_per_role:
        reasons.append("insufficient_safe_mirrored_probes")
    gradient: NDArray[np.float64] = np.zeros(len(values), dtype=np.float64)
    if safe:
        perturbations = np.asarray([item.perturbation for item in safe], dtype=np.float64)
        advantages = np.asarray(
            [item.positive_growth_score - item.negative_growth_score for item in safe],
            dtype=np.float64,
        )
        gradient = np.mean(advantages[:, None] * perturbations, axis=0) / (
            2.0 * config.perturbation_sigma
        )
        norm = float(np.linalg.norm(gradient))
        if norm > config.gradient_norm_limit:
            gradient *= config.gradient_norm_limit / norm
    raw_update = config.learning_rate * gradient
    bounded_update = np.clip(
        raw_update,
        -np.asarray(space.maximum_updates, dtype=np.float64),
        np.asarray(space.maximum_updates, dtype=np.float64),
    )
    candidate_values = np.clip(values + bounded_update, lower, upper)
    if float(np.max(np.abs(candidate_values - values))) < config.minimum_parameter_change:
        reasons.append("no_measured_learning_signal")
    candidate = None
    if not reasons:
        candidate = RolePolicyVector(
            role=parent.role,
            generation=parent.generation + 1,
            parameter_names=parent.parameter_names,
            values=tuple(float(value) for value in candidate_values),
        )
    return RolePolicySearchUpdate(
        role=parent.role,
        parent=parent,
        candidate=candidate,
        gradient=tuple(float(value) for value in gradient),
        safe_probe_seeds=tuple(sorted(item.seed for item in safe)),
        rejected_probe_seeds=tuple(sorted(item.seed for item in rejected)),
        passed=not reasons,
        reasons=tuple(reasons),
    )


def _validate_aligned_probes(
    probe_by_role: dict[SoccerRole, tuple[MirroredRoleProbe, ...]],
) -> None:
    seed_sets = [{item.seed for item in probe_by_role[role]} for role in SoccerRole]
    if not seed_sets[0] or any(values != seed_sets[0] for values in seed_sets[1:]):
        raise ValueError("all role searches must use the same shared-world seeds")
    for role in SoccerRole:
        if len(probe_by_role[role]) != len(seed_sets[0]):
            raise ValueError("a role search contains duplicate probe seeds")
    for seed in seed_sets[0]:
        matched = [
            next(item for item in probe_by_role[role] if item.seed == seed) for role in SoccerRole
        ]
        if len({item.scenario_hash for item in matched}) != 1:
            raise ValueError("role probes for one seed do not share a scenario")
        if len({item.environment_hash for item in matched}) != 1:
            raise ValueError("role probes for one seed do not share an environment")


def _unique_by_role(items: tuple[Any, ...], label: str) -> dict[SoccerRole, Any]:
    if len(items) != len(SoccerRole) or {item.role for item in items} != set(SoccerRole):
        raise ValueError(f"joint policy search requires one {label} for every role")
    return {item.role: item for item in items}


__all__ = [
    "JointPolicySearchConfig",
    "JointPolicySearchDecision",
    "MirroredRoleProbe",
    "RolePolicySearchSpace",
    "RolePolicySearchUpdate",
    "RolePolicyVector",
    "default_three_role_search_spaces",
    "learn_joint_policy_generation",
]
