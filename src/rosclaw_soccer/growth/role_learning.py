"""Fail-closed multi-agent growth contracts for Soccer roles.

The passer, shooter, and goalkeeper are separate policy owners.  Training may
use centralized state and critics, but promotion is evaluated per role so a
candidate cannot improve the highlight outcome by sacrificing another agent's
stability or competence.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


class SoccerRole(StrEnum):
    PASSER = "passer"
    SHOOTER = "shooter"
    GOALKEEPER = "goalkeeper"


class SoccerSide(StrEnum):
    ATTACK = "attack"
    DEFENSE = "defense"


_ROLE_SIDE = {
    SoccerRole.PASSER: SoccerSide.ATTACK,
    SoccerRole.SHOOTER: SoccerSide.ATTACK,
    SoccerRole.GOALKEEPER: SoccerSide.DEFENSE,
}


@dataclass(frozen=True)
class RolePolicyBinding:
    """Identity and lineage of one independently trainable role policy."""

    agent_id: str
    role: SoccerRole
    policy_version: str
    artifact_hash: str
    parent_artifact_hash: str
    observation_contract_hash: str
    action_contract_hash: str
    generation: int
    schema_version: str = "rosclaw_soccer.role_policy_binding.v1"

    def __post_init__(self) -> None:
        _identifier("agent_id", self.agent_id)
        _identifier("policy_version", self.policy_version)
        if not isinstance(self.role, SoccerRole):
            raise ValueError("role policy binding has an unknown role")
        for label in (
            "artifact_hash",
            "parent_artifact_hash",
            "observation_contract_hash",
            "action_contract_hash",
        ):
            _content_hash(label, getattr(self, label))
        if not 1 <= self.generation <= 1_000_000:
            raise ValueError("role policy generation must be in [1, 1000000]")

    @property
    def changed_from_parent(self) -> bool:
        return self.artifact_hash != self.parent_artifact_hash

    @property
    def side(self) -> SoccerSide:
        return _ROLE_SIDE[self.role]

    @property
    def binding_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        value["side"] = self.side.value
        return value


@dataclass(frozen=True)
class RoleEpisodeOutcome:
    """One role's measured contribution in a shared-world episode.

    ``counterfactual_side_reward`` is produced by a matched policy ablation,
    not a critic prediction.  Its difference from ``side_reward`` prevents a
    passer from receiving full credit for a finish it did not enable and lets
    goalkeeper competence be scored against the defense side rather than the
    attacking highlight.
    """

    agent_id: str
    role: SoccerRole
    policy_artifact_hash: str
    action_trace_hash: str
    counterfactual_evidence_hash: str
    counterfactual_parent_artifact_hash: str
    counterfactual_scenario_hash: str
    counterfactual_environment_hash: str
    counterfactual_seed: int
    individual_reward: float
    side_reward: float
    counterfactual_side_reward: float
    stability_score: float
    safety_cost: float
    schema_version: str = "rosclaw_soccer.role_episode_outcome.v1"

    def __post_init__(self) -> None:
        _identifier("agent_id", self.agent_id)
        if not isinstance(self.role, SoccerRole):
            raise ValueError("role outcome has an unknown role")
        for label in (
            "policy_artifact_hash",
            "action_trace_hash",
            "counterfactual_evidence_hash",
            "counterfactual_parent_artifact_hash",
            "counterfactual_scenario_hash",
            "counterfactual_environment_hash",
        ):
            _content_hash(label, getattr(self, label))
        if (
            isinstance(self.counterfactual_seed, bool)
            or not 0 <= self.counterfactual_seed <= 2**32 - 1
        ):
            raise ValueError("counterfactual seed must be an unsigned 32-bit integer")
        for label in (
            "individual_reward",
            "side_reward",
            "counterfactual_side_reward",
            "stability_score",
            "safety_cost",
        ):
            value = getattr(self, label)
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if not -1.0 <= self.individual_reward <= 1.0:
            raise ValueError("individual_reward must be normalized to [-1, 1]")
        if not -1.0 <= self.side_reward <= 1.0:
            raise ValueError("side_reward must be normalized to [-1, 1]")
        if not -1.0 <= self.counterfactual_side_reward <= 1.0:
            raise ValueError("counterfactual_side_reward must be normalized to [-1, 1]")
        if not 0.0 <= self.stability_score <= 1.0:
            raise ValueError("stability_score must be in [0, 1]")
        if not 0.0 <= self.safety_cost <= 1.0:
            raise ValueError("safety_cost must be in [0, 1]")

    @property
    def side(self) -> SoccerSide:
        return _ROLE_SIDE[self.role]

    @property
    def difference_reward(self) -> float:
        return self.side_reward - self.counterfactual_side_reward

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        value["side"] = self.side.value
        value["difference_reward"] = self.difference_reward
        return value


@dataclass(frozen=True)
class SharedWorldTeamEpisode:
    """Hash-bound episode containing exactly three independent role agents."""

    episode_id: str
    scenario_hash: str
    environment_hash: str
    trajectory_hash: str
    seed: int
    policies: tuple[RolePolicyBinding, ...]
    outcomes: tuple[RoleEpisodeOutcome, ...]
    strict_replay: bool
    rolling_authenticity_passed: bool
    physical_event_order_passed: bool
    pixels_used_for_scoring: bool = False
    activation_ceiling: str = "SIM_ONLY"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.shared_world_team_episode.v1"

    def __post_init__(self) -> None:
        _identifier("episode_id", self.episode_id)
        for label in ("scenario_hash", "environment_hash", "trajectory_hash"):
            _content_hash(label, getattr(self, label))
        if isinstance(self.seed, bool) or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("team episode seed must be an unsigned 32-bit integer")
        policies = tuple(self.policies)
        outcomes = tuple(self.outcomes)
        expected = set(SoccerRole)
        if {item.role for item in policies} != expected or len(policies) != len(expected):
            raise ValueError("team episode requires one independent policy for every role")
        if {item.role for item in outcomes} != expected or len(outcomes) != len(expected):
            raise ValueError("team episode requires one measured outcome for every role")
        policy_by_role = {item.role: item for item in policies}
        for outcome in outcomes:
            policy = policy_by_role[outcome.role]
            if outcome.agent_id != policy.agent_id:
                raise ValueError("role outcome agent identity does not match its policy")
            if outcome.policy_artifact_hash != policy.artifact_hash:
                raise ValueError("role outcome is not bound to its executed policy artifact")
            if outcome.counterfactual_parent_artifact_hash != policy.parent_artifact_hash:
                raise ValueError("role counterfactual did not ablate to the bound parent policy")
            if outcome.counterfactual_seed != self.seed:
                raise ValueError("role counterfactual seed does not match the team episode")
            if outcome.counterfactual_scenario_hash != self.scenario_hash:
                raise ValueError("role counterfactual scenario does not match the team episode")
            if outcome.counterfactual_environment_hash != self.environment_hash:
                raise ValueError("role counterfactual environment does not match the team episode")
        if len({item.agent_id for item in policies}) != len(expected):
            raise ValueError("team role agents must have unique identities")
        if self.activation_ceiling != "SIM_ONLY" or self.physics_authority != "CPU_MUJOCO":
            raise ValueError("team episode must remain SIM_ONLY CPU MuJoCo")
        if self.hardware_command_sent or self.pixels_used_for_scoring:
            raise ValueError("team episode violates its simulation evidence boundary")
        object.__setattr__(self, "policies", policies)
        object.__setattr__(self, "outcomes", outcomes)

    @property
    def promotion_eligible(self) -> bool:
        return bool(
            self.strict_replay
            and self.rolling_authenticity_passed
            and self.physical_event_order_passed
            and all(policy.changed_from_parent for policy in self.policies)
            and all(outcome.safety_cost == 0.0 for outcome in self.outcomes)
        )

    @property
    def episode_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def outcome(self, role: SoccerRole) -> RoleEpisodeOutcome:
        return next(item for item in self.outcomes if item.role is role)

    def policy(self, role: SoccerRole) -> RolePolicyBinding:
        return next(item for item in self.policies if item.role is role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "scenario_hash": self.scenario_hash,
            "environment_hash": self.environment_hash,
            "trajectory_hash": self.trajectory_hash,
            "seed": self.seed,
            "policies": [item.to_dict() for item in self.policies],
            "outcomes": [item.to_dict() for item in self.outcomes],
            "strict_replay": self.strict_replay,
            "rolling_authenticity_passed": self.rolling_authenticity_passed,
            "physical_event_order_passed": self.physical_event_order_passed,
            "promotion_eligible": self.promotion_eligible,
            "pixels_used_for_scoring": self.pixels_used_for_scoring,
            "activation_ceiling": self.activation_ceiling,
            "physics_authority": self.physics_authority,
            "hardware_command_sent": self.hardware_command_sent,
        }


@dataclass(frozen=True)
class JointGrowthGateConfig:
    minimum_seeds: int = 3
    minimum_individual_reward_improvement: float = 0.01
    minimum_difference_reward_improvement: float = 0.005
    maximum_stability_regression: float = 0.01
    cvar_fraction: float = 0.25
    schema_version: str = "rosclaw_soccer.joint_growth_gate_config.v1"

    def __post_init__(self) -> None:
        if not 3 <= self.minimum_seeds <= 1000:
            raise ValueError("joint growth minimum_seeds must be in [3, 1000]")
        for label in (
            "minimum_individual_reward_improvement",
            "minimum_difference_reward_improvement",
            "maximum_stability_regression",
        ):
            value = getattr(self, label)
            if not math.isfinite(value) or not 0.0 <= value <= 0.25:
                raise ValueError(f"{label} must be in [0, 0.25]")
        if not 0.05 <= self.cvar_fraction <= 0.50:
            raise ValueError("cvar_fraction must be in [0.05, 0.50]")


@dataclass(frozen=True)
class RoleGrowthMetrics:
    role: SoccerRole
    parent_individual_mean: float
    candidate_individual_mean: float
    individual_improvement: float
    parent_difference_mean: float
    candidate_difference_mean: float
    difference_improvement: float
    parent_difference_cvar: float
    candidate_difference_cvar: float
    parent_stability_worst: float
    candidate_stability_worst: float
    candidate_safety_cost_max: float
    passed: bool
    schema_version: str = "rosclaw_soccer.role_growth_metrics.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        return value


@dataclass(frozen=True)
class JointGrowthDecision:
    passed: bool
    reasons: tuple[str, ...]
    changed_roles: tuple[SoccerRole, ...]
    seed_count: int
    role_metrics: tuple[RoleGrowthMetrics, ...]
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.joint_growth_decision.v1"

    @property
    def decision_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "changed_roles": [role.value for role in self.changed_roles],
            "seed_count": self.seed_count,
            "role_metrics": [item.to_dict() for item in self.role_metrics],
            "activation_ceiling": self.activation_ceiling,
            "hardware_authorized": self.hardware_authorized,
        }


def evaluate_joint_growth(
    *,
    parent: tuple[SharedWorldTeamEpisode, ...],
    candidate: tuple[SharedWorldTeamEpisode, ...],
    config: JointGrowthGateConfig | None = None,
) -> JointGrowthDecision:
    """Promote only a jointly improved, retained, safe three-role generation."""

    gate = config or JointGrowthGateConfig()
    parent_by_seed = _episode_suite(parent, "parent")
    candidate_by_seed = _episode_suite(candidate, "candidate")
    if set(parent_by_seed) != set(candidate_by_seed):
        raise ValueError("parent and candidate team suites must use identical seeds")
    if len(parent_by_seed) < gate.minimum_seeds:
        raise ValueError("joint growth suite has insufficient aligned seeds")
    reasons: list[str] = []
    for seed in sorted(parent_by_seed):
        left, right = parent_by_seed[seed], candidate_by_seed[seed]
        if left.scenario_hash != right.scenario_hash:
            raise ValueError("joint growth seed pairs must share one scenario")
        if left.environment_hash != right.environment_hash:
            raise ValueError("joint growth seed pairs must share one environment")
        for role in SoccerRole:
            parent_policy = left.policy(role)
            candidate_policy = right.policy(role)
            if candidate_policy.agent_id != parent_policy.agent_id:
                raise ValueError("candidate role agent identity changed inside one generation")
            if (
                candidate_policy.observation_contract_hash
                != parent_policy.observation_contract_hash
            ):
                raise ValueError(
                    "candidate role observation contract changed inside one comparison"
                )
            if candidate_policy.action_contract_hash != parent_policy.action_contract_hash:
                raise ValueError("candidate role action contract changed inside one comparison")
            if candidate_policy.generation != parent_policy.generation + 1:
                raise ValueError("candidate role generation must increment its parent by one")
        if not left.promotion_eligible:
            reasons.append(f"parent_seed_{seed}_ineligible")
        if not right.promotion_eligible:
            reasons.append(f"candidate_seed_{seed}_ineligible")

    changed: list[SoccerRole] = []
    role_metrics: list[RoleGrowthMetrics] = []
    for role in SoccerRole:
        parent_policies = {
            episode.policy(role).artifact_hash for episode in parent_by_seed.values()
        }
        candidate_policies = {
            episode.policy(role).artifact_hash for episode in candidate_by_seed.values()
        }
        if len(parent_policies) != 1 or len(candidate_policies) != 1:
            raise ValueError("a role policy must be frozen across one evaluation suite")
        parent_hash = next(iter(parent_policies))
        candidate_hash = next(iter(candidate_policies))
        if candidate_hash != parent_hash:
            changed.append(role)
            for episode in candidate_by_seed.values():
                if episode.policy(role).parent_artifact_hash != parent_hash:
                    raise ValueError(
                        "candidate role policy lineage does not bind its evaluated parent"
                    )
        parent_outcomes = [parent_by_seed[seed].outcome(role) for seed in sorted(parent_by_seed)]
        candidate_outcomes = [
            candidate_by_seed[seed].outcome(role) for seed in sorted(candidate_by_seed)
        ]
        parent_individual = np.asarray(
            [item.individual_reward for item in parent_outcomes], dtype=np.float64
        )
        candidate_individual = np.asarray(
            [item.individual_reward for item in candidate_outcomes], dtype=np.float64
        )
        parent_difference = np.asarray(
            [item.difference_reward for item in parent_outcomes], dtype=np.float64
        )
        candidate_difference = np.asarray(
            [item.difference_reward for item in candidate_outcomes], dtype=np.float64
        )
        parent_stability = np.asarray(
            [item.stability_score for item in parent_outcomes], dtype=np.float64
        )
        candidate_stability = np.asarray(
            [item.stability_score for item in candidate_outcomes], dtype=np.float64
        )
        individual_improvement = float(np.mean(candidate_individual - parent_individual))
        difference_improvement = float(np.mean(candidate_difference - parent_difference))
        parent_cvar = _lower_cvar(parent_difference, gate.cvar_fraction)
        candidate_cvar = _lower_cvar(candidate_difference, gate.cvar_fraction)
        parent_worst = float(np.min(parent_stability))
        candidate_worst = float(np.min(candidate_stability))
        safety_max = float(max(item.safety_cost for item in candidate_outcomes))
        passed = bool(
            role in changed
            and individual_improvement >= gate.minimum_individual_reward_improvement
            and difference_improvement >= gate.minimum_difference_reward_improvement
            and candidate_cvar >= parent_cvar
            and candidate_worst >= parent_worst - gate.maximum_stability_regression
            and safety_max == 0.0
        )
        if not passed:
            reasons.append(f"{role.value}_growth_gate_failed")
        role_metrics.append(
            RoleGrowthMetrics(
                role=role,
                parent_individual_mean=float(np.mean(parent_individual)),
                candidate_individual_mean=float(np.mean(candidate_individual)),
                individual_improvement=individual_improvement,
                parent_difference_mean=float(np.mean(parent_difference)),
                candidate_difference_mean=float(np.mean(candidate_difference)),
                difference_improvement=difference_improvement,
                parent_difference_cvar=parent_cvar,
                candidate_difference_cvar=candidate_cvar,
                parent_stability_worst=parent_worst,
                candidate_stability_worst=candidate_worst,
                candidate_safety_cost_max=safety_max,
                passed=passed,
            )
        )
    if set(changed) != set(SoccerRole):
        reasons.append("not_all_role_policies_changed")
    return JointGrowthDecision(
        passed=not reasons and all(item.passed for item in role_metrics),
        reasons=tuple(dict.fromkeys(reasons)),
        changed_roles=tuple(changed),
        seed_count=len(parent_by_seed),
        role_metrics=tuple(role_metrics),
    )


def _episode_suite(
    episodes: tuple[SharedWorldTeamEpisode, ...], label: str
) -> dict[int, SharedWorldTeamEpisode]:
    values = tuple(episodes)
    if not values:
        raise ValueError(f"{label} team episode suite must not be empty")
    by_seed = {episode.seed: episode for episode in values}
    if len(by_seed) != len(values):
        raise ValueError(f"{label} team episode seeds must be unique")
    return by_seed


def _lower_cvar(values: NDArray[np.float64], fraction: float) -> float:
    count = max(1, int(math.ceil(len(values) * fraction)))
    return float(np.mean(np.sort(values)[:count]))


def _content_hash(label: str, value: str) -> None:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{label} must be a sha256 content hash")


def _identifier(label: str, value: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a normalized identifier")


__all__ = [
    "JointGrowthDecision",
    "JointGrowthGateConfig",
    "RoleEpisodeOutcome",
    "RoleGrowthMetrics",
    "RolePolicyBinding",
    "SharedWorldTeamEpisode",
    "SoccerRole",
    "SoccerSide",
    "evaluate_joint_growth",
]
