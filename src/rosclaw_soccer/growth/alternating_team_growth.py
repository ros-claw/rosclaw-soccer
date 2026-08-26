"""Role-isolated Growth contracts for a three-player soccer team.

Only one player may be plastic in a round.  The other two players are frozen,
and phase-local scores deliberately avoid using the final match outcome as a
proxy for their competence.  This keeps a goalkeeper improvement from being
credited merely because the striker regressed, and keeps a striker candidate
from hiding an unstable follow-through behind a spectacular shot.

The module owns Soccer-specific roles and phases.  Generic Practice, Dream,
Memory, and promotion primitives remain ROSClaw Core responsibilities.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rosclaw_soccer.growth.role_learning import SoccerRole
from rosclaw_soccer.sim.contracts import hash_json

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


class TeamSkillPhase(StrEnum):
    LEAD_PASS = "lead_pass"
    RUNNING_INTERCEPT = "running_intercept"
    STRIKE = "strike"
    GLOVE_SAVE = "glove_save"
    CONTROLLED_LANDING = "controlled_landing"
    SUCCESSOR_READY = "successor_ready"


_PHASE_OWNER = {
    TeamSkillPhase.LEAD_PASS: SoccerRole.PASSER,
    TeamSkillPhase.RUNNING_INTERCEPT: SoccerRole.SHOOTER,
    TeamSkillPhase.STRIKE: SoccerRole.SHOOTER,
    TeamSkillPhase.GLOVE_SAVE: SoccerRole.GOALKEEPER,
    TeamSkillPhase.CONTROLLED_LANDING: SoccerRole.GOALKEEPER,
    TeamSkillPhase.SUCCESSOR_READY: SoccerRole.GOALKEEPER,
}


class GrowthPartition(StrEnum):
    DISCOVERY = "DISCOVERY"
    HOLDOUT = "HOLDOUT"


class TeamFailureCode(StrEnum):
    PASS_BEHIND_RUNNER = "pass_behind_runner"
    PASS_SPEED_MISMATCH = "pass_speed_mismatch"
    STOPPED_BEFORE_CONTACT = "stopped_before_contact"
    MISSED_INTERCEPT = "missed_intercept"
    SHOT_NOT_IN_TARGET_REGION = "shot_not_in_target_region"
    POST_SHOT_UNSTABLE = "post_shot_unstable"
    MISSED_GLOVE = "missed_glove"
    UNSAFE_LANDING = "unsafe_landing"
    SECOND_SAVE_NOT_READY = "second_save_not_ready"


_FAILURE_OWNER = {
    TeamFailureCode.PASS_BEHIND_RUNNER: SoccerRole.PASSER,
    TeamFailureCode.PASS_SPEED_MISMATCH: SoccerRole.PASSER,
    TeamFailureCode.STOPPED_BEFORE_CONTACT: SoccerRole.SHOOTER,
    TeamFailureCode.MISSED_INTERCEPT: SoccerRole.SHOOTER,
    TeamFailureCode.SHOT_NOT_IN_TARGET_REGION: SoccerRole.SHOOTER,
    TeamFailureCode.POST_SHOT_UNSTABLE: SoccerRole.SHOOTER,
    TeamFailureCode.MISSED_GLOVE: SoccerRole.GOALKEEPER,
    TeamFailureCode.UNSAFE_LANDING: SoccerRole.GOALKEEPER,
    TeamFailureCode.SECOND_SAVE_NOT_READY: SoccerRole.GOALKEEPER,
}


@dataclass(frozen=True)
class RoleGenerationBinding:
    """One immutable player artifact in a team episode."""

    role: SoccerRole
    agent_id: str
    artifact_hash: str
    parent_artifact_hash: str
    generation: int
    schema_version: str = "rosclaw_soccer.role_generation_binding.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.role, SoccerRole):
            raise ValueError("role generation binding has an unknown role")
        if not _IDENTIFIER.fullmatch(self.agent_id):
            raise ValueError("role generation agent id is invalid")
        _content_hash(self.artifact_hash, "role artifact")
        _content_hash(self.parent_artifact_hash, "role parent artifact")
        if not 1 <= self.generation <= 1_000_000:
            raise ValueError("role generation must be in [1, 1000000]")

    @property
    def binding_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        return value


@dataclass(frozen=True)
class PhaseScore:
    """Physics-derived score for one skill handoff, never a pixel score."""

    phase: TeamSkillPhase
    role: SoccerRole
    policy_artifact_hash: str
    source_evidence_hash: str
    success: bool
    quality: float
    successor_value: float
    safety_cost: float
    event_start_sec: float
    event_end_sec: float
    strict_replay: bool = True
    pixels_used_for_scoring: bool = False
    schema_version: str = "rosclaw_soccer.team_phase_score.v1"

    def __post_init__(self) -> None:
        if _PHASE_OWNER.get(self.phase) is not self.role:
            raise ValueError("team phase is assigned to the wrong role")
        _content_hash(self.policy_artifact_hash, "phase policy artifact")
        _content_hash(self.source_evidence_hash, "phase source evidence")
        for label in ("quality", "successor_value", "safety_cost"):
            value = getattr(self, label)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"phase {label} must be in [0, 1]")
        if not (
            math.isfinite(self.event_start_sec)
            and math.isfinite(self.event_end_sec)
            and 0.0 <= self.event_start_sec <= self.event_end_sec <= 120.0
        ):
            raise ValueError("phase event window is invalid")
        if self.pixels_used_for_scoring:
            raise ValueError("team phase scores cannot use rendered pixels")

    @property
    def score_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phase"] = self.phase.value
        value["role"] = self.role.value
        return value


@dataclass(frozen=True)
class AlternatingTeamEpisode:
    """One matched CPU MuJoCo episode with phase-local attribution."""

    episode_id: str
    seed: int
    partition: GrowthPartition
    scenario_hash: str
    environment_hash: str
    trajectory_hash: str
    policies: tuple[RoleGenerationBinding, ...]
    phases: tuple[PhaseScore, ...]
    chain_success: bool
    strict_replay: bool = True
    activation_ceiling: str = "SIM_ONLY"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.alternating_team_episode.v1"

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.episode_id):
            raise ValueError("team episode id is invalid")
        if isinstance(self.seed, bool) or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("team episode seed is invalid")
        for value, label in (
            (self.scenario_hash, "scenario"),
            (self.environment_hash, "environment"),
            (self.trajectory_hash, "trajectory"),
        ):
            _content_hash(value, label)
        expected_roles = set(SoccerRole)
        if {item.role for item in self.policies} != expected_roles or len(self.policies) != 3:
            raise ValueError("team episode requires exactly one policy per role")
        if {item.phase for item in self.phases} != set(TeamSkillPhase):
            raise ValueError("team episode requires exactly one score per skill phase")
        policy_by_role = {item.role: item for item in self.policies}
        if any(
            phase.policy_artifact_hash != policy_by_role[phase.role].artifact_hash
            for phase in self.phases
        ):
            raise ValueError("phase score is not bound to the executed role policy")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.physics_authority != "CPU_MUJOCO"
            or self.hardware_command_sent
        ):
            raise ValueError("team episode must remain SIM_ONLY CPU MuJoCo")

    @property
    def promotion_eligible(self) -> bool:
        return bool(
            self.strict_replay
            and all(item.strict_replay for item in self.phases)
            and all(item.safety_cost == 0.0 for item in self.phases)
        )

    @property
    def episode_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def policy(self, role: SoccerRole) -> RoleGenerationBinding:
        return next(item for item in self.policies if item.role is role)

    def phase(self, phase: TeamSkillPhase) -> PhaseScore:
        return next(item for item in self.phases if item.phase is phase)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "partition": self.partition.value,
            "scenario_hash": self.scenario_hash,
            "environment_hash": self.environment_hash,
            "trajectory_hash": self.trajectory_hash,
            "policies": [item.to_dict() for item in self.policies],
            "phases": [item.to_dict() for item in self.phases],
            "chain_success": self.chain_success,
            "strict_replay": self.strict_replay,
            "promotion_eligible": self.promotion_eligible,
            "activation_ceiling": self.activation_ceiling,
            "physics_authority": self.physics_authority,
            "hardware_command_sent": self.hardware_command_sent,
        }


@dataclass(frozen=True)
class AlternatingGrowthGateConfig:
    minimum_seeds: int = 2
    minimum_quality_improvement: float = 0.01
    minimum_successor_improvement: float = 0.005
    maximum_phase_regression: float = 0.01
    maximum_chain_success_regression: float = 0.0
    schema_version: str = "rosclaw_soccer.alternating_growth_gate_config.v1"

    def __post_init__(self) -> None:
        if not 2 <= self.minimum_seeds <= 1000:
            raise ValueError("alternating growth minimum seeds must be in [2, 1000]")
        for label in (
            "minimum_quality_improvement",
            "minimum_successor_improvement",
            "maximum_phase_regression",
            "maximum_chain_success_regression",
        ):
            value = getattr(self, label)
            if not math.isfinite(value) or not 0.0 <= value <= 0.25:
                raise ValueError(f"{label} must be in [0, 0.25]")


@dataclass(frozen=True)
class PhaseDelta:
    phase: TeamSkillPhase
    role: SoccerRole
    parent_success_rate: float
    candidate_success_rate: float
    quality_delta: float
    successor_delta: float
    safety_cost_max: float
    passed: bool
    schema_version: str = "rosclaw_soccer.phase_delta.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phase"] = self.phase.value
        value["role"] = self.role.value
        return value


@dataclass(frozen=True)
class AlternatingGrowthDecision:
    passed: bool
    plastic_role: SoccerRole
    seed_count: int
    phase_deltas: tuple[PhaseDelta, ...]
    parent_chain_success_rate: float
    candidate_chain_success_rate: float
    promoted_policy: RoleGenerationBinding | None
    reasons: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.alternating_growth_decision.v1"

    @property
    def decision_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "plastic_role": self.plastic_role.value,
            "seed_count": self.seed_count,
            "phase_deltas": [item.to_dict() for item in self.phase_deltas],
            "parent_chain_success_rate": self.parent_chain_success_rate,
            "candidate_chain_success_rate": self.candidate_chain_success_rate,
            "promoted_policy": (
                None if self.promoted_policy is None else self.promoted_policy.to_dict()
            ),
            "reasons": list(self.reasons),
            "activation_ceiling": self.activation_ceiling,
            "hardware_authorized": self.hardware_authorized,
        }


@dataclass(frozen=True)
class AlternatingGrowthRoundDecision:
    passed: bool
    plastic_role: SoccerRole
    discovery: AlternatingGrowthDecision
    holdout: AlternatingGrowthDecision
    promoted_policy: RoleGenerationBinding | None
    reasons: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.alternating_growth_round_decision.v1"

    @property
    def round_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "plastic_role": self.plastic_role.value,
            "discovery": self.discovery.to_dict(),
            "holdout": self.holdout.to_dict(),
            "promoted_policy": (
                None if self.promoted_policy is None else self.promoted_policy.to_dict()
            ),
            "reasons": list(self.reasons),
            "activation_ceiling": self.activation_ceiling,
            "hardware_authorized": self.hardware_authorized,
        }


def evaluate_alternating_growth(
    *,
    parent: tuple[AlternatingTeamEpisode, ...],
    candidate: tuple[AlternatingTeamEpisode, ...],
    plastic_role: SoccerRole,
    config: AlternatingGrowthGateConfig | None = None,
) -> AlternatingGrowthDecision:
    """Evaluate one changed role while its two teammates remain immutable."""

    active = config or AlternatingGrowthGateConfig()
    left = _episode_suite(parent, "parent", active.minimum_seeds)
    right = _episode_suite(candidate, "candidate", active.minimum_seeds)
    if set(left) != set(right):
        raise ValueError("alternating growth suites must use identical seeds")
    reasons: list[str] = []
    for seed in sorted(left):
        parent_episode = left[seed]
        candidate_episode = right[seed]
        if parent_episode.partition is not candidate_episode.partition:
            raise ValueError("alternating growth seed pair changed partition")
        if parent_episode.scenario_hash != candidate_episode.scenario_hash:
            raise ValueError("alternating growth seed pair changed scenario")
        if parent_episode.environment_hash != candidate_episode.environment_hash:
            raise ValueError("alternating growth seed pair changed environment")
        for role in SoccerRole:
            parent_policy = parent_episode.policy(role)
            candidate_policy = candidate_episode.policy(role)
            if role is plastic_role:
                if (
                    candidate_policy.agent_id != parent_policy.agent_id
                    or candidate_policy.parent_artifact_hash != parent_policy.artifact_hash
                    or candidate_policy.generation != parent_policy.generation + 1
                    or candidate_policy.artifact_hash == parent_policy.artifact_hash
                ):
                    raise ValueError("plastic role lineage is not a one-generation child")
            elif candidate_policy != parent_policy:
                raise ValueError("non-plastic teammate changed during an alternating round")
        if not parent_episode.promotion_eligible:
            reasons.append(f"parent_seed_{seed}_ineligible")
        if not candidate_episode.promotion_eligible:
            reasons.append(f"candidate_seed_{seed}_ineligible")

    phase_deltas: list[PhaseDelta] = []
    for phase in TeamSkillPhase:
        owner = _PHASE_OWNER[phase]
        parent_scores = tuple(left[seed].phase(phase) for seed in sorted(left))
        candidate_scores = tuple(right[seed].phase(phase) for seed in sorted(right))
        parent_success = _mean(tuple(float(item.success) for item in parent_scores))
        candidate_success = _mean(tuple(float(item.success) for item in candidate_scores))
        quality_delta = _mean(
            tuple(
                candidate_score.quality - parent_score.quality
                for parent_score, candidate_score in zip(
                    parent_scores, candidate_scores, strict=True
                )
            )
        )
        successor_delta = _mean(
            tuple(
                candidate_score.successor_value - parent_score.successor_value
                for parent_score, candidate_score in zip(
                    parent_scores, candidate_scores, strict=True
                )
            )
        )
        safety_max = max(item.safety_cost for item in candidate_scores)
        if owner is plastic_role:
            passed = bool(
                candidate_success >= parent_success
                and quality_delta >= active.minimum_quality_improvement
                and successor_delta >= active.minimum_successor_improvement
                and safety_max == 0.0
            )
        else:
            passed = bool(
                candidate_success >= parent_success
                and quality_delta >= -active.maximum_phase_regression
                and successor_delta >= -active.maximum_phase_regression
                and safety_max == 0.0
            )
        if not passed:
            reasons.append(f"{phase.value}_gate_failed")
        phase_deltas.append(
            PhaseDelta(
                phase=phase,
                role=owner,
                parent_success_rate=parent_success,
                candidate_success_rate=candidate_success,
                quality_delta=quality_delta,
                successor_delta=successor_delta,
                safety_cost_max=safety_max,
                passed=passed,
            )
        )
    parent_chain = _mean(tuple(float(item.chain_success) for item in left.values()))
    candidate_chain = _mean(tuple(float(item.chain_success) for item in right.values()))
    if candidate_chain < parent_chain - active.maximum_chain_success_regression:
        reasons.append("team_chain_success_regressed")
    unique_reasons = tuple(dict.fromkeys(reasons))
    promoted = right[min(right)].policy(plastic_role) if not unique_reasons else None
    return AlternatingGrowthDecision(
        passed=not unique_reasons,
        plastic_role=plastic_role,
        seed_count=len(left),
        phase_deltas=tuple(phase_deltas),
        parent_chain_success_rate=parent_chain,
        candidate_chain_success_rate=candidate_chain,
        promoted_policy=promoted,
        reasons=unique_reasons,
    )


def evaluate_alternating_growth_round(
    *,
    discovery_parent: tuple[AlternatingTeamEpisode, ...],
    discovery_candidate: tuple[AlternatingTeamEpisode, ...],
    holdout_parent: tuple[AlternatingTeamEpisode, ...],
    holdout_candidate: tuple[AlternatingTeamEpisode, ...],
    plastic_role: SoccerRole,
    config: AlternatingGrowthGateConfig | None = None,
) -> AlternatingGrowthRoundDecision:
    """Require the same role-local child to pass disjoint sealed holdout."""

    discovery_seeds = {item.seed for item in (*discovery_parent, *discovery_candidate)}
    holdout_seeds = {item.seed for item in (*holdout_parent, *holdout_candidate)}
    if discovery_seeds & holdout_seeds:
        raise ValueError("alternating discovery and holdout seeds must be disjoint")
    for suite in (discovery_parent, discovery_candidate):
        if any(item.partition is not GrowthPartition.DISCOVERY for item in suite):
            raise ValueError("discovery suite contains a non-discovery episode")
    for suite in (holdout_parent, holdout_candidate):
        if any(item.partition is not GrowthPartition.HOLDOUT for item in suite):
            raise ValueError("holdout suite contains a non-holdout episode")
    discovery = evaluate_alternating_growth(
        parent=discovery_parent,
        candidate=discovery_candidate,
        plastic_role=plastic_role,
        config=config,
    )
    holdout = evaluate_alternating_growth(
        parent=holdout_parent,
        candidate=holdout_candidate,
        plastic_role=plastic_role,
        config=config,
    )
    if discovery_candidate[0].policy(plastic_role) != holdout_candidate[0].policy(plastic_role):
        raise ValueError("discovery and holdout evaluated different role candidates")
    reasons = tuple(
        reason
        for passed, reason in (
            (discovery.passed, "discovery_alternating_gate_failed"),
            (holdout.passed, "holdout_alternating_gate_failed"),
        )
        if not passed
    )
    return AlternatingGrowthRoundDecision(
        passed=not reasons,
        plastic_role=plastic_role,
        discovery=discovery,
        holdout=holdout,
        promoted_policy=(holdout.promoted_policy if not reasons else None),
        reasons=reasons,
    )


@dataclass(frozen=True)
class FailureMemoryRecord:
    """Role-private failure slice used to create development-only Dreams."""

    agent_id: str
    role: SoccerRole
    phase: TeamSkillPhase
    failure_code: TeamFailureCode
    source_evidence_hash: str
    snapshot_hash: str
    scenario_hash: str
    severity: float
    schema_version: str = "rosclaw_soccer.team_failure_memory_record.v1"

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.agent_id):
            raise ValueError("failure memory agent id is invalid")
        if _FAILURE_OWNER[self.failure_code] is not self.role:
            raise ValueError("failure memory is assigned to the wrong role")
        if _PHASE_OWNER[self.phase] is not self.role:
            raise ValueError("failure memory phase is assigned to the wrong role")
        for value, label in (
            (self.source_evidence_hash, "failure evidence"),
            (self.snapshot_hash, "failure snapshot"),
            (self.scenario_hash, "failure scenario"),
        ):
            _content_hash(value, label)
        if not math.isfinite(self.severity) or not 0.0 < self.severity <= 1.0:
            raise ValueError("failure severity must be in (0, 1]")

    @property
    def memory_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        value["phase"] = self.phase.value
        value["failure_code"] = self.failure_code.value
        return value


@dataclass(frozen=True)
class FailureConditionedDream:
    source_memory_hash: str
    role: SoccerRole
    failure_code: TeamFailureCode
    variant_index: int
    perturbations: tuple[tuple[str, float], ...]
    partition: GrowthPartition = GrowthPartition.DISCOVERY
    sealed_holdout_used: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.failure_conditioned_dream.v1"

    def __post_init__(self) -> None:
        _content_hash(self.source_memory_hash, "dream source memory")
        if _FAILURE_OWNER[self.failure_code] is not self.role:
            raise ValueError("dream failure is assigned to the wrong role")
        if not 0 <= self.variant_index <= 1024:
            raise ValueError("dream variant index is invalid")
        names = tuple(name for name, _ in self.perturbations)
        if not self.perturbations or len(set(names)) != len(names):
            raise ValueError("dream perturbations must be non-empty and unique")
        if any(not _IDENTIFIER.fullmatch(name) for name in names):
            raise ValueError("dream perturbation name is invalid")
        if any(not math.isfinite(value) or abs(value) > 1.0 for _, value in self.perturbations):
            raise ValueError("dream perturbation exceeds its normalized bound")
        if (
            self.partition is not GrowthPartition.DISCOVERY
            or self.sealed_holdout_used
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("failure-conditioned Dreams cannot consume sealed holdout")

    @property
    def dream_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        value["failure_code"] = self.failure_code.value
        value["partition"] = self.partition.value
        return value


_DREAM_TEMPLATES: dict[TeamFailureCode, tuple[tuple[str, float], ...]] = {
    TeamFailureCode.PASS_BEHIND_RUNNER: (
        ("receiver_speed_scale", 0.10),
        ("receiver_heading_delta", 0.08),
        ("arrival_time_delta", -0.05),
    ),
    TeamFailureCode.PASS_SPEED_MISMATCH: (
        ("pass_speed_scale", 0.12),
        ("ball_friction_delta", 0.08),
        ("receiver_speed_scale", -0.08),
    ),
    TeamFailureCode.STOPPED_BEFORE_CONTACT: (
        ("approach_speed_scale", 0.12),
        ("pass_speed_scale", 0.08),
        ("intercept_angle_delta", 0.10),
    ),
    TeamFailureCode.MISSED_INTERCEPT: (
        ("ball_arrival_time_delta", 0.06),
        ("ball_lateral_offset", 0.10),
        ("observation_delay_delta", 0.05),
    ),
    TeamFailureCode.SHOT_NOT_IN_TARGET_REGION: (
        ("target_height_delta", 0.10),
        ("target_lateral_delta", -0.10),
        ("contact_phase_delta", 0.06),
    ),
    TeamFailureCode.POST_SHOT_UNSTABLE: (
        ("support_friction_delta", -0.08),
        ("follow_through_scale", 0.10),
        ("actuator_delay_delta", 0.05),
    ),
    TeamFailureCode.MISSED_GLOVE: (
        ("impact_time_delta", -0.05),
        ("impact_lateral_delta", 0.10),
        ("impact_height_delta", 0.10),
    ),
    TeamFailureCode.UNSAFE_LANDING: (
        ("root_angular_momentum_delta", 0.10),
        ("ground_friction_delta", -0.08),
        ("impact_impulse_delta", 0.08),
    ),
    TeamFailureCode.SECOND_SAVE_NOT_READY: (
        ("second_threat_delay_delta", -0.10),
        ("second_ball_lateral_delta", 0.10),
        ("recovery_friction_delta", -0.05),
    ),
}


def build_failure_conditioned_dreams(
    memory: FailureMemoryRecord, *, variants: int = 5
) -> tuple[FailureConditionedDream, ...]:
    """Expand an exact failure with symmetric, bounded development variants."""

    if not 2 <= variants <= 9:
        raise ValueError("failure-conditioned Dream variants must be in [2, 9]")
    template = _DREAM_TEMPLATES[memory.failure_code]
    center = (variants - 1) / 2.0
    scale_denominator = max(center, 1.0)
    dreams = []
    for index in range(variants):
        scale = (index - center) / scale_denominator
        perturbations = tuple((name, value * scale) for name, value in template)
        dreams.append(
            FailureConditionedDream(
                source_memory_hash=memory.memory_hash,
                role=memory.role,
                failure_code=memory.failure_code,
                variant_index=index,
                perturbations=perturbations,
            )
        )
    return tuple(dreams)


@dataclass(frozen=True)
class CurriculumCell:
    cell_id: str
    role: SoccerRole
    difficulty: float
    attempts: int
    successes: int
    evidence_hash: str | None
    source: str = "development"
    schema_version: str = "rosclaw_soccer.team_curriculum_cell.v1"

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.cell_id):
            raise ValueError("curriculum cell id is invalid")
        if not math.isfinite(self.difficulty) or not 0.0 <= self.difficulty <= 1.0:
            raise ValueError("curriculum difficulty must be in [0, 1]")
        if not 0 <= self.successes <= self.attempts <= 1_000_000:
            raise ValueError("curriculum attempts/successes are invalid")
        if self.evidence_hash is not None:
            _content_hash(self.evidence_hash, "curriculum evidence")
        allowed_sources = {
            "development",
            "recent_failure",
            "historical",
            "nightmare",
            "social",
        }
        if self.source not in allowed_sources:
            raise ValueError("curriculum source is invalid")

    @property
    def success_rate(self) -> float | None:
        return None if self.attempts == 0 else self.successes / self.attempts

    @property
    def route(self) -> str:
        rate = self.success_rate
        if rate is None:
            return "PROBE_UNTESTED"
        if self.source == "nightmare":
            return "NIGHTMARE"
        if self.source == "social":
            return "SOCIAL_TEACHER"
        if 0.30 <= rate <= 0.70:
            return "CAPABILITY_FRONTIER"
        if rate < 0.30:
            return "RECENT_FAILURE"
        return "HISTORICAL_ANCHOR"


@dataclass(frozen=True)
class CurriculumPriority:
    cell_id: str
    role: SoccerRole
    route: str
    priority: float
    reason: str
    schema_version: str = "rosclaw_soccer.team_curriculum_priority.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        return value


def prioritize_team_curriculum(
    cells: tuple[CurriculumCell, ...],
) -> tuple[CurriculumPriority, ...]:
    """Prioritize untested prerequisites and the 30-70% learning frontier."""

    if not cells or len({item.cell_id for item in cells}) != len(cells):
        raise ValueError("team curriculum requires unique non-empty cells")
    route_base = {
        "PROBE_UNTESTED": 1.00,
        "CAPABILITY_FRONTIER": 0.90,
        "RECENT_FAILURE": 0.75,
        "HISTORICAL_ANCHOR": 0.40,
        "NIGHTMARE": 0.30,
        "SOCIAL_TEACHER": 0.30,
    }
    priorities = []
    for cell in cells:
        rate = cell.success_rate
        frontier_bonus = 0.0 if rate is None else max(0.0, 0.20 - abs(rate - 0.50) * 0.40)
        priority = min(1.0, route_base[cell.route] + 0.05 * cell.difficulty + frontier_bonus)
        priorities.append(
            CurriculumPriority(
                cell_id=cell.cell_id,
                role=cell.role,
                route=cell.route,
                priority=priority,
                reason=(
                    "establish a fresh physics baseline before learning"
                    if rate is None
                    else f"success_rate={rate:.3f}; difficulty={cell.difficulty:.3f}"
                ),
            )
        )
    difficulty_by_id = {item.cell_id: item.difficulty for item in cells}
    return tuple(
        sorted(
            priorities,
            key=lambda item: (
                -item.priority,
                -difficulty_by_id[item.cell_id],
                item.cell_id,
            ),
        )
    )


def _episode_suite(
    episodes: tuple[AlternatingTeamEpisode, ...], label: str, minimum: int
) -> dict[int, AlternatingTeamEpisode]:
    if len(episodes) < minimum:
        raise ValueError(f"{label} alternating suite has insufficient seeds")
    by_seed = {item.seed: item for item in episodes}
    if len(by_seed) != len(episodes):
        raise ValueError(f"{label} alternating suite seeds must be unique")
    return by_seed


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def _content_hash(value: str, label: str) -> None:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{label} must be a sha256 content hash")


__all__ = [
    "AlternatingGrowthDecision",
    "AlternatingGrowthGateConfig",
    "AlternatingGrowthRoundDecision",
    "AlternatingTeamEpisode",
    "CurriculumCell",
    "CurriculumPriority",
    "FailureConditionedDream",
    "FailureMemoryRecord",
    "GrowthPartition",
    "PhaseDelta",
    "PhaseScore",
    "RoleGenerationBinding",
    "TeamFailureCode",
    "TeamSkillPhase",
    "build_failure_conditioned_dreams",
    "evaluate_alternating_growth",
    "evaluate_alternating_growth_round",
    "prioritize_team_curriculum",
]
