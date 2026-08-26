from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw_soccer.growth.alternating_team_growth import (
    AlternatingTeamEpisode,
    CurriculumCell,
    FailureMemoryRecord,
    GrowthPartition,
    PhaseScore,
    RoleGenerationBinding,
    TeamFailureCode,
    TeamSkillPhase,
    build_failure_conditioned_dreams,
    evaluate_alternating_growth,
    evaluate_alternating_growth_round,
    prioritize_team_curriculum,
)
from rosclaw_soccer.growth.role_learning import SoccerRole
from rosclaw_soccer.training.team_growth_ledger import (
    build_regulation_team_growth_ledger,
    validate_regulation_team_growth_ledger,
)

_EVIDENCE = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s93-regulation-dead-corner-save-v4/evidence.json"
)


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _policies() -> tuple[RoleGenerationBinding, ...]:
    return tuple(
        RoleGenerationBinding(
            role=role,
            agent_id=f"soccer.{role.value}",
            artifact_hash=_hash(f"{role.value}.g1"),
            parent_artifact_hash=_hash(f"{role.value}.g0"),
            generation=1,
        )
        for role in SoccerRole
    )


def _episode(
    *,
    seed: int,
    partition: GrowthPartition,
    candidate_role: SoccerRole | None = None,
    quality_gain: float = 0.0,
    successor_gain: float = 0.0,
    teammate_regression: float = 0.0,
) -> AlternatingTeamEpisode:
    parent = _policies()
    policies = tuple(
        replace(
            item,
            artifact_hash=_hash(f"{item.role.value}.g2"),
            parent_artifact_hash=item.artifact_hash,
            generation=2,
        )
        if item.role is candidate_role
        else item
        for item in parent
    )
    policy_by_role = {item.role: item for item in policies}
    phases = []
    owners = {
        TeamSkillPhase.LEAD_PASS: SoccerRole.PASSER,
        TeamSkillPhase.RUNNING_INTERCEPT: SoccerRole.SHOOTER,
        TeamSkillPhase.STRIKE: SoccerRole.SHOOTER,
        TeamSkillPhase.GLOVE_SAVE: SoccerRole.GOALKEEPER,
        TeamSkillPhase.CONTROLLED_LANDING: SoccerRole.GOALKEEPER,
        TeamSkillPhase.SUCCESSOR_READY: SoccerRole.GOALKEEPER,
    }
    for index, (phase, role) in enumerate(owners.items()):
        gain = quality_gain if role is candidate_role else -teammate_regression
        next_gain = successor_gain if role is candidate_role else -teammate_regression
        phases.append(
            PhaseScore(
                phase=phase,
                role=role,
                policy_artifact_hash=policy_by_role[role].artifact_hash,
                source_evidence_hash=_hash(f"evidence.{seed}.{phase.value}"),
                success=True,
                quality=0.70 + gain,
                successor_value=0.75 + next_gain,
                safety_cost=0.0,
                event_start_sec=float(index),
                event_end_sec=float(index + 1),
            )
        )
    return AlternatingTeamEpisode(
        episode_id=f"episode.{partition.value.lower()}.{seed}",
        seed=seed,
        partition=partition,
        scenario_hash=_hash(f"scenario.{seed}"),
        environment_hash=_hash("environment"),
        trajectory_hash=_hash(
            f"trajectory.{partition.value}.{seed}.{candidate_role}.{quality_gain}"
        ),
        policies=policies,
        phases=tuple(phases),
        chain_success=True,
    )


def _suite(
    *,
    seeds: tuple[int, ...],
    partition: GrowthPartition,
    candidate_role: SoccerRole | None = None,
    quality_gain: float = 0.0,
    successor_gain: float = 0.0,
    teammate_regression: float = 0.0,
) -> tuple[AlternatingTeamEpisode, ...]:
    return tuple(
        _episode(
            seed=seed,
            partition=partition,
            candidate_role=candidate_role,
            quality_gain=quality_gain,
            successor_gain=successor_gain,
            teammate_regression=teammate_regression,
        )
        for seed in seeds
    )


def test_alternating_growth_changes_one_role_and_rewards_successor_quality() -> None:
    parent = _suite(seeds=(11, 17), partition=GrowthPartition.DISCOVERY)
    candidate = _suite(
        seeds=(11, 17),
        partition=GrowthPartition.DISCOVERY,
        candidate_role=SoccerRole.SHOOTER,
        quality_gain=0.02,
        successor_gain=0.01,
    )

    decision = evaluate_alternating_growth(
        parent=parent,
        candidate=candidate,
        plastic_role=SoccerRole.SHOOTER,
    )

    assert decision.passed
    assert decision.promoted_policy is not None
    assert decision.promoted_policy.role is SoccerRole.SHOOTER
    assert all(item.passed for item in decision.phase_deltas)
    assert not decision.hardware_authorized


def test_alternating_growth_rejects_a_changed_frozen_teammate() -> None:
    parent = _suite(seeds=(11, 17), partition=GrowthPartition.DISCOVERY)
    candidate = list(
        _suite(
            seeds=(11, 17),
            partition=GrowthPartition.DISCOVERY,
            candidate_role=SoccerRole.SHOOTER,
            quality_gain=0.02,
            successor_gain=0.01,
        )
    )
    first = candidate[0]
    policies = tuple(
        replace(item, artifact_hash=_hash("mutated-passer"))
        if item.role is SoccerRole.PASSER
        else item
        for item in first.policies
    )
    phases = tuple(
        replace(item, policy_artifact_hash=_hash("mutated-passer"))
        if item.role is SoccerRole.PASSER
        else item
        for item in first.phases
    )
    candidate[0] = replace(first, policies=policies, phases=phases)

    with pytest.raises(ValueError, match="non-plastic teammate changed"):
        evaluate_alternating_growth(
            parent=parent,
            candidate=tuple(candidate),
            plastic_role=SoccerRole.SHOOTER,
        )


def test_alternating_growth_rejects_pretty_action_without_successor_gain() -> None:
    decision = evaluate_alternating_growth(
        parent=_suite(seeds=(11, 17), partition=GrowthPartition.DISCOVERY),
        candidate=_suite(
            seeds=(11, 17),
            partition=GrowthPartition.DISCOVERY,
            candidate_role=SoccerRole.GOALKEEPER,
            quality_gain=0.04,
            successor_gain=0.0,
        ),
        plastic_role=SoccerRole.GOALKEEPER,
    )

    assert not decision.passed
    assert "successor_ready_gate_failed" in decision.reasons
    assert decision.promoted_policy is None


def test_alternating_round_requires_disjoint_holdout_and_same_child() -> None:
    discovery_parent = _suite(seeds=(11, 17), partition=GrowthPartition.DISCOVERY)
    discovery_candidate = _suite(
        seeds=(11, 17),
        partition=GrowthPartition.DISCOVERY,
        candidate_role=SoccerRole.PASSER,
        quality_gain=0.02,
        successor_gain=0.01,
    )
    holdout_parent = _suite(seeds=(23, 29), partition=GrowthPartition.HOLDOUT)
    holdout_candidate = _suite(
        seeds=(23, 29),
        partition=GrowthPartition.HOLDOUT,
        candidate_role=SoccerRole.PASSER,
        quality_gain=0.02,
        successor_gain=0.01,
    )
    decision = evaluate_alternating_growth_round(
        discovery_parent=discovery_parent,
        discovery_candidate=discovery_candidate,
        holdout_parent=holdout_parent,
        holdout_candidate=holdout_candidate,
        plastic_role=SoccerRole.PASSER,
    )

    assert decision.passed
    assert decision.promoted_policy is not None
    with pytest.raises(ValueError, match="seeds must be disjoint"):
        evaluate_alternating_growth_round(
            discovery_parent=discovery_parent,
            discovery_candidate=discovery_candidate,
            holdout_parent=_suite(seeds=(17, 29), partition=GrowthPartition.HOLDOUT),
            holdout_candidate=_suite(
                seeds=(17, 29),
                partition=GrowthPartition.HOLDOUT,
                candidate_role=SoccerRole.PASSER,
                quality_gain=0.02,
                successor_gain=0.01,
            ),
            plastic_role=SoccerRole.PASSER,
        )


def test_failure_memory_dreams_are_role_private_symmetric_and_no_holdout() -> None:
    memory = FailureMemoryRecord(
        agent_id="soccer.goalkeeper",
        role=SoccerRole.GOALKEEPER,
        phase=TeamSkillPhase.GLOVE_SAVE,
        failure_code=TeamFailureCode.MISSED_GLOVE,
        source_evidence_hash=_hash("evidence"),
        snapshot_hash=_hash("snapshot"),
        scenario_hash=_hash("scenario"),
        severity=0.8,
    )
    dreams = build_failure_conditioned_dreams(memory, variants=5)

    assert len(dreams) == 5
    assert all(item.role is SoccerRole.GOALKEEPER for item in dreams)
    assert all(item.partition is GrowthPartition.DISCOVERY for item in dreams)
    assert all(not item.sealed_holdout_used for item in dreams)
    assert dreams[2].perturbations == (
        ("impact_time_delta", 0.0),
        ("impact_lateral_delta", 0.0),
        ("impact_height_delta", 0.0),
    )
    assert tuple(value for _, value in dreams[0].perturbations) == tuple(
        -value for _, value in dreams[-1].perturbations
    )


def test_curriculum_probes_untested_cells_before_replaying_mastered_anchors() -> None:
    cells = (
        CurriculumCell("mastered", SoccerRole.PASSER, 0.5, 20, 20, _hash("mastered")),
        CurriculumCell("frontier", SoccerRole.SHOOTER, 0.7, 20, 10, _hash("frontier")),
        CurriculumCell("untested", SoccerRole.GOALKEEPER, 0.9, 0, 0, None),
    )
    priorities = prioritize_team_curriculum(cells)

    assert tuple(item.cell_id for item in priorities) == ("untested", "frontier", "mastered")
    assert priorities[0].route == "PROBE_UNTESTED"
    assert priorities[1].route == "CAPABILITY_FRONTIER"


@pytest.mark.skipif(not _EVIDENCE.is_file(), reason="S93 frozen evidence unavailable")
def test_actual_s93_builds_an_honest_phase_ledger(tmp_path: Path) -> None:
    output = tmp_path / "team-growth-ledger.json"
    report = build_regulation_team_growth_ledger(
        evidence_path=_EVIDENCE,
        output_path=output,
        source_checkout=Path(__file__).parents[1],
    )

    assert report["status"] == "READY_FOR_ALTERNATING_GROWTH"
    assert len(report["episodes"]) == 2
    assert all(item["chain_success"] for item in report["episodes"])
    assert report["recommended_role_order"][0]["plastic_role"] == "passer"
    assert report["evidence_ceiling"]["fresh_training_performed"] is False
    assert validate_regulation_team_growth_ledger(output)["ledger_hash"].startswith("sha256:")

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["selection_semantics"]["plastic_roles_per_round"] = 3
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authority contract"):
        validate_regulation_team_growth_ledger(output)
