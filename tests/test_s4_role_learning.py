from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from rosclaw_soccer.growth.role_learning import (
    RoleEpisodeOutcome,
    RolePolicyBinding,
    SharedWorldTeamEpisode,
    SoccerRole,
    evaluate_joint_growth,
)


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _policy(role: SoccerRole, generation: int) -> RolePolicyBinding:
    version = f"league.{role.value}.v{generation}"
    return RolePolicyBinding(
        agent_id=f"claw7.{role.value}",
        role=role,
        policy_version=version,
        artifact_hash=_hash(version),
        parent_artifact_hash=_hash(f"league.{role.value}.v{generation - 1}"),
        observation_contract_hash=_hash(f"observation.{role.value}.v1"),
        action_contract_hash=_hash(f"action.{role.value}.v1"),
        generation=generation,
    )


def _episode(seed: int, generation: int, improvement: float) -> SharedWorldTeamEpisode:
    policies = tuple(_policy(role, generation) for role in SoccerRole)
    outcomes = tuple(
        RoleEpisodeOutcome(
            agent_id=policy.agent_id,
            role=policy.role,
            policy_artifact_hash=policy.artifact_hash,
            action_trace_hash=_hash(f"action.{seed}.{policy.role}.{generation}"),
            counterfactual_evidence_hash=_hash(f"counterfactual.{seed}.{policy.role}.{generation}"),
            counterfactual_parent_artifact_hash=policy.parent_artifact_hash,
            counterfactual_scenario_hash=_hash(f"scenario.{seed}"),
            counterfactual_environment_hash=_hash("environment.v1"),
            counterfactual_seed=seed,
            individual_reward=0.40 + improvement - 0.01 * seed,
            side_reward=0.55 + improvement - 0.01 * seed,
            counterfactual_side_reward=0.30,
            stability_score=0.85 + min(improvement, 0.05),
            safety_cost=0.0,
        )
        for policy in policies
    )
    return SharedWorldTeamEpisode(
        episode_id=f"league.seed-{seed}.generation-{generation}",
        scenario_hash=_hash(f"scenario.{seed}"),
        environment_hash=_hash("environment.v1"),
        trajectory_hash=_hash(f"trajectory.{seed}.{generation}"),
        seed=seed,
        policies=policies,
        outcomes=outcomes,
        strict_replay=True,
        rolling_authenticity_passed=True,
        physical_event_order_passed=True,
    )


def _suite(generation: int, improvement: float) -> tuple[SharedWorldTeamEpisode, ...]:
    return tuple(_episode(seed, generation, improvement) for seed in (11, 17, 23))


def test_joint_growth_accepts_three_independent_improved_role_policies() -> None:
    decision = evaluate_joint_growth(
        parent=_suite(1, 0.0),
        candidate=_suite(2, 0.05),
    )

    assert decision.passed
    assert decision.changed_roles == tuple(SoccerRole)
    assert not decision.hardware_authorized
    assert all(metric.passed for metric in decision.role_metrics)


def test_joint_growth_rejects_unchanged_goalkeeper_even_if_attack_improves() -> None:
    parent = _suite(1, 0.0)
    candidate = list(_suite(2, 0.05))
    for index, episode in enumerate(candidate):
        policies = tuple(
            replace(
                parent[index].policy(SoccerRole.GOALKEEPER),
                parent_artifact_hash=parent[index].policy(SoccerRole.GOALKEEPER).artifact_hash,
                generation=2,
            )
            if item.role is SoccerRole.GOALKEEPER
            else item
            for item in episode.policies
        )
        outcomes = tuple(
            replace(
                item,
                policy_artifact_hash=next(
                    policy.artifact_hash for policy in policies if policy.role is item.role
                ),
                counterfactual_parent_artifact_hash=next(
                    policy.parent_artifact_hash for policy in policies if policy.role is item.role
                ),
            )
            for item in episode.outcomes
        )
        candidate[index] = replace(episode, policies=policies, outcomes=outcomes)

    decision = evaluate_joint_growth(parent=parent, candidate=tuple(candidate))

    assert not decision.passed
    assert "goalkeeper_growth_gate_failed" in decision.reasons
    assert "not_all_role_policies_changed" in decision.reasons
    assert "candidate_seed_11_ineligible" in decision.reasons


def test_joint_growth_rejects_a_candidate_with_role_safety_cost() -> None:
    candidate = list(_suite(2, 0.05))
    outcome = candidate[0].outcome(SoccerRole.PASSER)
    outcomes = tuple(
        replace(item, safety_cost=0.2) if item.role is SoccerRole.PASSER else item
        for item in candidate[0].outcomes
    )
    assert outcome.safety_cost == 0.0
    candidate[0] = replace(candidate[0], outcomes=outcomes)

    decision = evaluate_joint_growth(parent=_suite(1, 0.0), candidate=tuple(candidate))

    assert not decision.passed
    assert "candidate_seed_11_ineligible" in decision.reasons
    assert "passer_growth_gate_failed" in decision.reasons


def test_joint_growth_rejects_a_free_rider_without_counterfactual_contribution() -> None:
    candidate = list(_suite(2, 0.05))
    for index, episode in enumerate(candidate):
        outcomes = tuple(
            replace(
                item,
                counterfactual_side_reward=item.side_reward,
            )
            if item.role is SoccerRole.PASSER
            else item
            for item in episode.outcomes
        )
        candidate[index] = replace(episode, outcomes=outcomes)

    decision = evaluate_joint_growth(parent=_suite(1, 0.0), candidate=tuple(candidate))

    assert not decision.passed
    assert "passer_growth_gate_failed" in decision.reasons


def test_team_episode_fails_closed_without_every_role() -> None:
    episode = _episode(11, 1, 0.0)
    with pytest.raises(ValueError, match="one independent policy for every role"):
        replace(episode, policies=episode.policies[:-1])


def test_team_episode_rejects_pixels_and_non_sim_authority() -> None:
    episode = _episode(11, 1, 0.0)
    with pytest.raises(ValueError, match="simulation evidence boundary"):
        replace(episode, pixels_used_for_scoring=True)
    with pytest.raises(ValueError, match="SIM_ONLY CPU MuJoCo"):
        replace(episode, activation_ceiling="REAL")


def test_team_episode_rejects_counterfactual_from_a_different_world() -> None:
    episode = _episode(11, 1, 0.0)
    outcomes = tuple(
        replace(item, counterfactual_seed=12) if item.role is SoccerRole.GOALKEEPER else item
        for item in episode.outcomes
    )

    with pytest.raises(ValueError, match="counterfactual seed"):
        replace(episode, outcomes=outcomes)


def test_joint_growth_rejects_contract_or_generation_drift() -> None:
    candidate = list(_suite(2, 0.05))
    policies = tuple(
        replace(item, action_contract_hash=_hash("action.goalkeeper.v2"))
        if item.role is SoccerRole.GOALKEEPER
        else item
        for item in candidate[0].policies
    )
    candidate[0] = replace(candidate[0], policies=policies)
    with pytest.raises(ValueError, match="action contract changed"):
        evaluate_joint_growth(parent=_suite(1, 0.0), candidate=tuple(candidate))

    candidate = list(_suite(2, 0.05))
    policies = tuple(
        replace(item, generation=3) if item.role is SoccerRole.PASSER else item
        for item in candidate[0].policies
    )
    candidate[0] = replace(candidate[0], policies=policies)
    with pytest.raises(ValueError, match="generation must increment"):
        evaluate_joint_growth(parent=_suite(1, 0.0), candidate=tuple(candidate))
