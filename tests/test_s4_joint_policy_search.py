from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from rosclaw_soccer.growth.joint_policy_search import (
    MirroredRoleProbe,
    RolePolicyVector,
    default_three_role_search_spaces,
    learn_joint_policy_generation,
)
from rosclaw_soccer.growth.role_learning import SoccerRole


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _parents() -> tuple[RolePolicyVector, ...]:
    values = {
        SoccerRole.PASSER: (-0.04, 0.10, -0.04, 0.75, 0.80, 0.03),
        SoccerRole.SHOOTER: (2.10, 0.085, 0.010, -0.065, 0.90, 0.055),
        SoccerRole.GOALKEEPER: (0.12, 1.35, 0.25, 0.06, 0.20, 0.05),
    }
    spaces = {item.role: item for item in default_three_role_search_spaces()}
    return tuple(
        RolePolicyVector(
            role=role,
            generation=4,
            parameter_names=spaces[role].parameter_names,
            values=values[role],
        )
        for role in SoccerRole
    )


def _probes(parents: tuple[RolePolicyVector, ...]) -> tuple[MirroredRoleProbe, ...]:
    probes = []
    direction = (1.0, -0.5, 0.25, 0.75, -0.25, 0.5)
    for role_index, parent in enumerate(parents):
        for probe_index, seed in enumerate((11, 17, 23, 29)):
            sign = -1.0 if probe_index % 2 else 1.0
            perturbation = tuple(sign * value for value in direction)
            advantage = sign * (0.10 + 0.01 * role_index)
            probes.append(
                MirroredRoleProbe(
                    role=parent.role,
                    seed=seed,
                    scenario_hash=_hash(f"scenario.{seed}"),
                    environment_hash=_hash("environment.v1"),
                    parent_artifact_hash=parent.artifact_hash,
                    perturbation=perturbation,
                    positive_growth_score=advantage,
                    negative_growth_score=-advantage,
                    positive_safety_cost=0.0,
                    negative_safety_cost=0.0,
                    positive_action_trace_hash=_hash(f"positive.{role_index}.{seed}"),
                    negative_action_trace_hash=_hash(f"negative.{role_index}.{seed}"),
                )
            )
    return tuple(probes)


def test_joint_search_learns_one_independent_candidate_per_role() -> None:
    parents = _parents()
    decision = learn_joint_policy_generation(
        parents=parents,
        spaces=default_three_role_search_spaces(),
        probes=_probes(parents),
    )

    assert decision.passed
    assert tuple(item.role for item in decision.candidates) == tuple(SoccerRole)
    assert all(item.generation == 5 for item in decision.candidates)
    assert all(
        candidate.artifact_hash != parent.artifact_hash
        for candidate, parent in zip(decision.candidates, parents, strict=True)
    )
    assert not decision.hardware_authorized


def test_joint_search_rejects_role_without_a_learning_signal() -> None:
    parents = _parents()
    probes = tuple(
        replace(item, positive_growth_score=0.0, negative_growth_score=0.0)
        if item.role is SoccerRole.GOALKEEPER
        else item
        for item in _probes(parents)
    )

    decision = learn_joint_policy_generation(
        parents=parents,
        spaces=default_three_role_search_spaces(),
        probes=probes,
    )

    assert not decision.passed
    assert decision.reasons == ("goalkeeper_search_failed",)
    assert not decision.candidates


def test_joint_search_excludes_unsafe_probe_pairs_fail_closed() -> None:
    parents = _parents()
    probes = tuple(
        replace(item, positive_safety_cost=0.2)
        if item.role is SoccerRole.PASSER and item.seed in {11, 17}
        else item
        for item in _probes(parents)
    )
    decision = learn_joint_policy_generation(
        parents=parents,
        spaces=default_three_role_search_spaces(),
        probes=probes,
    )

    assert not decision.passed
    passer = next(item for item in decision.updates if item.role is SoccerRole.PASSER)
    assert passer.rejected_probe_seeds == (11, 17)
    assert passer.reasons == ("insufficient_safe_mirrored_probes",)


def test_joint_search_requires_shared_world_seed_and_scenario_alignment() -> None:
    parents = _parents()
    probes = list(_probes(parents))
    keeper = next(
        index
        for index, item in enumerate(probes)
        if item.role is SoccerRole.GOALKEEPER and item.seed == 11
    )
    probes[keeper] = replace(probes[keeper], scenario_hash=_hash("wrong-scenario"))
    with pytest.raises(ValueError, match="share a scenario"):
        learn_joint_policy_generation(
            parents=parents,
            spaces=default_three_role_search_spaces(),
            probes=tuple(probes),
        )


def test_joint_search_rejects_wrong_parent_binding() -> None:
    parents = _parents()
    probes = list(_probes(parents))
    probes[0] = replace(probes[0], parent_artifact_hash=_hash("wrong-parent"))
    with pytest.raises(ValueError, match="not bound to its parent"):
        learn_joint_policy_generation(
            parents=parents,
            spaces=default_three_role_search_spaces(),
            probes=tuple(probes),
        )
