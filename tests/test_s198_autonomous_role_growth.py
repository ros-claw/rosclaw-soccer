from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw_soccer.growth.reactive_route_actor import (
    ReactiveRouteSample,
    fit_reactive_route_actor,
    save_reactive_route_actor,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.shared_world import G1RoleAutonomyConfig
from rosclaw_soccer.training.autonomous_role_growth import (
    build_autonomous_role_plan,
    default_autonomous_development_cases,
    default_autonomous_retention_cases,
    default_autonomous_role_candidates,
)
from rosclaw_soccer.training.full_body_tactical_2v1 import (
    FullBodyAutonomousRoleMovementPlan,
)


def _actor(tmp_path: Path):  # type: ignore[no-untyped-def]
    samples = tuple(
        ReactiveRouteSample(
            episode_id=f"episode-{index % 8}",
            features=tuple(
                [0.1 * ((index + feature) % 7) for feature in range(10)]
                + [float(index % 2), float((index + 1) % 2)] * 2
            ),
            teacher_world_command_xy_mps=(0.20, 0.04),
        )
        for index in range(1_024)
    )
    actor = fit_reactive_route_actor(
        samples,
        source_stage_hash=hash_json({"stage": "s197-test"}),
    )
    path = tmp_path / "route-actor.json"
    save_reactive_route_actor(actor, path)
    return actor, path


def test_role_autonomy_is_bounded_and_permanently_sim_only() -> None:
    config = G1RoleAutonomyConfig(role="teammate", team_id="red")
    assert config.config_hash.startswith("sha256:")
    assert config.execution_mode == "SIM_ONLY"
    assert config.hardware_authorized is False
    with pytest.raises(ValueError, match="SIM-only contract"):
        G1RoleAutonomyConfig(role="teammate", team_id="red", hardware_authorized=True)
    with pytest.raises(ValueError, match="SIM-only contract"):
        G1RoleAutonomyConfig(role="goalkeeper", team_id="blue")
    with pytest.raises(ValueError, match="SIM-only contract"):
        G1RoleAutonomyConfig(
            role="defender",
            team_id="blue",
            maximum_target_shift_m=0.50,
        )


def test_autonomous_plan_binds_independent_red_blue_roles(tmp_path: Path) -> None:
    actor, path = _actor(tmp_path)
    candidate = default_autonomous_role_candidates()[1]
    case = default_autonomous_development_cases()[0]
    plan = build_autonomous_role_plan(
        case=case,
        candidate=candidate,
        actor_path=path,
        actor=actor,
    )
    assert isinstance(plan, FullBodyAutonomousRoleMovementPlan)
    assert plan.teammate_autonomy.team_id == "red"
    assert plan.defender_autonomy.team_id == "blue"
    assert plan.teammate_autonomy.role == "teammate"
    assert plan.defender_autonomy.role == "defender"
    assert plan.teammate_movement.actor_hash == actor.actor_hash
    assert plan.defender_movement.actor_hash == actor.actor_hash
    assert plan.activation_ceiling == "SIM_ONLY"
    assert plan.hardware_authorized is False
    assert plan.plan_hash.startswith("sha256:")


def test_selection_and_retention_curricula_are_balanced_and_disjoint() -> None:
    development = default_autonomous_development_cases()
    retention = default_autonomous_retention_cases()
    candidates = default_autonomous_role_candidates()
    assert len(development) == len(retention) == 4
    assert [case.action.value for case in development].count("pass") == 2
    assert [case.action.value for case in development].count("shoot") == 2
    assert [case.action.value for case in retention].count("pass") == 2
    assert [case.action.value for case in retention].count("shoot") == 2
    assert {case.case_hash for case in development}.isdisjoint(case.case_hash for case in retention)
    assert [candidate.maximum_target_shift_m for candidate in candidates] == [
        0.08,
        0.12,
        0.22,
    ]
