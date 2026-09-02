from __future__ import annotations

import pytest

from rosclaw_soccer.growth.tactical_2v1 import TacticalAction
from rosclaw_soccer.skills.team.shared_world import (
    G1MovementWaypoint,
    G1TacticalMovementConfig,
)
from rosclaw_soccer.training.active_off_ball_growth import (
    ActiveRouteCandidate,
    build_action_conditioned_movement_plan,
    default_active_acquisition_scenarios,
    default_active_retention_manifest,
    default_active_route_candidates,
)
from rosclaw_soccer.training.full_body_tactical_2v1 import FullBodyRoleMovementPlan


def test_tactical_movement_contract_is_bounded_and_sim_only() -> None:
    waypoints = (
        G1MovementWaypoint(0.0, (4.0, -0.4, 0.0)),
        G1MovementWaypoint(2.0, (4.5, -0.2, 0.0)),
    )
    config = G1TacticalMovementConfig(waypoints=waypoints)
    assert config.execution_mode == "SIM_ONLY"
    assert config.hardware_authorized is False
    assert config.config_hash.startswith("sha256:")
    with pytest.raises(ValueError, match="strictly increasing"):
        G1TacticalMovementConfig(waypoints=(waypoints[1], waypoints[0]))
    with pytest.raises(ValueError, match="permanently SIM_ONLY"):
        G1TacticalMovementConfig(waypoints=waypoints, hardware_authorized=True)
    with pytest.raises(ValueError, match="ground plane"):
        G1MovementWaypoint(1.0, (4.0, 0.0, 0.1))


def test_action_conditioned_plan_activates_teammate_and_defender() -> None:
    scenario = default_active_retention_manifest().scenarios[0]
    candidate = default_active_route_candidates()[1]
    plan = build_action_conditioned_movement_plan(scenario, TacticalAction.PASS, candidate)
    assert isinstance(plan, FullBodyRoleMovementPlan)
    assert plan.activation_ceiling == "SIM_ONLY"
    assert plan.hardware_authorized is False
    assert plan.teammate_origin_m != scenario.teammate_origin_m
    assert plan.teammate_movement.waypoints[-1].time_sec == pytest.approx(7.0)
    assert plan.defender_movement.waypoints[-1].time_sec == pytest.approx(7.0)
    assert plan.plan_hash.startswith("sha256:")
    shoot = build_action_conditioned_movement_plan(
        default_active_retention_manifest().scenarios[-1],
        TacticalAction.SHOOT,
        candidate,
    )
    assert shoot.plan_hash != plan.plan_hash
    assert len(shoot.teammate_movement.waypoints) == 5


def test_route_portfolio_and_sealed_retention_are_disjoint() -> None:
    candidates = default_active_route_candidates()
    assert len(candidates) == 3
    assert len({candidate.candidate_hash for candidate in candidates}) == 3
    acquisition = default_active_acquisition_scenarios()
    retention = default_active_retention_manifest()
    assert len(acquisition) == len(retention.scenarios) == 8
    assert {scenario.scenario_hash for scenario in acquisition}.isdisjoint(
        scenario.scenario_hash for scenario in retention.scenarios
    )
    assert retention.training_access_allowed is False
    with pytest.raises(ValueError, match="qualified envelope"):
        ActiveRouteCandidate("s121.bad", 2.0, 0.4, 0.7, 0.9)
