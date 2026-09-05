from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw_soccer.growth.physical_option_router import PhysicalSoccerOption
from rosclaw_soccer.sim.contracts import ShotParameters
from rosclaw_soccer.skills.team.independent_option_world import (
    IndependentOptionScenario,
    IndependentOptionWorldConfig,
    physical_option_policy_hash,
)
from rosclaw_soccer.training.independent_option_growth import (
    build_physical_option_fixture,
    default_independent_option_scenarios,
)
from rosclaw_soccer.training.independent_team_growth import (
    build_independent_three_vs_three_fixture,
)


def test_s200_scenarios_cover_independent_pass_and_shoot() -> None:
    scenarios = default_independent_option_scenarios()

    assert len(scenarios) == 2
    assert {value.expected_option for value in scenarios} == {
        PhysicalSoccerOption.PASS,
        PhysicalSoccerOption.SHOOT,
    }
    assert len({value.scenario_hash for value in scenarios}) == 2
    assert all(value.parameters.kick_foot == "right" for value in scenarios)


def test_s200_contracts_fail_closed_outside_sim_envelope() -> None:
    with pytest.raises(ValueError, match="SIM-only envelope"):
        IndependentOptionWorldConfig(hardware_authorized=True)
    with pytest.raises(ValueError, match="SIM-only envelope"):
        IndependentOptionWorldConfig(
            simulation_duration_sec=9.0,
            locomotion=replace(
                IndependentOptionWorldConfig().locomotion,
                simulation_duration_sec=10.0,
            ),
        )
    with pytest.raises(ValueError, match="scenario is invalid"):
        IndependentOptionScenario(
            scenario_id="other.physical",
            option_agent_id="red.finisher",
            expected_option=PhysicalSoccerOption.SHOOT,
            ball_initial_position_m=(1.25, 0.0, 0.115),
            ball_initial_velocity_mps=(0.0, 0.0, 0.0),
            preferred_target_m=(7.5, 0.8, 1.5),
            parameters=ShotParameters(),
            seed=1,
        )


@pytest.mark.integration
def test_physical_option_fixture_assigns_the_source_body_to_option_owner() -> None:
    value = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if value is None:
        pytest.skip("ROSCLAW_G1_ASSET_ROOT is not configured")
    root = Path(value)
    base = build_independent_three_vs_three_fixture(root)

    for scenario in default_independent_option_scenarios():
        fixture = build_physical_option_fixture(base, scenario)
        source = next(player for player in fixture.players if not player.body_prefix)
        assert source.agent_id == scenario.option_agent_id
        assert source.origin_m == (0.0, 0.0, 0.0)
        assert len({player.body_prefix for player in fixture.players}) == 6
        assert physical_option_policy_hash(root).startswith("sha256:")
