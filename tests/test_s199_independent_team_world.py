from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.growth.role_self_model import MatchRole
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.independent_team_world import (
    AgentWorldQuality,
    IndependentTeamWorldConfig,
    IndependentTeamWorldResult,
    IndependentTeamWorldScenario,
    simulate_independent_team_world,
)
from rosclaw_soccer.training.independent_team_growth import (
    build_independent_three_vs_three_fixture,
    default_independent_team_retention_cases,
)
from rosclaw_soccer.world.multi_player import (
    G1PitchPlayerSpec,
    build_g1_multi_player_stadium_model,
)


def _hash(label: str) -> str:
    return str(hash_json({"fixture": label}))


def _quality(agent_id: str, role: MatchRole) -> AgentWorldQuality:
    return AgentWorldQuality(
        agent_id=agent_id,
        role=role,
        active_fraction=0.75,
        displacement_m=0.8,
        distinct_intent_count=2,
        intent_switch_count=1,
        decision_count=10,
        pass_intent_count=1 if role is MatchRole.PLAYMAKER else 0,
        shot_intent_count=1 if role is MatchRole.FINISHER else 0,
        save_intent_count=1 if role is MatchRole.GOALKEEPER else 0,
        distribution_intent_count=0,
        minimum_pelvis_height_m=0.74,
        maximum_tilt_rad=0.18,
        joint_limit_violation=False,
        torque_limit_violation=False,
        required_minimum_pelvis_height_m=0.55,
        allowed_maximum_tilt_rad=0.80,
    )


def test_independent_world_result_requires_all_six_current_decisions() -> None:
    qualities = (
        _quality("red.goalkeeper", MatchRole.GOALKEEPER),
        _quality("red.playmaker", MatchRole.PLAYMAKER),
        _quality("red.finisher", MatchRole.FINISHER),
        _quality("blue.goalkeeper", MatchRole.GOALKEEPER),
        _quality("blue.playmaker", MatchRole.PLAYMAKER),
        _quality("blue.finisher", MatchRole.FINISHER),
    )
    result = IndependentTeamWorldResult(
        scenario_hash=_hash("scenario"),
        roster_hash=_hash("roster"),
        config_hash=_hash("config"),
        trajectory_hash=_hash("trajectory"),
        player_count=6,
        red_player_count=3,
        blue_player_count=3,
        decision_frame_count=10,
        coordination_frame_hashes=tuple(_hash(f"frame-{index}") for index in range(10)),
        qualities=qualities,
        pass_handshake_count=2,
        pass_intent_count=2,
        shot_intent_count=2,
        save_intent_count=2,
        distribution_intent_count=0,
        finite_state=True,
        robot_robot_contact_count=0,
        rolling_distance_m=1.2,
        peak_ball_speed_mps=2.1,
    )

    assert result.passed
    observed = replace(result, physics_evidence_hashes=(("red.finisher", _hash("observer")),))
    assert observed.passed and not observed.motor_policy_hashes
    assert "physics_evidence_hashes" not in result.to_dict()
    assert observed.result_hash != result.result_hash
    failed_evidence = replace(observed, physics_evidence_fault_agents=("red.finisher",))
    assert failed_evidence.safe and not failed_evidence.passed
    assert failed_evidence.to_dict()["physics_evidence_fault_agents"] == ["red.finisher"]
    for invalid in (None, [], (("red.finisher", 1),), (("unknown", _hash("observer")),)):
        with pytest.raises(ValueError):
            replace(result, physics_evidence_hashes=invalid)
    with pytest.raises(ValueError):
        replace(result, physics_evidence_fault_agents=("red.finisher",))
    with_motor = replace(result, motor_policy_hashes=(("red.finisher", _hash("motor")),))
    persistent = replace(with_motor, persistent_physics_observer_ids=("red.finisher",))
    assert "persistent_physics_observer_ids" not in with_motor.to_dict()
    assert persistent.to_dict()["persistent_physics_observer_ids"] == ["red.finisher"]
    assert persistent.result_hash != with_motor.result_hash
    for invalid in (None, [], (1,), ("unknown",), ("red.finisher", "red.finisher")):
        with pytest.raises(ValueError):
            replace(with_motor, persistent_physics_observer_ids=invalid)
    faulted_motor = replace(with_motor, motor_fault_agents=("red.finisher",))
    assert with_motor.passed and faulted_motor.safe and not faulted_motor.passed
    assert faulted_motor.to_dict()["motor_fault_agents"] == ["red.finisher"]
    assert faulted_motor.to_dict()["motor_policy_hashes"] == {"red.finisher": _hash("motor")}
    assert faulted_motor.result_hash != with_motor.result_hash
    with pytest.raises(ValueError):
        replace(result, motor_fault_agents=("red.finisher",))
    assert result.role_complete_both_teams
    stale = replace(qualities[0], decision_count=9)
    assert not replace(result, qualities=(stale, *qualities[1:])).passed
    low = replace(qualities[0], minimum_pelvis_height_m=0.54)
    assert not replace(result, qualities=(low, *qualities[1:])).safe
    duplicate_role = replace(qualities[1], role=MatchRole.FINISHER)
    assert not replace(result, qualities=(qualities[0], duplicate_role, *qualities[2:])).passed
    with pytest.raises(ValueError, match="result contract"):
        replace(result, pass_handshake_count=1)


def test_sim_only_config_and_scenario_fail_closed() -> None:
    with pytest.raises(ValueError, match="SIM-only"):
        IndependentTeamWorldConfig(hardware_authorized=True)
    with pytest.raises(ValueError, match="invalid"):
        IndependentTeamWorldScenario("other.stage", (1.0, 0.0, 0.115), (0.0, 0.0, 0.0), 1)
    with pytest.raises(ValueError, match="invalid"):
        IndependentTeamWorldScenario("s199.bad-vector", (), (0.0, 0.0, 0.0), 1)
    with pytest.raises(ValueError, match="prefix"):
        G1PitchPlayerSpec("red.player", "bad-prefix_", (0.0, 0.0, 0.0), 0.0)


def test_retention_suite_covers_four_distinct_role_situations() -> None:
    cases = default_independent_team_retention_cases()

    assert len(cases) == 4
    assert len({case.scenario_hash for case in cases}) == 4
    assert {case.scenario_id.rsplit(".", 1)[-1] for case in cases} == {
        "red-transition",
        "blue-attack-red-save",
        "red-finish",
        "blue-distribution",
    }


@pytest.mark.integration
def test_multi_player_builder_compiles_174_independent_actuators() -> None:
    value = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if value is None:
        pytest.skip("ROSCLAW_G1_ASSET_ROOT is not configured")
    root = Path(value)
    fixture = build_independent_three_vs_three_fixture(root)

    model = build_g1_multi_player_stadium_model(
        root,
        players=fixture.players,
        spec=fixture.goal,
    )

    assert model.nu == 6 * 29
    assert model.nq == 223


@pytest.mark.integration
def test_boundary_observation_is_independent_of_exit_termination() -> None:
    value = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if value is None:
        pytest.skip("ROSCLAW_G1_ASSET_ROOT is not configured")
    root = Path(value)
    traces = []
    for stop in (True, False):
        fixture = build_independent_three_vs_three_fixture(root)
        _, trace = simulate_independent_team_world(
            asset_root=root,
            roster=fixture.roster,
            cells=fixture.cells,
            players=fixture.players,
            scenario=IndependentTeamWorldScenario(
                "s199.boundary-observation",
                (fixture.goal.plane_x_m - 0.01, 0.0, fixture.goal.ball_radius_m),
                (1.0, 0.0, 0.0),
                2506,
            ),
            goal=fixture.goal,
            config=IndependentTeamWorldConfig(simulation_duration_sec=5.0, stop_on_ball_exit=stop),
        )
        expected = [
            -1.5,
            fixture.goal.plane_x_m,
            3.0,
            fixture.goal.ball_radius_m,
            fixture.goal.width_m,
            fixture.goal.height_m,
        ]
        np.testing.assert_array_equal(
            trace["pitch_boundary_geometry"], np.tile(expected, (len(trace["time"]), 1))
        )
        traces.append(trace)
    bounded, full = traces
    assert 0 < len(bounded["time"]) < len(full["time"]) == 250
    assert bounded.keys() == full.keys()
    for key in bounded:
        np.testing.assert_array_equal(bounded[key], full[key][: len(bounded[key])])


@pytest.mark.integration
def test_entry_support_capture_does_not_change_physical_trajectory() -> None:
    value = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if value is None:
        pytest.skip("ROSCLAW_G1_ASSET_ROOT is not configured")
    root = Path(value)
    traces = []
    for enabled in (False, True):
        fixture = build_independent_three_vs_three_fixture(root)
        _, trace = simulate_independent_team_world(
            asset_root=root,
            roster=fixture.roster,
            cells=fixture.cells,
            players=fixture.players,
            scenario=IndependentTeamWorldScenario(
                "s199.entry-support",
                (2.0, 0.0, fixture.goal.ball_radius_m),
                (0.2, 0.0, 0.0),
                920601,
            ),
            goal=fixture.goal,
            config=IndependentTeamWorldConfig(simulation_duration_sec=5.0),
            capture_initial_physics=True,
            capture_initial_support=enabled,
            physics_checkpoint_frame=30,
        )
        traces.append(trace)
    baseline, observed = traces
    for key in baseline:
        np.testing.assert_array_equal(baseline[key], observed[key])
    measurements = observed["initial_support_measurements"]
    assert measurements.shape == (6, 13)
    assert np.isfinite(measurements).all()
    assert set(measurements[:, 11]) <= {0, 1, 2, 3}
    assert len(set(observed["initial_support_agent_ids"])) == 6
    assert len(observed["initial_support_columns"]) == 13
