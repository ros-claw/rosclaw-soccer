"""Fail-closed contract for passive simulation-only pass telemetry."""

import os
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    IndependentTeamWorldScenario,
    simulate_independent_team_world,
)
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture


@pytest.mark.parametrize("value", [True, 1, "yes", None])
def test_entry_capture_requires_explicit_bool_and_strict_handoff(value):
    with pytest.raises(ValueError, match="pass control-entry capture requires strict handoff"):
        simulate_independent_team_world(
            asset_root=Path("/nonexistent-sim-assets"),
            roster=None,
            cells=(),
            players=(),
            scenario=None,
            goal=None,
            capture_pass_control_entry=value,
        )


@pytest.mark.integration
def test_pass_entry_capture_preserves_physics_and_records_frame_boundary():
    value = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if value is None:
        pytest.skip("ROSCLAW_G1_ASSET_ROOT is not configured")
    root = Path(value)
    traces = []
    for capture in (False, True):
        fixture = build_independent_three_vs_three_fixture(root)
        _, trace = simulate_independent_team_world(
            asset_root=root,
            roster=fixture.roster,
            cells=fixture.cells,
            players=fixture.players,
            scenario=IndependentTeamWorldScenario(
                "s199.pass-entry-passive-test",
                (2.0, 0.0, fixture.goal.ball_radius_m),
                (0.2, 0.0, 0.0),
                920604,
            ),
            goal=fixture.goal,
            config=IndependentTeamWorldConfig(
                simulation_duration_sec=5.0,
                strict_receive_handoff=True,
                directed_pass_launch=True,
            ),
            capture_pass_control_entry=capture,
        )
        traces.append(trace)
    baseline, captured = traces
    for key in baseline:
        np.testing.assert_array_equal(baseline[key], captured[key])
    count = len(captured["time"])
    assert captured["pass_entry_target_xy_m"].shape == (count, 2)
    assert captured["pass_entry_target_valid"].dtype == np.bool_
    np.testing.assert_allclose(captured["pass_entry_time_sec"] + 0.02, captured["time"], atol=1e-8)
    np.testing.assert_array_equal(
        captured["pass_entry_ball_velocity_xy_mps"], captured["ball_control_entry_velocity"][:, :2]
    )
    valid = captured["pass_entry_target_valid"]
    assert np.all(np.isfinite(captured["pass_entry_target_xy_m"]))
    assert np.all(
        captured["pass_entry_created_sec"][valid] <= captured["pass_entry_time_sec"][valid]
    )
