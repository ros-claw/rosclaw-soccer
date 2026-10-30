from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.team_native_state_capture import capture_team_native_states


def layout(count: int = 2):
    q = np.arange(7 + count * 36, dtype=np.float64)
    v = np.arange(6 + count * 35, dtype=np.float64)
    controllers = tuple(
        SimpleNamespace(
            qpos_base=7 + i * 36,
            qvel_base=6 + i * 35,
            joint_qpos=np.arange(14 + i * 36, 43 + i * 36, dtype=np.int64),
            joint_qvel=np.arange(12 + i * 35, 41 + i * 35, dtype=np.int64),
        )
        for i in range(count)
    )
    return q, v, controllers


def test_all_eight_players_share_same_ball_and_are_copies():
    q, v, controllers = layout(8)
    before_q, before_v = q.copy(), v.copy()
    a, b = capture_team_native_states(
        qpos=q, qvel=v, controllers=controllers, ball_qpos=0, ball_qvel=0
    )
    assert a.shape == (8, 43) and b.shape == (8, 41)
    for i in range(8):
        np.testing.assert_array_equal(a[i, :36], q[7 + i * 36 : 43 + i * 36])
        np.testing.assert_array_equal(a[i, 36:], q[:7])
        np.testing.assert_array_equal(b[i, :35], v[6 + i * 35 : 41 + i * 35])
        np.testing.assert_array_equal(b[i, 35:], v[:6])
    a[0, 0] = -1
    b[0, 0] = -1
    np.testing.assert_array_equal(q, before_q)
    np.testing.assert_array_equal(v, before_v)


def test_overlapping_or_invalid_layout_and_nonfinite_state_fail_closed():
    q, v, controllers = layout()
    with pytest.raises(ValueError, match="disjoint"):
        capture_team_native_states(
            qpos=q, qvel=v, controllers=(controllers[0], controllers[0]), ball_qpos=0, ball_qvel=0
        )
    with pytest.raises(ValueError):
        capture_team_native_states(
            qpos=q, qvel=v, controllers=controllers, ball_qpos=1, ball_qvel=0
        )
    q[0] = np.nan
    with pytest.raises(ValueError):
        capture_team_native_states(
            qpos=q, qvel=v, controllers=controllers, ball_qpos=0, ball_qvel=0
        )


def test_bad_capture_flag_rejected_before_loading_assets():
    from pathlib import Path

    from rosclaw_soccer.skills.team.independent_team_world import simulate_independent_team_world

    with pytest.raises(ValueError, match="team-state capture"):
        simulate_independent_team_world(
            asset_root=Path("not-loaded"),
            roster=None,
            cells=(),
            players=(),
            scenario=None,
            goal=None,
            capture_team_native_states=1,
        )  # type: ignore[arg-type]


@pytest.mark.integration
def test_four_vs_four_capture_preserves_physical_trajectory():
    import os
    from dataclasses import replace
    from pathlib import Path

    from rosclaw_soccer.skills.team.independent_team_world import (
        IndependentTeamWorldConfig,
        IndependentTeamWorldScenario,
        simulate_independent_team_world,
    )
    from rosclaw_soccer.training.four_vs_four_match import build_four_vs_four_fixture

    value = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if value is None:
        pytest.skip("ROSCLAW_G1_ASSET_ROOT is not configured")
    root = Path(value)
    traces = []
    for enabled in (False, True):
        fixture = build_four_vs_four_fixture(root, basic_ball_play=True)
        goal = replace(fixture.goal, ball_radius_m=0.11, ball_mass_kg=0.43)
        _, trace = simulate_independent_team_world(
            asset_root=root,
            roster=fixture.roster,
            cells=fixture.cells,
            players=fixture.players,
            scenario=IndependentTeamWorldScenario(
                "s199.team-state-capture",
                (2.8, -1.3, goal.ball_radius_m),
                (-0.1, 0.05, 0.0),
                1928,
            ),
            goal=goal,
            config=IndependentTeamWorldConfig(simulation_duration_sec=5.0),
            capture_team_native_states=enabled,
        )
        traces.append(trace)
    baseline, captured = traces
    for key in baseline:
        np.testing.assert_array_equal(baseline[key], captured[key])
    ids = captured["team_native_agent_ids"]
    q, v = captured["team_native_pre_control_qpos"], captured["team_native_pre_control_qvel"]
    assert list(ids) == sorted(str(x) for x in ids)
    assert q.shape == (len(captured["time"]), 8, 43)
    assert v.shape == (len(captured["time"]), 8, 41)
    assert np.isfinite(q).all() and np.isfinite(v).all()
    for j, agent in enumerate(ids):
        key = str(agent).replace(".", "_")
        np.testing.assert_array_equal(q[1:, j, :7], captured[key + "_pelvis_pose"][:-1])
        np.testing.assert_array_equal(q[1:, j, 7:36], captured[key + "_joint_position"][:-1])
        np.testing.assert_array_equal(q[1:, j, 36:], captured["ball_pose"][:-1])
        np.testing.assert_array_equal(v[1:, j, 6:35], captured[key + "_joint_velocity"][:-1])
        np.testing.assert_array_equal(v[1:, j, 35:], captured["ball_velocity"][:-1])
