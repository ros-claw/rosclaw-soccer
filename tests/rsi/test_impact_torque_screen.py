"""Synthetic action/proxy tests, not an actual G1 teacher or M0 improvement."""

import ast
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.physics.g1_joint_task_kinematics import G1JointTaskKinematics
from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecast
from rosclaw_soccer.rsi import impact_torque_screen as screen


def sample():
    values = {f.name: np.zeros(1) for f in fields(G1JointTaskKinematics)}
    rng = np.random.default_rng(20261019)
    values["foot_material_point_jacobian_m_per_rad"] = rng.normal(size=(2, 3, 29))
    values["foot_rotation_jacobian"] = rng.normal(size=(2, 3, 29))
    return G1JointTaskKinematics(**values)


def forecast():
    values = {f.name: np.zeros(1) for f in fields(G1PDTorqueForecast)}
    values.update(
        donor_integration_state_hash="synthetic-not-physics",
        nonfoot_robot_contact_over_1n=False,
        canonical_qvel_after_substep=np.zeros((10, 41)),
        pelvis_height_solver_epoch_m=np.full(10, 0.7),
        robot_force_on_ball_world_n=np.tile([2.0, 0.0, 0.0], (10, 1)),
    )
    values["canonical_qvel_after_substep"][:, 35] = 2.0
    return G1PDTorqueForecast(**values)


@pytest.mark.parametrize("side", [0, 1])
def test_candidates_are_bounded_independent_selected_leg_only(side):
    s = sample()
    original = s.foot_material_point_jacobian_m_per_rad.copy()
    rows = screen.impact_torque_candidates(s, side)
    assert rows[0][0] == "baseline" and np.all(rows[0][1] == 0)
    assert 1 < len(rows) <= screen.MAX_CANDIDATES == 13
    assert len({a.tobytes() for _, a in rows}) == len(rows)
    for _, a in rows:
        assert a.shape == (29,) and a.dtype == np.float64
        assert np.all(abs(a) <= screen.CAPS)
        allowed = np.zeros(29, dtype=bool)
        allowed[6 * side : 6 * side + 6] = True
        assert np.all(a[~allowed] == 0)
        assert not np.signbit(a[a == 0]).any()
    rows[1][1][:] = 0
    np.testing.assert_array_equal(s.foot_material_point_jacobian_m_per_rad, original)


def test_degenerate_action_bases_deduplicate_without_inflating_candidate_count():
    s = sample()
    s.foot_material_point_jacobian_m_per_rad[:] = 0
    s.foot_rotation_jacobian[:] = 0
    rows = screen.impact_torque_candidates(s, 0)
    assert len(rows) == 1 and rows[0][0] == "baseline"


@pytest.mark.parametrize("side", [-1, 2, True, 0.0])
def test_invalid_leg_rejected(side):
    with pytest.raises(ValueError):
        screen.impact_torque_candidates(sample(), side)


def test_proxy_prefers_aligned_forward_impact_but_never_claims_post60_success():
    f = forecast()
    zero = np.zeros(29)
    a = screen.impact_prediction_score(f, zero)
    assert a["feasible_predicted_impact"] and a["proxy_cost"] == 0
    f.canonical_qvel_after_substep[:, 36] = 0.8
    b = screen.impact_prediction_score(f, zero)
    assert b["proxy_cost"] > a["proxy_cost"]
    for k in (
        "final_m0_success_proven",
        "full_body_stability_after_window_proven",
        "trained_policy",
        "runtime_execution_authorized",
        "teacher_qualified",
        "hardware_authorized",
        "promotion_authorized",
    ):
        assert b[k] is False
    assert b["outcome_horizon_sec"] == 0.02


@pytest.mark.parametrize("veto", ["nonfoot", "height", "no_contact"])
def test_infeasible_prediction_cannot_win_via_zero_or_nonfinite_cost(veto):
    f = forecast()
    if veto == "nonfoot":
        object.__setattr__(f, "nonfoot_robot_contact_over_1n", True)
    elif veto == "height":
        f.pelvis_height_solver_epoch_m[4] = 0.649
    else:
        f.robot_force_on_ball_world_n[:] = 0
    r = screen.impact_prediction_score(f, np.zeros(29))
    assert not r["feasible_predicted_impact"] and r["proxy_cost"] is None
    assert r["rejection_reason"] is not None


@pytest.mark.parametrize("bad", ["offset", "nan", "dtype", "short", "huge"])
def test_whole_finite_prediction_and_offset_required(bad):
    f = forecast()
    o = np.zeros(29)
    if bad == "offset":
        o[0] = screen.CAPS[0] + 1e-6
    elif bad == "nan":
        f.canonical_qvel_after_substep[0, 35] = np.nan
    elif bad == "dtype":
        object.__setattr__(f, "pelvis_height_solver_epoch_m", np.ones(10, dtype=np.float32))
    elif bad == "short":
        object.__setattr__(f, "canonical_qvel_after_substep", np.zeros((9, 41)))
    else:
        f.canonical_qvel_after_substep[0, 35] = 1e300
    with pytest.raises(ValueError):
        screen.impact_prediction_score(f, o)


def test_numeric_proposal_and_proxy_do_not_run_physics_or_modify_any_body():
    tree = ast.parse(Path(screen.__file__).read_text())
    calls = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert not calls.intersection(
        {"mj_step", "mj_forward", "forecast", "MjData", "MjModel", "request_action"}
    )
