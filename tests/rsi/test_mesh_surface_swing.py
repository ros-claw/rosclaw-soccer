import numpy as np
import pytest

from rosclaw_soccer.rsi.mesh_surface_swing import surface_strike_joint_delta


def inputs():
    return dict(
        surface_point_world=np.zeros(3),
        ball_center_world=np.array([0.2, 0.03, 0.02]),
        surface_point_jacobian=np.concatenate((np.eye(3), np.zeros((3, 3))), axis=1),
        baseline_target=np.zeros(6),
        joint_limits=np.tile([-1.0, 1.0], (6, 1)),
        strike_direction_world=np.array([1.0, 0.0, 0.0]),
        ball_radius_m=0.11,
        forward_cap_m=0.08,
        lateral_cap_m=0.15,
        vertical_cap_m=0.04,
    )


def test_surface_target_uses_ball_radius_not_body_origin_standoff():
    values = inputs()
    result = surface_strike_joint_delta(**values)
    np.testing.assert_allclose(result[:3], np.array([0.08, 0.03, 0.02]) / 1.0025)
    assert np.all(result[3:] == 0)
    assert np.linalg.norm(values["ball_center_world"] - [0.11, 0, 0] - result[:3]) < 0.011


def test_zero_delta_at_actual_target():
    values = inputs()
    values["surface_point_world"] = values["ball_center_world"] - [0.11, 0, 0]
    np.testing.assert_array_equal(surface_strike_joint_delta(**values), np.zeros(6))


def test_vertical_cap_is_a_cap_not_unconditional_lift():
    values = inputs()
    values["ball_center_world"][2] = -0.02
    assert surface_strike_joint_delta(**values)[2] < 0


def test_guard_does_not_move_outward_from_joint_boundary():
    values = inputs()
    values["baseline_target"][0] = 0.999
    assert surface_strike_joint_delta(**values)[0] == 0
    values["baseline_target"][0] = 1.1
    assert surface_strike_joint_delta(**values)[0] == 0
    values["surface_point_world"][0] = 0.3
    assert surface_strike_joint_delta(**values)[0] < 0


@pytest.mark.parametrize(
    "key", ["surface_point_world", "surface_point_jacobian", "baseline_target"]
)
def test_nonfinite_input_is_rejected(key):
    values = inputs()
    values[key].flat[0] = np.nan
    with pytest.raises(ValueError):
        surface_strike_joint_delta(**values)


def test_nonunit_direction_rejected():
    values = inputs()
    values["strike_direction_world"][0] = 2
    with pytest.raises(ValueError):
        surface_strike_joint_delta(**values)


def test_unphysical_finite_jacobian_rejected_before_linear_solve():
    values = inputs()
    values["surface_point_jacobian"][:] = 1e100
    with pytest.raises(ValueError):
        surface_strike_joint_delta(**values)


def test_joint_delta_cap_and_input_ownership():
    values = inputs()
    values["surface_point_jacobian"] *= 0.01
    snapshots = {k: v.copy() for k, v in values.items() if isinstance(v, np.ndarray)}
    assert np.max(np.abs(surface_strike_joint_delta(**values))) <= 0.35
    for key, original in snapshots.items():
        np.testing.assert_array_equal(values[key], original)
