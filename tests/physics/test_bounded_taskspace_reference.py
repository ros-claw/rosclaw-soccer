import numpy as np
import pytest

from rosclaw_soccer.physics.bounded_taskspace_reference import solve_bounded_taskspace_reference


def problem():
    u = np.linspace(0, 1, 49)
    nominal = np.c_[0.4 + 0.2 * np.sin(4 * np.pi * u), np.zeros(49), np.full(49, 0.1)]
    anchor = np.tile([0.4, 0, 0.9], (49, 1))
    waypoint = np.array([0.0, 0.05, 0.15])
    tau = np.where(u <= 0.5, u * 2, u * 2 - 1)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    w = np.where(u <= 0.5, s, 1 - s)
    preferred = w[:, None] * (waypoint - nominal)
    return nominal, anchor, waypoint, preferred


def solve(nominal, anchor, waypoint, preferred):
    return solve_bounded_taskspace_reference(
        nominal,
        anchor,
        waypoint,
        preferred,
        peak=24,
        dt=0.02,
        speed_limit=3.0,
        workspace_radius=1.25,
        floor_z=-0.05,
    )


def test_moving_reference_blend_speed_breach_replaced_by_bounded_c2_vector_curve():
    from scipy.interpolate import BSpline

    nominal, anchor, waypoint, preferred = problem()
    assert np.linalg.norm(np.diff(preferred, axis=0) / 0.02, axis=1).max() > 3
    saved = nominal.copy()
    position, certificate = solve(nominal, anchor, waypoint, preferred)
    np.testing.assert_array_equal(nominal, saved)
    np.testing.assert_array_equal(position[[0, 48]], nominal[[0, 48]])
    np.testing.assert_allclose(position[24], waypoint, atol=1e-8)
    offset = BSpline(certificate["knots_seconds"], certificate["offset_coefficients"], 5)
    speed = np.linalg.norm(offset.derivative()(np.linspace(0, 0.96, 10001)), axis=1)
    assert speed.max() <= 3 + 1e-8
    assert certificate["maximum_derivative_control_vector_speed_m_s"] <= 3 + 1e-8
    for order in (0, 1, 2):
        np.testing.assert_allclose(offset.derivative(order)([0.0, 0.96]), 0.0, atol=1e-7)
    np.testing.assert_allclose(offset.derivative()(0.48), 0.0, atol=1e-7)
    np.testing.assert_allclose(
        offset.derivative(2)(0.48 - 1e-10), offset.derivative(2)(0.48 + 1e-10), atol=1e-5
    )
    assert np.linalg.norm(position - anchor, axis=1).max() <= 1.25 + 1e-9
    assert position[:, 2].min() >= -0.05 - 1e-9
    assert not certificate["physical_feasibility_proven"]


def test_impossible_waypoint_abstains_without_modifying_inputs():
    nominal, anchor, waypoint, preferred = problem()
    waypoint[0] = 10
    saved = nominal.copy()
    with pytest.raises(ValueError, match="workspace"):
        solve(nominal, anchor, waypoint, preferred)
    np.testing.assert_array_equal(nominal, saved)


def test_impossible_speed_abstains_not_clips_endpoint():
    nominal, anchor, waypoint, preferred = problem()
    with pytest.raises(ValueError, match="abstained"):
        solve_bounded_taskspace_reference(
            nominal,
            anchor,
            waypoint,
            preferred,
            peak=24,
            dt=0.02,
            speed_limit=0.01,
            workspace_radius=1.25,
            floor_z=-0.05,
        )


@pytest.mark.parametrize("bad", ["nan", "shape", "dtype", "clock"])
def test_bad_problem_rejected_before_optimizer(bad):
    nominal, anchor, waypoint, preferred = problem()
    dt = 0.02
    if bad == "nan":
        nominal[0, 0] = np.nan
    elif bad == "shape":
        waypoint = waypoint[:2]
    elif bad == "dtype":
        nominal = nominal.astype(np.float32)
    else:
        dt = 0.0
    with pytest.raises(ValueError, match="typed"):
        solve_bounded_taskspace_reference(
            nominal,
            anchor,
            waypoint,
            preferred,
            peak=24,
            dt=dt,
            speed_limit=3.0,
            workspace_radius=1.25,
            floor_z=-0.05,
        )
