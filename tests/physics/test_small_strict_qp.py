import numpy as np
import pytest

from rosclaw_soccer.physics.small_strict_qp import solve_small_l1_qp, solve_small_strict_qp


def test_unconstrained_optimum_inside_declared_box():
    h = np.eye(2)
    g = np.array([-1.0, 2.0])
    a = np.r_[np.eye(2), -np.eye(2)]
    b = np.full(4, 3.0)
    x, c = solve_small_strict_qp(h, g, a, b)
    np.testing.assert_array_equal(x, [1.0, -2.0])
    assert c["maximum_stationarity_residual"] < 1e-9 and c["physical_feasibility_not_implied"]


def test_active_box_and_oblique_linear_constraints_have_kkt_certificate():
    h = np.eye(2)
    g = np.array([-2.0, -2.0])
    a = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
    b = np.array([1.0, 1.0, 0.0, 0.0, 0.5])
    x, c = solve_small_strict_qp(h, g, a, b)
    np.testing.assert_allclose(x, [0.25, 0.25], atol=1e-12)
    assert c["maximum_primal_violation"] < 1e-9 and c["minimum_active_multiplier"] >= 0


def test_four_dimensional_box_enumeration_is_bounded():
    h = np.eye(4)
    g = np.array([-2.0, 2.0, -0.5, 0.5])
    a = np.r_[np.eye(4), -np.eye(4)]
    b = np.ones(8)
    x, c = solve_small_strict_qp(h, g, a, b)
    np.testing.assert_allclose(x, [1.0, -1.0, 0.5, -0.5])
    assert c["enumerated_active_sets"] == 163


@pytest.mark.parametrize("defect", ["nonconvex", "nan", "asymmetric", "infeasible", "wrongshape"])
def test_invalid_or_uncertified_problem_rejects_without_relaxation(defect):
    h = np.eye(2)
    g = np.zeros(2)
    a = np.array([[1.0, 0.0], [-1.0, 0.0]])
    b = np.ones(2)
    if defect == "nonconvex":
        h[0, 0] = -1
    elif defect == "nan":
        g[0] = np.nan
    elif defect == "asymmetric":
        h[0, 1] = 1
    elif defect == "infeasible":
        b[:] = -1
    elif defect == "wrongshape":
        b = np.ones(3)
    with pytest.raises(ValueError):
        solve_small_strict_qp(h, g, a, b)


def test_l1_matches_analytic_soft_threshold_and_preserves_physical_caveat():
    h = np.eye(2)
    g = np.array([-0.1, -2.0])
    a = np.r_[np.eye(2), -np.eye(2)]
    b = np.full(4, 3.0)
    x, c = solve_small_l1_qp(h, g, a, b, penalty=0.25)
    np.testing.assert_allclose(x, [0.0, 1.75], atol=1e-12)
    assert c["sign_regions_considered"] == 4 and c["physical_feasibility_not_implied"]


def test_l1_still_obeys_oblique_constraints_not_posthoc_clipping():
    h = np.eye(2)
    g = np.array([-2.0, -2.0])
    a = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
    b = np.array([1.0, 1.0, 0.0, 0.0, 0.5])
    x, _ = solve_small_l1_qp(h, g, a, b, penalty=0.25)
    np.testing.assert_allclose(x, [0.25, 0.25], atol=1e-12)
