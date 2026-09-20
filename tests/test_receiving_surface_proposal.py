import numpy as np
import pytest

from rosclaw_soccer.training.receiving_surface_proposal import bounded_surface_correction


def solve(matrix, target, lower=None, upper=None):
    matrix = np.asarray(matrix)
    n = matrix.shape[-1]
    return bounded_surface_correction(
        matrix,
        np.asarray(target),
        np.full(n, -0.1) if lower is None else np.asarray(lower),
        np.full(n, 0.1) if upper is None else np.asarray(upper),
    )


def test_coupled_hip_ankle_can_separate_shin_without_retracting_foot_locally():
    matrix = np.array([[0.41, 0.127], [0.54, 0.0]])
    before = matrix.copy()
    delta = solve(matrix, [0.0, 0.01])
    assert delta[0] > 0 and delta[1] < 0
    np.testing.assert_allclose(matrix @ delta, [0.0, 0.01], atol=0.0001)
    np.testing.assert_array_equal(matrix, before)
    np.testing.assert_array_equal(delta, solve(matrix, [0.0, 0.01]))


def test_box_saturation_is_not_reported_as_feasibility():
    delta = solve([[0.1]], [0.05], [-0.01], [0.02])
    assert delta[0] == 0.02
    assert abs(0.1 * delta[0] - 0.05) > 0.04


def test_zero_and_conflicting_jacobians_remain_finite():
    np.testing.assert_array_equal(solve([[0.0, 0.0]], [0.01]), [0.0, 0.0])
    np.testing.assert_allclose(solve([[1.0], [1.0]], [0.01, -0.01]), [0.0], atol=1e-12)


@pytest.mark.parametrize(
    "matrix,target,lower,upper",
    [
        ([[float("nan")]], [0.0], [-0.1], [0.1]),
        ([[1.0]], [float("inf")], [-0.1], [0.1]),
        ([[True]], [0.0], [-0.1], [0.1]),
        ([[3.0]], [0.0], [-0.1], [0.1]),
        ([[1.0]], [0.051], [-0.1], [0.1]),
        ([[1.0]], [0.0], [0.01], [0.1]),
        ([[1.0]], [0.0], [-0.1], [-0.01]),
        ([[1.0]], [0.0], [-0.201], [0.1]),
        ([[1.0]], [0.0], [-0.1], [0.201]),
        ([[1.0]], [0.0, 0.0], [-0.1], [0.1]),
        ([[1.0] * 30], [0.0], [-0.1] * 30, [0.1] * 30),
    ],
)
def test_rejects_unbounded_or_malformed_inputs(matrix, target, lower, upper):
    with pytest.raises(ValueError):
        solve(matrix, target, lower, upper)
