import numpy as np
import pytest

from rosclaw_soccer.rsi.transition_feasibility import transition_feasibility


def test_box_intersection_lower_bound_and_exact_boundary():
    previous = np.array([0.0, 0.5, -0.8])
    reference = np.zeros(3)
    result = transition_feasibility(previous, reference, reference_radius=0.2, max_step=0.6)
    np.testing.assert_allclose(result["minimum_jump_rad"], [0.0, 0.3, 0.6])
    assert result["limiting_joint_index"] == 2
    # Binary floating arithmetic may make 0.8 - 0.2 slightly above 0.6;
    # callers must declare any tolerance explicitly, never assume one.
    assert result["feasible"] is False
    assert transition_feasibility(previous, reference, reference_radius=0.2, max_step=0.601)[
        "feasible"
    ]


def test_projection_attains_lower_bound_and_infeasibility_is_per_joint():
    previous = np.array([1.2, -0.4, 0.1])
    reference = np.array([0.1, 0.0, 0.0])
    result = transition_feasibility(previous, reference, reference_radius=0.2, max_step=0.25)
    projection = np.clip(previous, reference - 0.2, reference + 0.2)
    np.testing.assert_allclose(result["minimum_jump_rad"], np.abs(projection - previous))
    np.testing.assert_array_equal(result["violating_joint_indices"], [0])
    assert result["feasible"] is False


@pytest.mark.parametrize("radius,step", [(float("nan"), 0.1), (0.2, -1), (True, 0.1)])
def test_reject_invalid_bounds(radius, step):
    with pytest.raises(ValueError):
        transition_feasibility(np.zeros(2), np.zeros(2), reference_radius=radius, max_step=step)


def test_reject_nonfinite_or_mismatched_targets():
    for previous, reference in ((np.array([np.nan]), np.zeros(1)), (np.zeros(2), np.zeros(1))):
        with pytest.raises(ValueError):
            transition_feasibility(previous, reference, reference_radius=0.2, max_step=0.1)
