import numpy as np
import pytest

from rosclaw_soccer.rsi.counterfactual_motor_replay import intervention_actions


def inputs():
    return np.zeros(12), np.zeros(12), np.zeros(12), np.tile([-1.0, 1.0], (12, 1))


def test_owned_axis_interventions():
    values = inputs()
    result = intervention_actions(*values)
    assert result.shape == (25, 12)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result[0], values[0])
    for joint in range(12):
        for side, sign in enumerate((-1, 1)):
            expected = np.zeros(12)
            expected[joint] = sign * 0.003
            np.testing.assert_array_equal(result[1 + 2 * joint + side], expected)
    result[:] = 99
    for value in values[:3]:
        assert not value.any()


def test_projection_preserves_original_action_and_all_three_bounds():
    actual, previous, nominal, limits = inputs()
    actual[0] = previous[0] = 0.16
    actual[1] = 0.012
    limits[2] = [-0.002, 0.002]
    result = intervention_actions(actual, previous, nominal, limits, radius=0.012)
    assert np.abs(result).max() <= 0.16
    assert np.abs(result - previous).max() <= 0.012 + 1e-15
    assert (result + nominal >= limits[:, 0]).all()
    assert (result + nominal <= limits[:, 1]).all()
    np.testing.assert_array_equal(result[2], result[0])
    assert len(np.unique(result, axis=0)) < 25


@pytest.mark.parametrize("radius", [True, 1, 0.0, -0.001, 0.013, float("nan"), float("inf")])
def test_invalid_radius(radius):
    with pytest.raises(ValueError):
        intervention_actions(*inputs(), radius=radius)


@pytest.mark.parametrize("index", range(4))
def test_nonfinite_input(index):
    values = list(inputs())
    values[index].flat[0] = np.nan
    with pytest.raises(ValueError):
        intervention_actions(*values)


def test_empty_box_and_outside_original_rejected():
    actual, previous, nominal, limits = inputs()
    previous[:] = 0.5
    with pytest.raises(ValueError):
        intervention_actions(actual, previous, nominal, limits)
    previous[:] = 0
    actual[0] = 0.02
    with pytest.raises(ValueError):
        intervention_actions(actual, previous, nominal, limits)
    actual[:] = 0
    limits[0] = [1, -1]
    with pytest.raises(ValueError):
        intervention_actions(actual, previous, nominal, limits)


def test_shape_and_boolean_arrays_rejected():
    values = list(inputs())
    values[0] = np.zeros(11)
    with pytest.raises(ValueError):
        intervention_actions(*values)
    values[0] = np.zeros(12, dtype=bool)
    with pytest.raises(ValueError):
        intervention_actions(*values)


def test_moving_nominal_joint_shield_can_override_final_delta_slew():
    actual, previous, nominal, limits = inputs()
    previous[0] = -0.13
    nominal[0] = -0.895
    actual[0] = limits[0, 0] - nominal[0]
    result = intervention_actions(actual, previous, nominal, limits)
    np.testing.assert_array_equal(result[:, 0], np.full(25, actual[0]))
    assert abs(result[0, 0] - previous[0]) > 0.012


def test_existing_foundation_limit_violation_not_forced_in_range():
    actual, previous, nominal, limits = inputs()
    nominal[0] = 1.2
    result = intervention_actions(actual, previous, nominal, limits)
    assert result[:, 0].max() == 0
    assert result[:, 0].min() == -0.003
