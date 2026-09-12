import numpy as np
import pytest

from rosclaw_soccer.training.directed_plane_crossing import first_directed_plane_crossing


def evaluate(positions, *, origin=None, forward=None):
    return first_directed_plane_crossing(
        np.asarray(positions, dtype=np.float64),
        origin=np.zeros(2) if origin is None else origin,
        forward=np.array([1.0, 0.0]) if forward is None else forward,
    )


def test_signed_interpolation_and_explicit_miss():
    result = evaluate([[[-1, -2], [-2, 3]], [[1, 0], [-1, 3]]])
    np.testing.assert_array_equal(result.crossed, [True, False])
    np.testing.assert_array_equal(result.segment_index, [0, -1])
    assert result.fraction[0] == 0.5
    assert result.signed_lateral_error[0] == -1.0
    assert np.isnan(result.fraction[1]) and np.isnan(result.signed_lateral_error[1])
    for value in vars(result).values():
        assert not value.flags.writeable


def test_first_forward_crossing_not_backward_or_initial_plane():
    result = evaluate([[[0, 5]], [[1, 5]], [[-1, 1]], [[1, 3]], [[-1, 9]], [[1, 9]]])
    assert result.segment_index[0] == 2
    assert result.signed_lateral_error[0] == 2


def test_rotation_translation_and_axis_scale_preserve_error():
    points = np.array([[[-1.0, -2]], [[3, 2]]])
    expected = evaluate(points)
    angle = 1.234
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    origin = np.array([12.0, -19.0])
    result = evaluate(points @ rotation.T + origin, origin=origin, forward=7 * rotation[:, 0])
    np.testing.assert_allclose(result.signed_lateral_error, expected.signed_lateral_error)
    np.testing.assert_allclose(result.fraction, expected.fraction)


def test_backward_attack_uses_its_own_left():
    result = evaluate([[[1, 2]], [[-1, 2]]], forward=np.array([-1.0, 0.0]))
    assert result.signed_lateral_error[0] == -2


def test_exact_endpoint_and_input_unchanged():
    points = np.array([[[-1.0, 1]], [[0, 2]], [[1, 3]]], dtype=np.float32)
    old = points.copy()
    result = first_directed_plane_crossing(points, origin=np.zeros(2), forward=np.array([1.0, 0.0]))
    assert result.fraction[0] == 1 and result.segment_index[0] == 0
    assert result.signed_lateral_error[0] == 2
    np.testing.assert_array_equal(points, old)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e7])
@pytest.mark.parametrize("field", ["positions", "origin", "forward"])
def test_reject_invalid_numbers(value, field):
    inputs = dict(positions=np.zeros((2, 1, 2)), origin=np.zeros(2), forward=np.ones(2))
    inputs[field].flat[0] = value
    with pytest.raises(ValueError):
        first_directed_plane_crossing(**inputs)


@pytest.mark.parametrize("shape", [(1, 1, 2), (2, 0, 2), (2, 1, 3), (2, 2)])
def test_reject_wrong_trace_shape(shape):
    with pytest.raises(ValueError):
        evaluate(np.zeros(shape))


@pytest.mark.parametrize("axis", [np.zeros(2), np.ones(3), np.array([True, False]), [1.0, 0.0]])
def test_reject_invalid_axis(axis):
    with pytest.raises(ValueError):
        evaluate([[[-1, 0]], [[1, 0]]], forward=axis)
