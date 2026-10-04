import numpy as np
import pytest

from rosclaw_soccer.rsi.body_ball_response_features import (
    central_body_ball_labels,
    measured_body_ball_features,
)


def state():
    q = np.zeros((2, 43))
    q[:, 2] = 0.7
    q[:, 3] = q[:, 39] = 1
    q[:, 36] = 2
    return q, np.zeros((2, 41)), np.zeros((2, 29))


def test_features_keep_yaw_and_world_ball_spin_without_future_or_global_xy():
    q, v, target = state()
    original = q.copy()
    baseline = measured_body_ball_features(q, v, target)
    assert baseline.shape == (2, 112) and not baseline.flags.writeable
    np.testing.assert_array_equal(q, original)
    q[:, :2] += [5, -3]
    q[:, 36:38] += [5, -3]
    np.testing.assert_array_equal(measured_body_ball_features(q, v, target), baseline)
    q[:, 3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    q[:, 39:43] = q[:, 3:7]
    v[:, 38] = 2
    actual = measured_body_ball_features(q, v, target)
    np.testing.assert_allclose(actual[:, -3:], [[0, 2, 0]] * 2, atol=1e-6, rtol=0)
    assert not np.array_equal(actual[:, 1:10], baseline[:, 1:10])
    q[:, 3:7] *= -1
    q[:, 39:43] *= -1
    np.testing.assert_array_equal(measured_body_ball_features(q, v, target), actual)


@pytest.mark.parametrize("dimensions", [12, 29])
def test_labels_include_actual_ball_effect_and_complete_ordered_pairs(dimensions):
    coords = np.array([(f, j, s) for f in (5, 7) for j in range(dimensions) for s in (-1, 1)])
    effects = np.zeros((len(coords), 38))
    effects[:, 3] = (coords[:, 1] == dimensions - 1) * coords[:, 2] * 0.02
    effects[:, 35] = (coords[:, 1] == dimensions - 1) * coords[:, 2] * 0.03
    result = central_body_ball_labels(
        effects, coords, increment_rad=0.01, action_dimensions=dimensions
    )
    matrix = result.reshape(2, 38, dimensions)
    np.testing.assert_array_equal(matrix[:, 3, -1], [2, 2])
    np.testing.assert_array_equal(matrix[:, 35, -1], [3, 3])
    assert not result.flags.writeable
    with pytest.raises(ValueError):
        central_body_ball_labels(
            effects, coords[::-1], increment_rad=0.01, action_dimensions=dimensions
        )


def test_incomplete_or_nonfinite_physical_features_rejected():
    q, v, target = state()
    for index, value in ((39, 2), (3, 0), (36, np.nan)):
        bad = q.copy()
        bad[:, index] = value
        with pytest.raises(ValueError):
            measured_body_ball_features(bad, v, target)
    with pytest.raises(ValueError):
        measured_body_ball_features(q[:1], v, target)
