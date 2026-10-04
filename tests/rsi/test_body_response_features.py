import numpy as np
import pytest

from rosclaw_soccer.rsi.body_response_features import (
    central_response_labels,
    local_velocity_response,
    measured_response_features,
)


def test_current_features_are_owned_and_central_labels_linear_zero_exact():
    q = np.zeros((2, 43))
    q[:, 2] = 0.7
    q[:, 3] = 1
    v = np.zeros((2, 41))
    target = np.zeros((2, 29))
    ball = np.ones((2, 6))
    features = measured_response_features(q, v, target, ball)
    assert features.shape == (2, 103)
    np.testing.assert_array_equal(
        features[:, :4], np.asarray([[0.7, 0, 0, -1]] * 2, dtype=np.float32)
    )
    q[:, 2] += 1
    assert features[0, 0] == pytest.approx(0.7)
    jac = np.arange(420).reshape(35, 12) / 100
    coordinates = np.asarray([(f, j, s) for f in (40, 80) for j in range(12) for s in (-1, 1)])
    effects = np.asarray([s * 0.01 * jac[:, j] for _, j, s in coordinates])
    labels = central_response_labels(effects, coordinates, increment_rad=0.01)
    np.testing.assert_allclose(labels, np.tile(jac.reshape(1, 420), (2, 1)), atol=1e-6, rtol=0)
    zero = local_velocity_response(labels, np.zeros((2, 12)))
    assert np.array_equal(zero, np.zeros((2, 35)))
    delta = np.zeros((2, 12))
    delta[:, 4] = 0.02
    np.testing.assert_allclose(
        local_velocity_response(labels, delta), np.tile(jac[:, 4] * 0.02, (2, 1)), atol=1e-8
    )
    assert not features.flags.writeable and not labels.flags.writeable and not zero.flags.writeable


def test_type_range_order_and_nonfinite_fail_closed():
    coords = np.asarray([(40, j, s) for j in range(12) for s in (-1, 1)])
    effects = np.zeros((24, 35))
    for increment in (True, 0, 0.1, float("nan")):
        with pytest.raises(ValueError):
            central_response_labels(effects, coords, increment_rad=increment)
    with pytest.raises(ValueError):
        central_response_labels(effects, coords[::-1], increment_rad=0.01)
    for delta in (np.full((1, 12), 0.03), np.full((1, 12), np.nan), np.zeros((1, 12), dtype=bool)):
        with pytest.raises(ValueError):
            local_velocity_response(np.zeros((1, 420)), delta)
    q = np.zeros((1, 43))
    with pytest.raises(ValueError):
        measured_response_features(q, np.zeros((1, 41)), np.zeros((1, 29)), np.zeros((1, 6)))
