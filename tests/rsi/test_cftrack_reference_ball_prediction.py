import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_reference_ball_prediction import predict_ball_reference


def test_prediction_is_bounded_and_does_not_modify_measurement():
    position = np.array([1.0, 0.1, 0.11])
    velocity = np.array([0.0, 0.12, 0.0])
    before = position.copy(), velocity.copy()
    predicted, receipt = predict_ball_reference(position, velocity, horizon_s=1.74)
    np.testing.assert_allclose(predicted, [1.0, 0.3088, 0.11])
    np.testing.assert_array_equal(position, before[0])
    np.testing.assert_array_equal(velocity, before[1])
    assert receipt["prediction_is_not_future_observation"]
    assert not receipt["physical_state_modified"]


@pytest.mark.parametrize(
    "velocity,horizon",
    [([0.21, 0, 0], 1.74), ([0, 0, 0.01], 1.74), ([0, 0, 0], 2.01), ([0.2, 0, 0], 1.74)],
)
def test_rejects_excess_speed_vertical_motion_horizon_or_displacement(velocity, horizon):
    with pytest.raises(ValueError):
        predict_ball_reference(
            np.array([1.0, 0.0, 0.11]), np.array(velocity, dtype=float), horizon_s=horizon
        )
