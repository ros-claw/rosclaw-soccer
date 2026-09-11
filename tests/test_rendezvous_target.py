import math

import pytest

from rosclaw_soccer.training.rendezvous_target import predict_ground_rendezvous


def case(**overrides):
    values = dict(
        ball_xy=(1.0, 0.0),
        ball_velocity_xy=(1.0, 0.0),
        passer_xy=(2.0, 0.0),
        passer_velocity_xy=(0.0, 0.0),
        receiver_xy=(0.0, 0.0),
        receiver_velocity_xy=(0.0, 0.0),
        contact_standoff_m=0.2,
        outgoing_speed_mps=1.3,
    )
    values.update(overrides)
    return predict_ground_rendezvous(**values)


def test_stationary_receiver_is_not_shifted():
    result = case()
    assert result.incoming_contact_delay_sec == pytest.approx(0.8)
    assert result.predicted_contact_xy == pytest.approx((1.8, 0))
    assert result.target_xy == (0.0, 0.0)
    assert result.outgoing_flight_sec == pytest.approx(1.8 / 1.3)


def test_moving_receiver_solution_matches_flight_and_target_motion():
    result = case(receiver_velocity_xy=(0.1, 0.2))
    assert result.target_xy == pytest.approx((0.1 * result.horizon_sec, 0.2 * result.horizon_sec))
    assert math.dist(result.target_xy, result.predicted_contact_xy) == pytest.approx(
        result.outgoing_flight_sec * 1.3
    )


@pytest.mark.parametrize(
    "values",
    [
        {"ball_velocity_xy": (-1.0, 0.0)},
        {"ball_velocity_xy": (float("nan"), 0.0)},
        {"receiver_velocity_xy": (1.3, 0.0)},
        {"maximum_horizon_sec": 0.2},
        {"contact_standoff_m": True},
        {"receiver_xy": [0.0, 0.0]},
    ],
)
def test_unusable_prediction_fails_closed(values):
    with pytest.raises(ValueError):
        case(**values)
