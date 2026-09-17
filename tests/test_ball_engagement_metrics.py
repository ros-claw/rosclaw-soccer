import numpy as np
import pytest

from rosclaw_soccer.training.ball_engagement_metrics import ball_engagement_metrics


def fixture():
    n = 20
    return dict(
        time=np.arange(n) * 0.02,
        player_xy=np.zeros((n, 2)),
        ball_xy=np.tile([1.0, 0.0], (n, 1)),
        eligible=np.ones(n, dtype=bool),
        foot_contact=np.zeros(n, dtype=bool),
    )


def test_ball_flight_does_not_count_as_player_response():
    x = fixture()
    x["ball_xy"][:, 0] = np.linspace(1, 0.2, 20)
    result = ball_engagement_metrics(**x)
    assert result["responded_count"] == result["contacted_count"] == 0
    assert result["opportunities"][0]["own_toward_ball_distance_m"] == 0


def test_measured_approach_and_contact_are_separate():
    x = fixture()
    x["player_xy"][:, 0] = x["time"] * 0.5
    x["foot_contact"][10] = True
    row = ball_engagement_metrics(**x)["opportunities"][0]
    assert row["sustained_approach_latency_sec"] == pytest.approx(0.06)
    assert row["first_foot_contact_latency_sec"] == pytest.approx(0.2)


def test_no_cross_task_or_external_reset_credit():
    x = fixture()
    x["eligible"][:10] = False
    x["player_xy"][10:, 0] = 0.8
    x["foot_contact"][9] = True
    result = ball_engagement_metrics(**x)
    assert result["responded_count"] == result["contacted_count"] == 0
    assert result["opportunities"][0]["own_toward_ball_distance_m"] == 0


def test_away_motion_and_single_frame_burst_not_response():
    x = fixture()
    x["player_xy"][:, 0] = -x["time"]
    row = ball_engagement_metrics(**x)["opportunities"][0]
    assert row["own_away_from_ball_distance_m"] > 0
    assert row["sustained_approach_latency_sec"] is None
    x = fixture()
    x["player_xy"][3, 0] = 0.1
    assert ball_engagement_metrics(**x)["responded_count"] == 0


@pytest.mark.parametrize("bad", ["time_gap", "nan", "mask"])
def test_invalid_measurement_rejected(bad):
    x = fixture()
    if bad == "time_gap":
        x["time"][10:] += 0.02
    elif bad == "nan":
        x["player_xy"][0, 0] = np.nan
    else:
        x["eligible"] = x["eligible"].astype(int)
    with pytest.raises(ValueError):
        ball_engagement_metrics(**x)


def test_no_task_is_not_perfect_reaction():
    x = fixture()
    x["eligible"][:] = False
    result = ball_engagement_metrics(**x)
    assert result["opportunity_count"] == result["responded_count"] == 0
