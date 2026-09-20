from copy import deepcopy

import pytest

from rosclaw_soccer.training.skillcorner_waypoints import skillcorner_waypoint_pair


def frames():
    a = dict(
        frame=100,
        period=1,
        ball_data=dict(x=10.0, y=5.0, is_detected=True),
        player_data=[dict(player_id=1, x=0.0, y=0.0, is_detected=True)],
    )
    b = deepcopy(a)
    b["frame"] = 105
    b["player_data"][0]["x"] = 1.0
    return a, b


def test_measured_target_and_no_fabricated_intent():
    a, b = frames()
    (row,) = skillcorner_waypoint_pair(a, b, pitch_length=100, pitch_width=50)
    assert row["ball_relative_normalized"] == [0.1, 0.1]
    assert row["waypoint_delta_normalized"] == [0.01, 0.0]
    assert row["intent_label"] is None


@pytest.mark.parametrize("change", ["ball_missing", "undetected", "period", "gap", "coordinate"])
def test_no_zero_fill_or_cross_period(change):
    a, b = frames()
    if change == "ball_missing":
        a["ball_data"]["x"] = None
    if change == "undetected":
        b["player_data"][0]["is_detected"] = False
    if change == "period":
        b["period"] = 2
    if change == "gap":
        b["frame"] = 106
    if change == "coordinate":
        b["player_data"][0]["x"] = float("nan")
    assert skillcorner_waypoint_pair(a, b, pitch_length=100, pitch_width=50) == ()


def test_duplicate_player_rejected():
    a, b = frames()
    a["player_data"] *= 2
    with pytest.raises(ValueError):
        skillcorner_waypoint_pair(a, b, pitch_length=100, pitch_width=50)


def test_invalid_units_rejected():
    a, b = frames()
    with pytest.raises(ValueError):
        skillcorner_waypoint_pair(a, b, pitch_length=float("inf"), pitch_width=50)
