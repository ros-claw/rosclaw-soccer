import numpy as np
import pytest

from rosclaw_soccer.rsi.team_trace_capture import ROSTER, extract_team_trace


def recorded():
    trace = {
        "time": np.arange(3, dtype=float),
        "ball_pose": np.zeros((3, 7)),
        "ball_velocity": np.zeros((3, 6)),
    }
    for player in ROSTER:
        trace[f"{player}_pelvis_pose"] = np.zeros((3, 7))
        for field in ("joint_position", "joint_velocity", "joint_torque"):
            trace[f"{player}_{field}"] = np.zeros((3, 29))
    return trace


def test_full_team_snapshots_are_copied():
    original = recorded()
    data = extract_team_trace(original)
    assert len(data) == 35
    original["ball_pose"][0, 0] = 7
    assert data["ball_pose"][0, 0] == 0


def test_rejects_missing_or_invalid_player():
    trace = recorded()
    del trace["blue_finisher_joint_position"]
    with pytest.raises(ValueError):
        extract_team_trace(trace)
    trace = recorded()
    trace["red_goalkeeper_joint_torque"][0, 0] = np.nan
    with pytest.raises(ValueError):
        extract_team_trace(trace)
