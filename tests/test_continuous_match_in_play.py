from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.world.match_boundary import uninterrupted_in_play


def _trace() -> dict[str, np.ndarray]:
    ball = np.zeros((5, 7))
    ball[:, 2] = 0.11
    ball[:, 3] = 1
    return {
        "time": np.arange(1, 6, dtype=np.float64) * 0.02,
        "ball_pose": ball,
        "pitch_boundary_geometry": np.tile((-1.5, 7.5, 3.0, 0.11, 3.0, 2.0), (5, 1)),
    }


def test_in_play_does_not_erase_the_first_exit_when_ball_returns() -> None:
    trace = _trace()
    assert uninterrupted_in_play(trace)["uninterrupted"] is True
    trace["ball_pose"][2, 1] = 3.12
    result = uninterrupted_in_play(trace)
    assert result["uninterrupted"] is False
    assert result["first_exit_frame"] == 2
    assert result["first_exit_reason"] == "TOUCHLINE_OUT"


def test_goal_at_last_frame_is_terminal_not_continuous() -> None:
    trace = _trace()
    trace["ball_pose"][-1, 0] = 7.62
    result = uninterrupted_in_play(trace)
    assert result["first_exit_reason"] == "RED_GOAL_CROSSING"
    assert result["uninterrupted"] is False


@pytest.mark.parametrize("field", ["time", "ball_pose", "pitch_boundary_geometry"])
def test_in_play_rejects_missing_or_nonfinite_evidence(field: str) -> None:
    trace = _trace()
    del trace[field]
    with pytest.raises(ValueError):
        uninterrupted_in_play(trace)
    trace = _trace()
    trace[field].flat[0] = np.nan
    with pytest.raises(ValueError):
        uninterrupted_in_play(trace)


def test_in_play_rejects_changed_or_incompatible_geometry() -> None:
    trace = _trace()
    trace["pitch_boundary_geometry"][2, 1] += 1
    with pytest.raises(ValueError):
        uninterrupted_in_play(trace)
    trace = _trace()
    trace["pitch_boundary_geometry"][:, 2] = 4
    with pytest.raises(ValueError):
        uninterrupted_in_play(trace)
