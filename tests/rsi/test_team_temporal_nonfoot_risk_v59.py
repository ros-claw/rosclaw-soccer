"""Temporal contact diagnosis is causal and counts false alerts by episode."""

import numpy as np
import pytest

from scripts.rsi_team_temporal_nonfoot_risk_v59 import _causal_feature, _diagnose, _future_event


def test_feature_uses_past_ball_motion_only() -> None:
    trace = {
        "pre_step_ball_position_local_m": np.zeros((3, 1, 3)),
        "pre_step_foot_link_position_w": np.zeros((3, 1, 2, 3)),
        "pre_step_foot_linear_velocity_w": np.zeros((3, 1, 2, 3)),
        "pre_step_focal_qpos": np.zeros((3, 1, 43)),
        "pre_step_focal_qvel": np.zeros((3, 1, 41)),
        "taskspace_selected_side": np.zeros((3, 1)),
    }
    trace["pre_step_ball_position_local_m"][1, 0, 0] = 0.02
    first = _causal_feature(trace, 1)
    assert first.shape == (29,)
    assert first[9] == pytest.approx(1.0)
    trace["pre_step_ball_position_local_m"][2, 0, 0] = 100.0
    assert np.array_equal(first, _causal_feature(trace, 1))


def test_diagnose_does_not_hide_clean_episode_false_alert() -> None:
    result = _diagnose(
        np.asarray([0.5, 0.8, 0.1, 0.6]),
        np.asarray([False, True, False, False]),
        np.asarray([1, 1, 2, 3]),
        0.4,
    )
    assert result["event_episodes_with_alert"] == 1
    assert result["clean_episodes_with_false_alert"] == 1
    assert result["false_alert_frames"] == 2


def test_no_event_sentinel_never_becomes_a_positive_at_episode_end() -> None:
    assert not _future_event(250, 248, 10)
    assert _future_event(249, 240, 10)
    assert not _future_event(249, 238, 10)
