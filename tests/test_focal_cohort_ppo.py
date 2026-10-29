import numpy as np
import pytest

from scripts.rsi_focal_cohort_ppo import complete_window_segments


def test_complete_windows_are_cut_from_later_censored_rewards():
    n, roles = 220, ("blue.playmaker", "red.finisher")
    trace = {
        "time": np.arange(n) * 0.02,
        "marker": np.arange(n),
        "residual_active": np.ones((n, 2), dtype=bool),
        "residual_exploration_mask": np.ones((n, 2), dtype=bool),
        "residual_pd_target_injected_substeps": np.full((n, 2), 10),
        "residual_pd_target_override_substeps": np.zeros((n, 2), dtype=int),
    }
    trace["residual_active"][5, 1] = False
    trace["residual_pd_target_injected_substeps"][6, 1] = 9
    trials = [
        {"agent_id": "red.finisher", "censored": False, "start_frame": 1, "frames": 100},
        {"agent_id": "red.finisher", "censored": True, "start_frame": 101, "frames": 119},
    ]
    segments = complete_window_segments(
        trace, agent_ids=roles, agent_id="red.finisher", trials=trials
    )
    assert len(segments) == 1 and segments[0]["marker"].tolist() == list(range(101))
    mask = segments[0]["residual_learning_mask"]
    assert not mask[0].any() and not mask[:, 0].any()
    assert mask[:, 1].sum() == 98 and not mask[5, 1] and not mask[6, 1]
    trace["residual_pd_target_override_substeps"][7, 1] = 1
    with pytest.raises(ValueError):
        complete_window_segments(trace, agent_ids=roles, agent_id="red.finisher", trials=trials)


def test_invalid_complete_window_fails_closed():
    trace = {"time": np.arange(100)}
    trials = [{"agent_id": "red.finisher", "censored": False, "start_frame": 1, "frames": 100}]
    with pytest.raises(ValueError):
        complete_window_segments(
            trace, agent_ids=("red.finisher",), agent_id="red.finisher", trials=trials
        )
