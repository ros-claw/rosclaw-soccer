from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.training.active_team_probe import engagement_rows


def test_engagement_uses_actual_motion_and_counts_continuous_idle_time() -> None:
    trace = {
        "time": np.arange(5, dtype=float),
        "red_player_pelvis_pose": np.asarray(((0, 0), (0.2, 0), (0.4, 0), (0.4, 0), (0.4, 0))),
        "ball_contact_agent_code": np.asarray((0, 0, 1, 0, 0)),
    }
    row = engagement_rows(trace, ("red.player",))[0]
    assert row["moving_fraction"] == pytest.approx(0.5)
    assert row["path_length_m"] == pytest.approx(0.4)
    assert row["longest_stationary_sec"] == pytest.approx(2.0)
    assert row["physical_ball_contact_frames"] == 1


def test_engagement_rejects_duplicate_timestamps() -> None:
    with pytest.raises(ValueError, match="increasing"):
        engagement_rows({"time": np.asarray((0.0, 0.0))}, ())
