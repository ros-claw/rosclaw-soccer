"""Physical CEM reward favors clean foot contact over hard shank collisions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.rsi_snapshot_cem_train import score_replay


def test_cem_score_counts_independent_lanes_and_root_guard(tmp_path: Path) -> None:
    force = np.zeros((12, 2, 6))
    force[2, 0, 1] = 2.0
    force[2, 1, 5] = 8.0
    ball = np.zeros((12, 2, 3))
    ball[8, :, 0] = 0.2
    root = np.zeros((12, 2, 7))
    root[:, :, 2] = 0.75
    trace = tmp_path / "trace.npz"
    np.savez_compressed(
        trace,
        observed_ball_body_contact_force_peak_n=force,
        observed_ball_position_local_m=ball,
        observed_root_pose_local_xyzw_m=root,
    )
    score = score_replay(trace)
    assert score["clean_foot_count"] == 1
    assert score["first_foot_count"] == 1
    assert score["sample_count"] == 2
    assert score["mean_reward"] > 0
    root[:, :, 2] = 0.5
    np.savez_compressed(
        trace,
        observed_ball_body_contact_force_peak_n=force,
        observed_ball_position_local_m=ball,
        observed_root_pose_local_xyzw_m=root,
    )
    with pytest.raises(ValueError, match="unsafe"):
        score_replay(trace)
