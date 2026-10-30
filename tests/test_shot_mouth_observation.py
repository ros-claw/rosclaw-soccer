from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.world.shot_mouth_observation import observe_shot_mouth_crossing


def trajectory(positions: list[tuple[float, float, float]]) -> dict[str, np.ndarray]:
    count = len(positions)
    return {
        "time": np.arange(count, dtype=np.float64) * 0.02,
        "ball_pose": np.column_stack(
            (np.asarray(positions, dtype=np.float64), np.zeros((count, 4)))
        ),
        "pitch_boundary_geometry": np.tile((-1.5, 7.5, 3.0, 0.11, 3.0, 2.0), (count, 1)),
    }


def test_measured_left_crossing_preserves_ball_size_and_target_error() -> None:
    data = trajectory([(0.0, 0.0, 0.5), (-1.0, -1.5, 0.5), (-2.0, -1.7, 0.5)])
    result = observe_shot_mouth_crossing(
        data, shot_frame=0, goal_side="left", target_yz_m=(0.0, 0.5)
    )
    assert result["crossed"] is True
    assert result["crossing_frame_before"] == 1
    assert result["crossing_center_yz_m"] == pytest.approx([-1.6, 0.5])
    assert result["whole_ball_lateral_clearance_m"] == pytest.approx(-0.21)
    assert result["target_error_m"] == pytest.approx(1.6)
    assert result["center_crossing_fits_mouth"] is False
    assert result["goal_scored"] is None


def test_right_crossing_and_no_fabricated_future() -> None:
    data = trajectory([(6.0, 0.0, 0.5), (7.0, 0.2, 0.5), (8.0, 0.4, 0.5)])
    result = observe_shot_mouth_crossing(
        data, shot_frame=0, goal_side="right", target_yz_m=(0.3, 0.5)
    )
    assert result["crossed"] is True
    assert result["target_error_m"] == pytest.approx(0.0)
    assert result["center_crossing_fits_mouth"] is True
    missing = observe_shot_mouth_crossing(
        data, shot_frame=0, goal_side="left", target_yz_m=(0.0, 0.5)
    )
    assert missing["crossed"] is False
    assert missing["target_error_m"] is None
    assert missing["goal_scored"] is None


def test_prior_crossing_is_not_attributed_to_a_later_shot() -> None:
    data = trajectory([(0.0, 0.0, 0.5), (-2.0, 0.0, 0.5), (0.0, 0.0, 0.5), (0.1, 0.0, 0.5)])
    result = observe_shot_mouth_crossing(
        data, shot_frame=2, goal_side="left", target_yz_m=(0.0, 0.5)
    )
    assert result["crossed"] is False


def test_invalid_or_moving_geometry_and_past_plane_fail_closed() -> None:
    data = trajectory([(0.0, 0.0, 0.5), (-2.0, 0.0, 0.5)])
    with pytest.raises(ValueError, match="inside"):
        observe_shot_mouth_crossing(data, shot_frame=1, goal_side="left", target_yz_m=(0.0, 0.5))
    with pytest.raises(ValueError, match="target"):
        observe_shot_mouth_crossing(data, shot_frame=0, goal_side="left", target_yz_m=(1.5, 0.5))
    data["pitch_boundary_geometry"][1, 3] = 0.12
    with pytest.raises(ValueError, match="constant geometry"):
        observe_shot_mouth_crossing(data, shot_frame=0, goal_side="left", target_yz_m=(0.0, 0.5))


def test_nonfinite_state_and_unordered_time_rejected() -> None:
    data = trajectory([(0.0, 0.0, 0.5), (-2.0, 0.0, 0.5)])
    data["ball_pose"][1, 1] = np.nan
    with pytest.raises(ValueError, match="finite aligned"):
        observe_shot_mouth_crossing(data, shot_frame=0, goal_side="left", target_yz_m=(0.0, 0.5))
    data["ball_pose"][1, 1] = 0.0
    data["time"][1] = data["time"][0]
    with pytest.raises(ValueError, match="finite aligned"):
        observe_shot_mouth_crossing(data, shot_frame=0, goal_side="left", target_yz_m=(0.0, 0.5))
