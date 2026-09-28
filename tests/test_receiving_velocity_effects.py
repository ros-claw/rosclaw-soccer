import numpy as np
import pytest

from rosclaw_soccer.training.receiving_velocity_effects import receiving_velocity_effects


def _trace():
    frames = 8
    ball_velocity = np.zeros((frames, 6))
    ball_velocity[:, 0] = -1.0
    ball_velocity[3:, :2] = (-0.4, -0.3)
    contact = np.zeros(frames, dtype=int)
    contact[2:4] = 6
    foot = np.zeros(frames, dtype=int)
    foot[2:4] = 1
    ball_pose = np.zeros((frames, 7))
    ball_pose[:, 2] = 0.11
    foot_position = np.zeros((frames, 3))
    foot_position[:, 0] = 0.2
    return {
        "ball_contact_agent_code": contact,
        "ball_contact_foot_code": foot,
        "ball_nonfoot_contact_agent_code": np.zeros(frames, dtype=int),
        "ball_pose": ball_pose,
        "ball_velocity": ball_velocity,
        "red_finisher_pelvis_pose": np.zeros((frames, 7)),
        "red_finisher_left_foot_position": foot_position,
        "red_finisher_right_foot_position": foot_position + 1,
    }


def test_receiving_contact_diagnostics_separates_retained_and_lateral_velocity():
    row = receiving_velocity_effects(
        _trace(), agent_id="red.finisher", agent_code=6, tail=slice(5, 8)
    )
    assert row["first_foot_frame"] == 2
    assert row["first_contact_end_frame"] == 3
    assert row["observed_parallel_speed_removed_mps"] == pytest.approx(0.6)
    assert abs(row["outgoing_lateral_mps"]) == pytest.approx(0.3)
    assert not row["task_contact_diagnostic_passed"]
    assert not row["promotion_authorized"]


def test_receiving_contact_diagnostics_rejects_corrupt_trace():
    trace = _trace()
    trace["ball_velocity"][3, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        receiving_velocity_effects(trace, agent_id="red.finisher", agent_code=6, tail=slice(5, 8))


def test_receiving_contact_diagnostics_rejects_missing_contact():
    trace = _trace()
    trace["ball_contact_agent_code"][:] = 0
    with pytest.raises(ValueError, match="own-foot"):
        receiving_velocity_effects(trace, agent_id="red.finisher", agent_code=6, tail=slice(5, 8))
