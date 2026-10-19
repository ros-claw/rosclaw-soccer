from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.reference_ball_strike_metrics import summarize_reference_ball_strike


def inputs():
    names = [
        "world",
        "pelvis",
        "left_ankle_roll_link",
        "left_ankle_pitch_link",
        "right_ankle_roll_link",
        "right_ankle_pitch_link",
        "left_elbow",
        "ball",
    ]
    model = SimpleNamespace(
        nbody=8,
        body_rootid=np.array([0, 1, 1, 1, 1, 1, 1, 7]),
        body=lambda name: SimpleNamespace(id=names.index(name)),
    )
    arrays = dict(
        ball_contact_force_by_body_n=np.zeros((300, 10, 8)),
        target_body_canonical_qpos_after_substep=np.zeros((300, 10, 43)),
        target_body_canonical_qvel_after_substep=np.zeros((300, 10, 41)),
        canonical_qpos=np.zeros((300, 43)),
    )
    return model, arrays


def test_floor_ignored_but_elbow_is_counted_and_contact_timing_is_native():
    model, arrays = inputs()
    force = arrays["ball_contact_force_by_body_n"]
    force[:, :, 0] = 10
    force[3, 8, 2] = 5
    force[3, 9, 6] = 8
    arrays["target_body_canonical_qpos_after_substep"][-1, -1, 36] = 2
    report = summarize_reference_ball_strike(model, arrays)
    assert report["first_foot_contact_native_index"] == 38
    assert report["first_nonfoot_robot_contact_native_index"] == 39
    assert report["maximum_nonfoot_robot_force_n"] == 8
    assert report["final_ball_displacement_m"] == [2, 0, 0]
    assert report["official_football_exam_pass"] is False


def test_missing_force_channels_cannot_claim_contact_complete():
    model, arrays = inputs()
    arrays["ball_contact_force_by_body_n"] = np.zeros((300, 10, 6))
    with pytest.raises(ValueError, match="complete"):
        summarize_reference_ball_strike(model, arrays)
