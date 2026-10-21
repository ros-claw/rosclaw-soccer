import numpy as np
import pytest

from rosclaw_soccer.rsi.precontact_curriculum import select_precontact_frames


def arrays():
    q = np.zeros((220, 43))
    q[:, 9] = 0.8
    q[:, 10] = 1
    feet = np.ones((220, 2, 3))
    feet[120, 0] = 0.01
    return dict(
        source="soccer-standard-002_left.npz",
        foot_force=np.zeros((220, 10, 2)),
        nonfoot_force=np.zeros((220, 10)),
        ball_to_feet=feet,
        native_qpos=q,
        expected_foot=0,
        pelvis_qadr=7,
    )


def test_before_knee_failure_is_not_post_contact_or_expert_label():
    a = arrays()
    a["nonfoot_force"][100, 4] = 20
    a["foot_force"][120, 2, 0] = 50
    d = select_precontact_frames(**a)
    assert d["first_robot_contact_substep"] == 1004
    assert [c["frame"] for c in d["cases"]] == [90, 80, 60, 20]
    assert all(c["expert_success_label"] is False for c in d["cases"])


def test_near_miss_is_kept_without_fabricating_contact():
    d = select_precontact_frames(**arrays())
    assert d["anchor_frame"] == 120
    assert d["first_robot_contact_substep"] is None
    assert len(d["cases"]) == 4


def test_reject_dev_invalid_forces_and_unstable_initial_pose():
    a = arrays()
    a["source"] = "soccer-standard-007_left.npz"
    with pytest.raises(ValueError, match="TRAIN8"):
        select_precontact_frames(**a)
    a = arrays()
    a["foot_force"][10, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        select_precontact_frames(**a)
    a = arrays()
    a["native_qpos"][:, 9] = 0.5
    d = select_precontact_frames(**a)
    assert not d["cases"] and len(d["rejected"]) == 4


def test_early_collision_does_not_invent_negative_frames():
    a = arrays()
    a["foot_force"][5, 0, 0] = 2
    d = select_precontact_frames(**a)
    assert not d["cases"]
    assert all(c["reason"] == "BEFORE_EPISODE_START" for c in d["rejected"])
