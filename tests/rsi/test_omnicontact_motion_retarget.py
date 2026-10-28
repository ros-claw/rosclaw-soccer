import numpy as np
import pytest

from rosclaw_soccer.rsi.omnicontact_motion_retarget import retarget_window


def motion():
    n = 181
    shapes = {
        "joint_pos": (29,),
        "joint_vel": (29,),
        "body_pos_w": (39, 3),
        "body_quat_w": (39, 4),
        "body_lin_vel_w": (39, 3),
        "body_ang_vel_w": (39, 3),
        "ee_pos_w": (5, 3),
        "ee_quat_w": (5, 4),
        "object_pos_w": (3,),
        "object_quat_w": (4,),
        "object_lin_vel_w": (3,),
        "object_ang_vel_w": (3,),
        "contact_info": (4, 1),
    }
    result = {key: np.zeros((n, *shape), dtype=np.float32) for key, shape in shapes.items()}
    for key in ("body_quat_w", "ee_quat_w", "object_quat_w"):
        result[key][..., 0] = 1
    result["joint_pos"][:, 0] = np.arange(n) / 90
    result["object_pos_w"][:, 2] = 0.05
    result["contact_info"][90, 0, 0] = 1
    return result


def test_retarget_preserves_timing_and_marks_reference_lift():
    source = motion()
    result = retarget_window(source, event=90, source_fps=90, ball_lift_m=0.06)
    assert result["joint_pos"].shape == (101, 29)
    assert result["joint_pos"][50, 0] == pytest.approx(1.0)
    assert result["object_pos_w"][50, 2] == pytest.approx(0.11)
    assert source["object_pos_w"][90, 2] == pytest.approx(0.05)
    assert result["contact_info"][50, 0, 0] == 1
    np.testing.assert_allclose(result["body_quat_w"][..., 0], 1)


def test_retarget_refuses_truncated_window_and_invalid_rotation():
    source = motion()
    with pytest.raises(ValueError, match="full two-second"):
        retarget_window(source, event=89, source_fps=90, ball_lift_m=0.06)
    source["object_quat_w"][90] = 0
    with pytest.raises(ValueError, match="unit source quaternion"):
        retarget_window(source, event=90, source_fps=90, ball_lift_m=0.06)
