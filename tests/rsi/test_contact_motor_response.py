import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_motor_response import precontact_response


def trace():
    return {
        "root_pose_xyzw_m": np.zeros((300, 1, 7)),
        "joint_target_rad": np.zeros((300, 1, 29)),
        "foot_geometry_position_before_step_m": np.zeros((300, 1, 4, 3)),
        "ball_position_before_step_m": np.zeros((300, 1, 3)),
    }


def test_control_response_excludes_later_contact_cascade_and_uses_earlier_branch():
    base = trace()
    candidate = copy.deepcopy(base)
    candidate["root_pose_xyzw_m"][40:71, 0, 0] = 0.03
    candidate["foot_geometry_position_before_step_m"][40:71, 0, 0, 0] = 0.05
    candidate["joint_target_rad"][40:71, 0, 0] = 0.1
    candidate["root_pose_xyzw_m"][71:, 0, 0] = 100
    result = precontact_response(base, candidate, 80, 70)
    assert result["first_joint_target_difference_frame"] == 40
    assert result["max_root_xy_response_before_contact_m"] == 0.03
    assert result["max_foot_link_response_before_contact_m"] == 0.05
    assert result["last_compared_precontact_frame"] == 70


def test_no_contact_includes_full_course_but_bad_frame_is_rejected():
    base = trace()
    assert precontact_response(base, base, None, None)["last_compared_precontact_frame"] == 299
    assert precontact_response(base, base, 0, None)["last_compared_precontact_frame"] == 0
    with pytest.raises(ValueError):
        precontact_response(base, base, -1, None)
