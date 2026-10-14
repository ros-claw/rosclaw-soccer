import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_action_observations import observe_action_response


def trace():
    value = {
        "force_n": np.zeros((300, 1, 6)),
        "pelvis_z_per_substep_m": np.full((300, 1, 10), 0.7),
        "ball_position_after_step_m": np.zeros((300, 1, 3)),
    }
    for key, shape in {
        "root_pose_xyzw_m": (300, 1, 7),
        "root_velocity_world": (300, 1, 6),
        "ball_position_before_step_m": (300, 1, 3),
        "ball_linear_velocity_before_step_m_s": (300, 1, 3),
        "foot_geometry_position_before_step_m": (300, 1, 4, 3),
        "navigation_command": (300, 1, 3),
        "joint_target_rad": (300, 1, 29),
        "joint_position_rad": (300, 1, 29),
        "motor_delta_rad": (300, 1, 12),
        "torque_nm": (300, 1, 10, 29),
        "actual_actuator_force_nm": (300, 1, 10, 29),
    }.items():
        value[key] = np.zeros(shape)
    value["root_pose_xyzw_m"][:, :, 6] = 1
    value["navigation_command"][:, :, 0] = 1.4
    value["root_velocity_world"][:, :, 0] = 0.7
    value["ball_position_before_step_m"][:, :, 0] = 2
    return value


def test_velocity_is_body_relative_and_no_causality_claim():
    value = trace()
    result = observe_action_response(value)
    assert result["precontact"]["measured_body_forward_mean_m_s"] == pytest.approx(0.7)
    assert result["precontact"]["minimum_ankle_body_center_to_ball_center_distance_m"] == 2
    assert result["causal_authority_proven"] is False
    assert result["complete_requested_to_executed_chain_available"] is False
    value["root_pose_xyzw_m"][:, :, 5:7] = np.sqrt(0.5)
    assert observe_action_response(value)["precontact"][
        "measured_body_forward_mean_m_s"
    ] == pytest.approx(0, abs=1e-12)


def test_contact_at_zero_has_no_fabricated_precontact_mean():
    value = trace()
    value["force_n"][0, 0, 0] = 2
    result = observe_action_response(value)
    assert result["pre_first_foot_contact_frames"] == 0
    assert result["precontact"] is None


def test_nonfinite_action_cannot_be_described_as_zero():
    value = trace()
    value["torque_nm"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        observe_action_response(value)
