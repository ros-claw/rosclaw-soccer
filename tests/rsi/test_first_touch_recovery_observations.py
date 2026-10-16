import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_recovery_observations import observe_first_touch_recovery


def measured(first=40):
    result = dict(
        force_n=np.zeros((300, 1, 6)),
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.7),
        ball_position_after_step_m=np.zeros((300, 1, 3)),
        root_pose_xyzw_m=np.zeros((300, 1, 7)),
        root_velocity_world=np.zeros((300, 1, 6)),
        joint_velocity_rad_s=np.zeros((300, 1, 29)),
    )
    result["root_pose_xyzw_m"][:, 0, 6] = 1.0
    if first is not None:
        result["force_n"][first, 0, 0] = 2.0
    return result


def test_retreat_is_maximum_drawdown_not_only_net_displacement():
    trace = measured()
    trace["root_pose_xyzw_m"][40:43, 0, 0] = [0.0, 1.0, 0.6]
    trace["root_pose_xyzw_m"][43:, 0, 0] = 2.0
    trace["root_velocity_world"][42, 0, 0] = -0.3
    before = copy.deepcopy(trace)
    result = observe_first_touch_recovery(trace)
    row = result["contact_to_plus60"]
    assert row["root_world_x_net_displacement_m"] == 2.0
    assert row["root_world_x_maximum_drawdown_m"] == pytest.approx(0.4)
    assert row["root_world_x_minimum_recorded_velocity_m_s"] == -0.3
    assert row["pre_step_body_snapshots"] == 61
    assert row["body_snapshot_span_s"] == 1.2
    for key in trace:
        np.testing.assert_array_equal(trace[key], before[key])
    assert result["task_or_reward_changed"] is False
    assert result["naturalness_or_successor_readiness_qualified"] is False


def test_tilt_uses_rotation_not_quaternion_sign_and_keeps_units():
    trace = measured()
    trace["root_pose_xyzw_m"][42, 0, 3:7] = [np.sin(0.2), 0, 0, np.cos(0.2)]
    trace["root_pose_xyzw_m"][43, 0, 3:7] *= -1
    trace["root_velocity_world"][44, 0, 3:] = [3, 4, 0]
    trace["joint_velocity_rad_s"][45:, 0, 0] = 2.0
    row = observe_first_touch_recovery(trace)["contact_to_plus60"]
    assert row["root_maximum_upright_tilt_rad"] == pytest.approx(0.4)
    assert row["root_maximum_recorded_world_angular_speed_rad_s"] == 5.0
    assert row["maximum_joint_velocity_step_l2_change_rad_s"] == 2.0


@pytest.mark.parametrize(
    "first,complete,last", [(0, True, 60), (239, True, 299), (240, False, 299), (260, False, 299)]
)
def test_contact_relative_window_and_censoring(first, complete, last):
    result = observe_first_touch_recovery(measured(first))
    row = result["contact_to_plus60"]
    assert row["complete_requested_window"] is complete
    assert row["last_frame"] == last
    assert row["first_frame"] == first
    assert result["outcome"]["quality_observed"] is complete


@pytest.mark.parametrize("first", [None, 299])
def test_missing_body_tail_is_none_not_fabricated_zero(first):
    result = observe_first_touch_recovery(measured(first))
    assert result["contact_to_plus60"] is None
    assert result["contact_to_episode_end"] is None


@pytest.mark.parametrize("key", ["root_pose_xyzw_m", "root_velocity_world", "joint_velocity_rad_s"])
def test_nonfinite_or_short_measurements_rejected(key):
    trace = measured()
    trace[key][5, 0, 0] = np.nan
    with pytest.raises(ValueError, match="recovery"):
        observe_first_touch_recovery(trace)
    trace = measured()
    trace[key] = trace[key][:-1]
    with pytest.raises(ValueError, match="recovery"):
        observe_first_touch_recovery(trace)


def test_invalid_quaternion_is_not_silently_normalized():
    trace = measured()
    trace["root_pose_xyzw_m"][0, 0, 6] = 2.0
    with pytest.raises(ValueError, match="quaternions"):
        observe_first_touch_recovery(trace)


def test_unsafe_result_stays_unsafe_even_if_motion_is_constant():
    trace = measured()
    trace["force_n"][50, 0, 4] = 2.0
    result = observe_first_touch_recovery(trace)
    assert result["contact_to_plus60"]["root_world_x_maximum_drawdown_m"] == 0.0
    assert result["outcome"]["safe"] is False
    assert result["outcome"]["task_pass"] is False
