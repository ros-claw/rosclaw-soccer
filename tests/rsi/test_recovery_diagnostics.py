"""Synthetic arrays only, no measured physical skill or fall detector."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.recovery_diagnostics import recovery_diagnostics
from tests.rsi.test_full_task_zero_readback import fixture


def trace():
    a = fixture()[2]["main"]
    a["root_pose_xyzw_m"][:, 0, 6] = 1.0
    return a


def test_ball_contact_veto_does_not_mean_fall():
    a = trace()
    a["force_n"][209, 0, 2] = 2.0
    r = recovery_diagnostics(a)
    assert not r["formal_outcome"]["safe"]
    assert r["unsafe_components"] == dict(
        pelvis_below_0_65_m=False, observed_six_channel_nonfoot_ball_contact=True
    )
    assert r["post_contact"]["maximum_root_tilt_deg"] == 0.0
    assert r["post_contact"]["minimum_native_pelvis_z_m"] == 0.79
    assert r["first_observed_nonfoot_ball_contact_frame"] == 209
    assert r["formal_task_or_reward_changed"] is False


def test_backwards_motion_tilt_and_torque_changes_are_only_diagnostics():
    a = trace()
    a["root_pose_xyzw_m"][186:, 0, 0] = -np.arange(114) * 0.01
    a["root_pose_xyzw_m"][186:, 0, 3] = np.sin(np.pi / 8)
    a["root_pose_xyzw_m"][186:, 0, 6] = np.cos(np.pi / 8)
    a["root_velocity_world"][186:, 0, 0] = -0.5
    a["root_velocity_world"][186:, 0, 3] = 2.0
    a["torque_nm"][200, 0, 5, 0] = 10.0
    r = recovery_diagnostics(a)["post_contact"]
    assert r["sampled_root_negative_world_x_travel_m"] == pytest.approx(1.13)
    assert r["maximum_root_tilt_deg"] == pytest.approx(45.0)
    assert r["maximum_world_angular_speed_rad_s"] == 2.0
    assert r["sampled_root_negative_world_x_velocity_frames"] == 114
    assert r["native_mean_abs_torque_change_nm"] > 0


def test_no_contact_no_fabricated_post_contact_metrics():
    a = trace()
    a["force_n"][:] = 0
    assert recovery_diagnostics(a)["post_contact"] is None


def test_last_bin_contact_has_no_future_metrics():
    a = trace()
    a["force_n"][:] = 0
    a["force_n"][299, 0, 0] = 2
    r = recovery_diagnostics(a)
    assert r["post_contact"] is None
    assert r["formal_outcome"]["measurement_status"] == "RIGHT_CENSORED"


def test_bad_quaternion_rejected_not_normalized_silently():
    a = trace()
    a["root_pose_xyzw_m"][100, 0, 6] = 2
    with pytest.raises(ValueError, match="quaternions"):
        recovery_diagnostics(a)
