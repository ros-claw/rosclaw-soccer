import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.postcontact_action_transition import (
    observe_postcontact_action_transition,
)


def measured(first=40, bounded=False):
    trace = dict(
        force_n=np.zeros((300, 1, 6)),
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.7),
        ball_position_after_step_m=np.zeros((300, 1, 3)),
        joint_velocity_rad_s=np.zeros((300, 1, 29)),
        joint_target_rad=np.zeros((300, 1, 29)),
        pre_motor_joint_target_rad=np.zeros((300, 1, 29)),
        motor_delta_rad=np.zeros((300, 1, 12)),
        torque_nm=np.zeros((300, 1, 10, 29)),
        actual_actuator_force_nm=np.zeros((300, 1, 10, 29)),
    )
    if first is not None:
        trace["force_n"][first, 0, 0] = 2.0
    if bounded:
        trace.update(
            foundation_navigation_joint_target_rad=np.zeros((300, 1, 29)),
            taskspace_joint_increment_rad=np.zeros((300, 1, 29)),
            pd_requested_torque_nm=np.zeros((300, 1, 10, 29)),
        )
    return trace


def test_interval_uses_held_target_not_future_target_and_does_not_mutate():
    trace = measured(bounded=True)
    trace["joint_velocity_rad_s"][43:, 0, 4] = 20
    trace["foundation_navigation_joint_target_rad"][42, 0, 4] = 0.3
    trace["taskspace_joint_increment_rad"][42, 0, 4] = 0.02
    trace["pre_motor_joint_target_rad"][42, 0, 4] = 0.32
    trace["motor_delta_rad"][42, 0, 4] = -0.01
    trace["joint_target_rad"][42, 0, 4] = 0.31
    # The next tick's large target cannot explain an earlier velocity change.
    trace["foundation_navigation_joint_target_rad"][43, 0, 4] = 3
    trace["pre_motor_joint_target_rad"][43, 0, 4] = 3
    trace["joint_target_rad"][43, 0, 4] = 3
    trace["force_n"][42, 0, 1] = 4.0
    trace["torque_nm"][42, 0, :, 4] = 50
    trace["actual_actuator_force_nm"][42, 0, :, 4] = 49
    trace["pd_requested_torque_nm"][42, 0, :, 4] = 60
    before = copy.deepcopy(trace)
    result = observe_postcontact_action_transition(trace)
    row = result["contact_to_plus60"]
    assert row["peak_velocity_change_interval_start_frame"] == 42
    assert row["peak_velocity_change_interval_end_frame"] == 43
    assert row["held_target_frame"] == 42
    assert row["largest_absolute_velocity_change_joint_name"] == "left_ankle_pitch_joint"
    assert row["maximum_joint_velocity_step_l2_change_rad_s"] == 20
    components = row["target_step_into_held_interval"]
    assert components["final_target"]["vector_rad"][4] == pytest.approx(0.31)
    assert components["applied_motor_residual"]["vector_rad"][4] == pytest.approx(-0.01)
    assert components["foundation_navigation_target"]["vector_rad"][4] == pytest.approx(0.3)
    assert row["recorded_ball_leg_force_n"] == [0, 4, 0, 0, 0, 0]
    assert row["held_interval_torque_at_hard_limit_fraction"] == pytest.approx(1 / 29)
    assert row["held_interval_requested_pd_above_hard_limit_fraction"] == pytest.approx(1 / 29)
    assert row["held_interval_command_minus_actuator_force_rmse_nm"] == pytest.approx(
        1 / np.sqrt(29)
    )
    for key in trace:
        np.testing.assert_array_equal(trace[key], before[key])
    assert result["causal_failure_reason_proven"] is False
    assert result["ground_contact_or_support_phase_observed"] is False


def test_legacy_trace_does_not_invent_foundation_or_pd_requests():
    result = observe_postcontact_action_transition(measured())
    row = result["contact_to_plus60"]
    assert result["bounded_foundation_taskspace_capture_available"] is False
    assert row["held_interval_maximum_absolute_requested_pd_torque_nm"] is None
    assert "foundation_navigation_target" not in row["target_step_into_held_interval"]


def test_frame_zero_has_no_fabricated_previous_command():
    row = observe_postcontact_action_transition(measured(first=0))["contact_to_plus60"]
    assert row["held_target_frame"] == 0  # zero differences tie: earliest interval
    assert row["target_step_into_held_interval"] is None


@pytest.mark.parametrize("first,complete,last", [(239, True, 299), (240, False, 299)])
def test_fixed_window_censoring(first, complete, last):
    row = observe_postcontact_action_transition(measured(first))["contact_to_plus60"]
    assert row["complete_requested_window"] is complete
    assert row["last_frame"] == last


@pytest.mark.parametrize("first", [None, 299])
def test_no_observed_postcontact_interval_is_missing_not_zero(first):
    result = observe_postcontact_action_transition(measured(first))
    assert result["contact_to_plus60"] is None
    assert result["contact_to_episode_end"] is None


@pytest.mark.parametrize("corruption", ["nan", "extreme", "shape", "dtype"])
def test_invalid_observations_rejected(corruption):
    trace = measured()
    key = "joint_velocity_rad_s"
    if corruption == "nan":
        trace[key][42, 0, 0] = np.nan
    elif corruption == "extreme":
        trace[key][42, 0, 0] = 1e308
    elif corruption == "shape":
        trace[key] = trace[key][:-1]
    else:
        trace[key] = trace[key].astype(np.float32)
    with pytest.raises(ValueError, match="action measurements"):
        observe_postcontact_action_transition(trace)


def test_target_identity_cannot_be_forged():
    trace = measured()
    trace["joint_target_rad"][42, 0, 2] = 0.01
    with pytest.raises(ValueError, match="final targets"):
        observe_postcontact_action_transition(trace)


def test_partial_optional_capture_rejected():
    trace = measured(bounded=True)
    del trace["pd_requested_torque_nm"]
    with pytest.raises(ValueError, match="complete optional"):
        observe_postcontact_action_transition(trace)


def test_optional_target_decomposition_must_match():
    trace = measured(bounded=True)
    trace["taskspace_joint_increment_rad"][42, 0, 2] = 0.01
    with pytest.raises(ValueError, match="navigation plus taskspace"):
        observe_postcontact_action_transition(trace)


def test_torque_command_above_limit_rejected_not_clipped():
    trace = measured()
    trace["torque_nm"][42, 0, 0, 4] = 50.01
    with pytest.raises(ValueError, match="hard limits"):
        observe_postcontact_action_transition(trace)


def test_known_unsafe_outcome_is_not_relabelled_by_smooth_motion():
    trace = measured()
    trace["force_n"][42, 0, 4] = 2
    result = observe_postcontact_action_transition(trace)
    assert result["outcome"]["safe"] is False
    assert result["outcome"]["task_pass"] is False
    assert result["task_or_reward_changed"] is False
