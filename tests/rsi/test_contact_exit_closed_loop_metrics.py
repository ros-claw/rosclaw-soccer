import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_exit_closed_loop_metrics import closed_loop_metrics, longest_run


def trace():
    pose = np.zeros((300, 1, 7))
    pose[:, 0, 6] = 1
    result = dict(
        force_n=np.zeros((300, 1, 6)),
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.7),
        ball_position_after_step_m=np.zeros((300, 1, 3)),
        root_pose_xyzw_m=pose,
        ball_linear_velocity_before_step_m_s=np.zeros((300, 1, 3)),
        torque_nm=np.zeros((300, 1, 10, 29)),
    )
    return result


def test_contact_peak_bins_do_not_claim_continuous_duration():
    t = trace()
    t["force_n"][74:77, 0, 0] = 20
    t["force_n"][78, 0, 0] = 20
    t["ball_position_after_step_m"][134, 0, 0] = 2
    t["ball_linear_velocity_before_step_m_s"][:75, 0, 0] = -1
    t["ball_linear_velocity_before_step_m_s"][75:80, 0, 0] = 1
    t["ball_linear_velocity_before_step_m_s"][80:, 0, 0] = -1
    t["root_pose_xyzw_m"][75:, 0, 0] = -0.02
    metrics = closed_loop_metrics(t)
    assert metrics["outcome"]["task_pass"] is True
    assert metrics["foot_contact_peak_bins_50hz"] == [4, 0]
    assert metrics["longest_foot_contact_peak_bin_run_50hz"] == [3, 0]
    assert metrics["peak_bins_are_not_continuous_contact_duration"] is True
    assert metrics["first_reverse_after_positive_ball_velocity_sample_50hz"] == 80
    assert metrics["postcontact_root_backward_sampled_travel_m"] == pytest.approx(0.02)
    assert metrics["successor_ready"] is None


def test_no_contact_and_censored_contact_are_not_fake_success():
    t = trace()
    metrics = closed_loop_metrics(t)
    assert metrics["foot_contact_peak_bins_50hz"] is None
    assert metrics["postcontact_actual_torque_rms_nm"] is None
    t["force_n"][270, 0, 0] = 20
    metrics = closed_loop_metrics(t)
    assert metrics["window_right_censored"] is True
    assert metrics["outcome"]["task_pass"] is False


def test_requested_and_applied_offsets_are_distinct_evidence():
    t = trace()
    t.update(
        contact_exit_mpc_requested_offset_nm=np.zeros((300, 1, 29)),
        contact_exit_mpc_applied_offset_nm=np.zeros((300, 1, 10, 29)),
        contact_exit_mpc_forecast_steps=np.zeros((300, 1, 1), dtype=np.int64),
    )
    t["contact_exit_mpc_requested_offset_nm"][80, 0, 0] = 2
    t["contact_exit_mpc_applied_offset_nm"][80, 0, :, 0] = 0.5
    t["contact_exit_mpc_forecast_steps"][80, 0, 0] = 50
    metrics = closed_loop_metrics(t)["teacher_intervention"]
    assert metrics["maximum_requested_offset_nm"] == 2
    assert metrics["maximum_actual_applied_offset_nm"] == 0.5
    assert metrics["nonzero_actual_offset_frames"] == [80]
    assert metrics["forecast_integrated_substeps"] == 50


@pytest.mark.parametrize("bad", ["nan", "quaternion", "dtype", "shape"])
def test_invalid_body_measurements_are_rejected(bad):
    t = trace()
    if bad == "nan":
        t["root_pose_xyzw_m"][0, 0, 0] = np.nan
    elif bad == "quaternion":
        t["root_pose_xyzw_m"][0, 0, 6] = 0.2
    elif bad == "dtype":
        t["torque_nm"] = t["torque_nm"].astype(np.float32)
    else:
        t["ball_linear_velocity_before_step_m_s"] = np.zeros((299, 1, 3))
    with pytest.raises(ValueError):
        closed_loop_metrics(t)


def test_longest_run_validates_bin_semantics():
    assert longest_run(np.zeros(0, dtype=np.bool_)) == 0
    with pytest.raises(ValueError):
        longest_run(np.ones(3))


def directional_trace():
    t = trace()
    t.update(
        directional_compliance_requested_delta_nm=np.zeros((300, 1, 10, 29)),
        directional_compliance_applied_delta_nm=np.zeros((300, 1, 10, 29)),
        directional_compliance_forecast_steps=np.zeros((300, 1, 1), dtype=np.int64),
    )
    return t


def test_gain_authority_is_not_retagged_as_old_capped_additive_offset():
    t = directional_trace()
    t["directional_compliance_requested_delta_nm"][80, 0, :, 0] = 15
    t["directional_compliance_applied_delta_nm"][80, 0, :, 0] = 3
    t["directional_compliance_forecast_steps"][80, 0, 0] = 10
    result = closed_loop_metrics(t)
    assert "teacher_intervention" not in result
    gain = result["directional_gain_intervention"]
    assert gain["maximum_requested_delta_nm"] == 15
    assert gain["maximum_actual_applied_delta_nm"] == 3
    assert gain["nonzero_actual_delta_frames"] == [80]
    assert gain["forecast_integrated_substeps"] == 10
    assert gain["this_is_not_the_old_capped_additive_residual"] is True


@pytest.mark.parametrize("bad", ["nan", "shape", "dtype", "negative_steps", "float_steps"])
def test_invalid_gain_authority_rejected(bad):
    t = directional_trace()
    if bad == "nan":
        t["directional_compliance_applied_delta_nm"][0, 0, 0, 0] = np.nan
    elif bad == "shape":
        t["directional_compliance_requested_delta_nm"] = np.zeros((300, 1, 29))
    elif bad == "dtype":
        t["directional_compliance_applied_delta_nm"] = t[
            "directional_compliance_applied_delta_nm"
        ].astype(np.float32)
    elif bad == "negative_steps":
        t["directional_compliance_forecast_steps"][0, 0, 0] = -1
    else:
        t["directional_compliance_forecast_steps"] = np.zeros((300, 1, 1))
    with pytest.raises(ValueError):
        closed_loop_metrics(t)
