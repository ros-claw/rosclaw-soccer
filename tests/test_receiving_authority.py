import numpy as np
import pytest

from rosclaw_soccer.training.receiving_authority import receiving_authority_diagnostics


def fixture():
    trace = {"time_sec": np.arange(20) * 0.002, "control_frame": np.arange(20) // 10}
    for name in (
        "foundation_target_rad",
        "requested_residual_rad",
        "added_residual_rad",
        "pd_target_rad",
        "joint_position_rad",
        "joint_velocity_radps",
        "kp",
        "kd",
        "raw_torque_nm",
        "projected_torque_nm",
        "executed_torque_nm",
    ):
        trace[name] = np.zeros((20, 29))
    trace["requested_residual_rad"][:, 0] = 0.05
    trace["added_residual_rad"][:10, 0] = 0.05
    trace["pd_target_rad"][:] = trace["added_residual_rad"]
    trace["kp"][:] = 10
    trace["raw_torque_nm"][:] = trace["kp"] * trace["pd_target_rad"]
    trace["projected_torque_nm"][:] = trace["raw_torque_nm"]
    trace["executed_torque_nm"][:] = trace["raw_torque_nm"]
    return {"receiving_authority_" + k: v for k, v in trace.items()}


def test_explicit_suppression_is_not_conflated_with_projection_or_success():
    trace = fixture()
    report = receiving_authority_diagnostics(trace)
    assert report["requested_joint_samples"] == 20
    assert report["suppressed_requested_joint_samples"] == 10
    assert report["joint_guard_changed_samples"] == report["torque_clip_changed_samples"] == 0
    assert report["additive_non_pd_torque_rms_nm"] == [0.0] * 29
    assert not report["contact_success_inferred"]


def test_projection_clipping_and_extra_task_torque_are_separate():
    trace = fixture()
    trace["receiving_authority_raw_torque_nm"][0, 0] += 1
    trace["receiving_authority_executed_torque_nm"][0, 0] = 0
    report = receiving_authority_diagnostics(trace)
    assert report["joint_guard_changed_samples"] == 1
    assert report["torque_clip_changed_samples"] == 1
    assert report["additive_non_pd_torque_rms_nm"][0] > 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("time_sec", np.zeros(20)),
        ("control_frame", np.zeros(20, dtype=int)),
        ("pd_target_rad", np.ones((20, 29))),
        ("kp", np.full((20, 29), -1.0)),
        ("kd", np.full((20, 29), np.nan)),
        ("requested_residual_rad", np.ones((20, 29))),
        ("joint_position_rad", np.zeros((20, 28))),
    ],
)
def test_invalid_measurements_fail_closed(key, value):
    trace = fixture()
    trace["receiving_authority_" + key] = value
    with pytest.raises(ValueError):
        receiving_authority_diagnostics(trace)


def test_missing_path_is_not_synthesized_from_motion():
    with pytest.raises(ValueError):
        receiving_authority_diagnostics({})
