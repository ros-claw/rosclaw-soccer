import numpy as np
import pytest

from rosclaw_soccer.rsi.cerebellar_recovery_pose_diagnostics import diagnose_recovery_pose


def trace(first=10):
    force = np.zeros((300, 1, 6))
    if first is not None:
        force[first, 0, 0] = 2
    ball = np.zeros((300, 1, 3))
    ball[:, 0, 0] = np.arange(300) * 0.02
    q = np.zeros((300, 1, 10, 43))
    q[:, :, :, 3] = 1
    return dict(
        force_n=force,
        ball_position_after_step_m=ball,
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.75),
        cerebellar_canonical_qpos_before_substep=q,
        cerebellar_canonical_qvel_before_substep=np.zeros((300, 1, 10, 41)),
        applied_cerebellar_torque_offset_nm=np.zeros((300, 1, 10, 29)),
        cerebellar_before_integration_substep_index=np.arange(3000, dtype=np.int64).reshape(
            300, 1, 10, 1
        ),
    )


def test_yaw_motion_is_not_reported_as_tilt_or_jitter():
    value = trace()
    q = value["cerebellar_canonical_qpos_before_substep"].reshape(3000, 43)
    q[110:710, 3] = np.cos(np.linspace(0, 0.2, 600))
    q[110:710, 6] = np.sin(np.linspace(0, 0.2, 600))
    value["cerebellar_canonical_qvel_before_substep"][:, :, :, 5] = 2
    result = diagnose_recovery_pose(value)
    d = result["diagnostics"]
    assert d["actual_substep_samples"] == 600
    assert d["root_tilt_maximum_rad"] == 0
    assert d["root_world_heading_change_rad"] == pytest.approx(0.4)
    assert d["root_local_angular_energy_fractions"] == [0, 0, 1]
    assert result["angular_rms_is_not_jitter_classification"]


def test_still_body_has_no_undefined_zero_division_or_invented_energy_share():
    d = diagnose_recovery_pose(trace())["diagnostics"]
    assert d["root_local_angular_energy_fractions"] is None
    assert d["root_local_angular_velocity_component_rms_rad_s"] == [0, 0, 0]


@pytest.mark.parametrize("first", [None, 240, 260, 299])
def test_missing_full_window_has_no_fabricated_pose_diagnostics(first):
    assert diagnose_recovery_pose(trace(first))["diagnostics"] is None


def test_degenerate_world_heading_is_explicitly_unobserved():
    value = trace()
    value["cerebellar_canonical_qpos_before_substep"][:, :, :, 3] = np.sqrt(0.5)
    value["cerebellar_canonical_qpos_before_substep"][:, :, :, 5] = np.sqrt(0.5)
    d = diagnose_recovery_pose(value)["diagnostics"]
    assert d["root_tilt_maximum_rad"] == pytest.approx(np.pi / 2)
    assert not d["root_world_heading_projection_observed"]
    assert d["root_world_heading_change_rad"] is None


def test_invalid_quaternion_not_silently_normalized():
    value = trace()
    value["cerebellar_canonical_qpos_before_substep"][0, 0, 0, 3] = 2
    with pytest.raises(ValueError, match="quaternion"):
        diagnose_recovery_pose(value)
