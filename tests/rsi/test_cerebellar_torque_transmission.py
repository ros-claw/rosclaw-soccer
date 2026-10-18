"""Synthetic numeric projection checks, not physics qualification."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.cerebellar_torque_transmission import diagnose_transmission
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS


def trace(offset=0.1, parent=0.0):
    q = np.zeros((300, 1, 10, 43), dtype=np.float64)
    q[..., 3] = 1
    q[..., 7:36] = -parent
    requested = np.full((300, 1, 10, 29), offset, dtype=np.float64)
    limits = np.asarray(G1_HARD_TORQUE_LIMITS)
    control = np.clip(parent + requested, -limits, limits)
    return dict(
        cerebellar_canonical_qpos_before_substep=q,
        cerebellar_canonical_qvel_before_substep=np.zeros((300, 1, 10, 41)),
        joint_target_rad=np.zeros((300, 1, 29)),
        requested_cerebellar_torque_offset_nm=requested,
        applied_cerebellar_torque_offset_nm=control - np.clip(parent, -limits, limits),
        torque_nm=control,
        actual_actuator_force_nm=control.copy(),
    )


def diagnose(value):
    return diagnose_transmission(value, kp=np.ones(29), kd=np.ones(29))


def test_transmitted_without_projection():
    result = diagnose(trace())
    assert result["applied_to_requested_l2_ratio"] == pytest.approx(1)
    assert result["neural_projection_changed_joint_samples"] == 0
    assert result["new_physical_steps"] == 0


def test_parent_saturation_can_remove_entire_neural_request():
    result = diagnose(trace(parent=1000))
    assert result["applied_to_requested_l2_ratio"] == 0
    assert result["neural_projection_changed_joint_samples"] == 87000
    assert result["parent_hard_clipped_joint_samples"] == 87000


def test_zero_request_has_no_fake_retention_ratio():
    assert diagnose(trace(0))["applied_to_requested_l2_ratio"] is None


def test_finite_but_overflow_risk_gains_rejected():
    with pytest.raises(ValueError):
        diagnose_transmission(trace(), kp=np.full(29, 1e308), kd=np.ones(29))


@pytest.mark.parametrize(
    "field", ["torque_nm", "actual_actuator_force_nm", "applied_cerebellar_torque_offset_nm"]
)
def test_tampered_transmission_rejected(field):
    value = trace()
    value[field][0, 0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="transmission"):
        diagnose(value)


def test_alternating_request_frame_average_is_not_whole_rms():
    value = trace()
    for name in (
        "requested_cerebellar_torque_offset_nm",
        "applied_cerebellar_torque_offset_nm",
        "torque_nm",
        "actual_actuator_force_nm",
    ):
        value[name][:, :, ::2] *= -1
    result = diagnose(value)
    assert result["applied"]["rms_joint_nm"] == pytest.approx(0.1)
    assert result["applied"]["control_frame_mean_rms_joint_nm"] == 0


@pytest.mark.parametrize("mutation", ["shape", "nan", "dtype", "cap"])
def test_malformed_capture_rejected(mutation):
    value = trace()
    key = "requested_cerebellar_torque_offset_nm"
    if mutation == "shape":
        value[key] = value[key][:-1]
    elif mutation == "dtype":
        value[key] = value[key].astype(np.float32)
    else:
        value[key][0, 0, 0, 0] = np.nan if mutation == "nan" else 10
    with pytest.raises(ValueError):
        diagnose(value)
