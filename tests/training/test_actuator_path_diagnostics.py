import numpy as np
import pytest

from rosclaw_soccer.training.actuator_path_diagnostics import diagnose_actuator_path


def _inputs():
    return {
        "position_rad": np.zeros((3, 2)),
        "velocity_radps": np.zeros((3, 2)),
        "target_rad": np.array([[2.0, -2.0], [0.0, 0.0], [1.0, 0.0]]),
        "kp": np.ones((3, 2)),
        "kd": np.ones((3, 2)),
        "extra_torque_nm": np.zeros((3, 2)),
        "applied_torque_nm": np.array([[1.0, 0.5], [0.0, 0.0], [-0.25, 0.0]]),
        "torque_cap_nm": np.array([1.0, 1.0]),
        "joint_names": ("a", "b"),
    }


def test_records_caps_projection_and_sign_reversal_without_authorization():
    report = diagnose_actuator_path(**_inputs())
    assert report["substeps"] == 3
    assert report["joints"][0]["at_cap_substeps"] == 1
    assert report["joints"][0]["post_cap_difference_substeps"] == 1
    assert report["joints"][0]["sign_reversal_substeps"] == 1
    assert report["joints"][1]["sign_reversal_substeps"] == 1
    assert report["input_evidence_authenticated"] is False
    assert report["cause_inferred"] is False
    assert report["training_authorized"] is False
    assert report["promotion_authorized"] is False


@pytest.mark.parametrize("field", ["position_rad", "kp", "extra_torque_nm", "torque_cap_nm"])
def test_rejects_nonfinite(field):
    values = _inputs()
    values[field].flat[0] = np.nan
    with pytest.raises(ValueError):
        diagnose_actuator_path(**values)


def test_rejects_unaligned_duplicate_or_over_cap_evidence():
    values = _inputs()
    values["velocity_radps"] = np.zeros((2, 2))
    with pytest.raises(ValueError):
        diagnose_actuator_path(**values)
    values = _inputs()
    values["torque_cap_nm"][0] = 1e-10
    with pytest.raises(ValueError):
        diagnose_actuator_path(**values)
    values = _inputs()
    values["joint_names"] = ("a", "a")
    with pytest.raises(ValueError):
        diagnose_actuator_path(**values)
    values = _inputs()
    values["applied_torque_nm"][0, 0] = 1.1
    with pytest.raises(ValueError):
        diagnose_actuator_path(**values)


def test_extra_torque_is_part_of_the_request():
    values = _inputs()
    values["target_rad"][:] = 0
    values["applied_torque_nm"][:] = 0
    values["extra_torque_nm"][0, 0] = 0.5
    values["applied_torque_nm"][0, 0] = 0.5
    report = diagnose_actuator_path(**values)
    assert report["joints"][0]["post_cap_difference_substeps"] == 0
    assert report["joints"][0]["max_abs_requested_nm"] == 0.5
