import numpy as np
import pytest

from rosclaw_soccer.rsi.pd_intent_audit import audit_pd_intent


def inputs():
    return dict(
        decoded=np.zeros((2, 29)),
        targets=np.zeros((2, 29)),
        positions=np.zeros((20, 29)),
        torques=np.zeros((20, 29)),
        limits=np.tile([-0.25, 0.25], (29, 1)),
        effort=np.full(29, 10.0),
    )


def test_setpoint_excess_is_not_joint_position_excess_and_inputs_unchanged():
    values = inputs()
    values["decoded"][0, 11] = 0.2833
    values["targets"][0, 11] = 0.2842
    values["torques"][0, 11] = 10.0
    before = {k: v.copy() for k, v in values.items()}
    row = audit_pd_intent(**values)
    assert row["decoded_setpoints"]["entries_outside_range"] == 1
    assert row["submitted_setpoints"]["entries_outside_range"] == 1
    assert row["measured_joint_positions"]["entries_outside_range"] == 0
    assert row["changed_entries_already_outside_in_original_nn"] == 1
    assert row["changed_entries_outside_range"] == 1
    assert row["maximum_applied_effort_fraction"] == 1.0
    assert not row["effort_cap_exceeded"]
    assert not row["safety_or_skill_qualification"]
    for key in values:
        np.testing.assert_array_equal(values[key], before[key])


def test_actual_stop_penetration_and_torque_excess_are_not_hidden():
    values = inputs()
    values["positions"][3, 7] = -0.251
    values["torques"][8, 7] = 10.01
    row = audit_pd_intent(**values)
    assert row["measured_joint_positions"]["counts_by_hg"][7] == 1
    assert row["measured_joint_positions"]["maximum_excess_rad"] == pytest.approx(0.001)
    assert row["effort_cap_exceeded"]


@pytest.mark.parametrize("key", ["decoded", "targets", "positions", "torques", "limits", "effort"])
def test_nonfinite_or_unaligned_data_cannot_be_silently_dropped(key):
    values = inputs()
    values[key].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        audit_pd_intent(**values)
    values = inputs()
    values[key] = values[key][:-1]
    with pytest.raises(ValueError):
        audit_pd_intent(**values)


def test_float_joint_arrays_and_strict_caps_required():
    values = inputs()
    values["effort"][0] = 0
    with pytest.raises(ValueError):
        audit_pd_intent(**values)
    values = inputs()
    values["targets"] = values["targets"].astype(int)
    with pytest.raises(ValueError):
        audit_pd_intent(**values)
