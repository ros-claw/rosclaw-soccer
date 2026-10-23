import numpy as np
import pytest

from scripts.rsi_diagnose_motor_target_response import summarize


def arrays():
    reference = np.array([[0.0, 0.0], [1.8, 0.0], [1.8, 0.0], [1.8, 0.0]])
    target = reference + np.array([[0.2, 0.0], [-0.2, 0.0], [-0.2, 0.0], [-0.2, 0.0]])
    measured = np.array([[0.0, 0.0], [0.05, 0.0], [0.1, 0.0], [0.15, 0.0]])
    velocity = np.array([[0.0, 0.0], [2.5, 0.0], [2.5, 0.0], [2.5, 0.0]])
    return target, reference, measured, velocity, ["knee", "ankle"]


def test_large_target_step_is_not_misreported_as_measured_joint_motion():
    r = summarize(*arrays())
    assert r["peak_target_change"]["control"] == 1
    assert r["peak_target_change"]["joint"] == "knee"
    assert r["peak_target_change"]["absolute_change_rad"] == pytest.approx(1.4)
    assert r["measured_same_joint_change_at_peak_target_rad"] == pytest.approx(0.05)
    assert r["measured_same_joint_velocity_at_peak_target_rad_s"] == 2.5
    assert r["unavoidable_target_jump_floor_under_pointwise_0p2rad_bound"] == pytest.approx(1.4)
    assert r["physical_smoothness_impossibility_proven"] is False
    assert r["bound_applies_only_to_recorded_reference_sequence"] is True
    assert r["closed_loop_future_nominal_can_change_with_actions"] is True


@pytest.mark.parametrize("fault", ["shape", "nonfinite", "names", "clock"])
def test_inconsistent_arrays_do_not_produce_a_diagnostic(fault):
    a = list(arrays())
    kw = {}
    if fault == "shape":
        a[2] = a[2][:-1]
    elif fault == "nonfinite":
        a[0][1, 0] = np.nan
    elif fault == "names":
        a[4] = ["knee", "knee"]
    else:
        kw["dt"] = 0.01
    with pytest.raises(ValueError):
        summarize(*a, **kw)
