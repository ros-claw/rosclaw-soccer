import numpy as np
import pytest

from rosclaw_soccer.rsi.pd_feedback_attribution import attribute_pd_feedback


def fixture():
    original = {
        k: np.zeros(shape)
        for k, shape in (
            ("joint_target_rad", (300, 1, 29)),
            ("joint_position_rad", (300, 1, 29)),
            ("joint_velocity_rad_s", (300, 1, 29)),
            ("pd_requested_torque_nm", (300, 1, 10, 29)),
            ("torque_nm", (300, 1, 10, 29)),
        )
    }
    changed = {k: v.copy() for k, v in original.items()}
    changed["contact_exit_mpc_requested_offset_nm"] = np.zeros((300, 1, 29))
    return original, changed


def test_zero_intervention_does_not_invent_ratio():
    result = attribute_pd_feedback(*fixture(), kp=np.full(29, 10.0), kd=np.ones(29))
    assert result["projections_on_requested_correction"] is None


def test_exact_position_velocity_compensation_is_not_energy_or_causal_claim():
    a, b = fixture()
    b["joint_position_rad"][77, 0, 0] = 0.1
    b["joint_velocity_rad_s"][77, 0, 0] = 0.5
    b["pd_requested_torque_nm"][77, 0, 0, 0] = -1.5
    b["contact_exit_mpc_requested_offset_nm"][77, 0, 0] = 2
    b["torque_nm"][77, 0, 0, 0] = 0.5
    r = attribute_pd_feedback(a, b, kp=np.full(29, 10.0), kd=np.ones(29))
    p = r["projections_on_requested_correction"]
    assert p["position_feedback"] == pytest.approx(-0.5)
    assert p["velocity_feedback"] == pytest.approx(-0.25)
    assert p["net_preclip"] == pytest.approx(0.25)
    assert p["actual_clipped_torque_difference"] == pytest.approx(0.25)
    assert r["causal_component_intervention_performed"] is False
    assert r["total_torque_or_energy_cancellation_claimed"] is False


@pytest.mark.parametrize("bad", ["nan", "gains", "equation", "offset"])
def test_bad_pd_or_shapes_rejected(bad):
    a, b = fixture()
    kp, kd = np.full(29, 10.0), np.ones(29)
    if bad == "nan":
        b["joint_target_rad"][0, 0, 0] = np.nan
    elif bad == "gains":
        kp[0] = 0
    elif bad == "equation":
        b["pd_requested_torque_nm"][0, 0, 0, 0] = 2
    else:
        b["contact_exit_mpc_requested_offset_nm"] = np.zeros((1, 1, 29))
    with pytest.raises(ValueError):
        attribute_pd_feedback(a, b, kp=kp, kd=kd)
