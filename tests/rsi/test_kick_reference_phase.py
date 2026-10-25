import numpy as np
import pytest

from rosclaw_soccer.rsi.kick_reference_phase import nearest_kick_reference_phase


def arrays():
    return np.zeros(29), np.zeros(29), np.ones((300, 29)), np.zeros((300, 29))


def test_actual_pose_and_velocity_choose_phase_without_changing_reference():
    q, v, rp, rv = arrays()
    rp[204] = 0
    before = rp.copy()
    r = nearest_kick_reference_phase(q, v, rp, rv)
    assert r["reference_frame"] == 204 and r["selected_cost_rad2"] == 0
    np.testing.assert_array_equal(before, rp)
    rv[204] = 100
    assert nearest_kick_reference_phase(q, v, rp, rv)["reference_frame"] == 0


def test_no_search_after250_and_ties_choose_first():
    q, v, rp, rv = arrays()
    rp[251] = 0
    r = nearest_kick_reference_phase(q, v, rp, rv)
    assert r["reference_frame"] == 0 and len(r["phase_costs_rad2"]) == 251


def test_explicit_contact_window_does_not_change_default_phase_selection():
    q, v, rp, rv = arrays()
    rp[53] = 0
    rp[204] = 0.1
    assert nearest_kick_reference_phase(q, v, rp, rv)["reference_frame"] == 53
    selected = nearest_kick_reference_phase(q, v, rp, rv, minimum_frame=180)
    assert selected["reference_frame"] == 204 and selected["search_first"] == 180
    with pytest.raises(ValueError):
        nearest_kick_reference_phase(q, v, rp, rv, minimum_frame=True)


@pytest.mark.parametrize("defect", ["short", "nonfinite", "wrong_order_size", "overflow"])
def test_bad_reference_rejected(defect):
    q, v, rp, rv = arrays()
    if defect == "short":
        rp, rv = rp[:250], rv[:250]
    elif defect == "nonfinite":
        q[0] = np.nan
    elif defect == "wrong_order_size":
        rp = rp[:, :28]
    else:
        q[:] = 1e308
    with pytest.raises(ValueError):
        nearest_kick_reference_phase(q, v, rp, rv)
