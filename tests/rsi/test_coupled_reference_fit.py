import numpy as np
import pytest
from test_contact_reference_ik import fixture

from rosclaw_soccer.rsi.coupled_reference_fit import fit_reference_contact


def test_fit_is_offline_and_preserves_entire_input_state():
    model, data, layout = fixture(0.12)
    before = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time
    result = fit_reference_contact(model, data, layout, foot="right")
    for actual, initial in zip((data.qpos, data.qvel, data.ctrl), before[:3], strict=True):
        np.testing.assert_array_equal(actual, initial)
    assert data.time == before[3]
    assert result["new_physical_steps"] == 0
    assert not result["physically_qualified"] and not result["hardware_authorized"]
    if result["status"] != "SOURCE_OUTSIDE_NATIVE_LIMITS":
        assert np.max(np.abs(result["delta_leg_rad"])) <= 0.35 + 1e-12


def test_invalid_foot_and_out_of_range_seed_fail_closed():
    model, data, layout = fixture(0.12)
    with pytest.raises(ValueError):
        fit_reference_contact(model, data, layout, foot="auto")
    for invalid_budget in (0, 2001, True):
        with pytest.raises(ValueError):
            fit_reference_contact(model, data, layout, foot="left", max_evaluations=invalid_budget)
    data.qpos[layout.qi[0]] = 3.0
    result = fit_reference_contact(model, data, layout, foot="left")
    assert result["status"] == "SOURCE_OUTSIDE_NATIVE_LIMITS"
    assert not result["physically_qualified"]
