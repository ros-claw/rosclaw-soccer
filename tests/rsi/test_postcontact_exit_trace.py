"""Synthetic raw exit causality checks, not independent physical evidence."""

import numpy as np
import pytest

from scripts.rsi_postcontact_exit_trace import check_exit_trace
from scripts.rsi_postcontact_release_transfer_inputs import require_scope
from tests.rsi.test_postcontact_release_readback import row_trace
from tests.rsi.test_postcontact_release_transfer import scope


def exit_trace(monkeypatch):
    _, trace, _ = row_trace(monkeypatch)
    trace = {k.replace("postcontact_release_", "postcontact_exit_"): v for k, v in trace.items()}
    trace["canonical_qpos"][:, :, 2] = 0.793
    trace["postcontact_exit_forecast_steps"][41] = 120
    review = dict(
        forecast_integrated_steps=120,
        forecast_attempted_steps=120,
        postcontact_exit=dict(
            trigger_frame=41,
            measured_contact_side=0,
            cancelled=False,
            maximum_forecast_steps=560,
            selected_name="lateral_plus",
            forecast_integrated_steps=120,
        ),
    )
    return trace, review


def test_single_actual_search_and_three_frame_clipped_control(monkeypatch):
    t, r = exit_trace(monkeypatch)
    c = check_exit_trace(t, r)
    assert c["first_physically_applied_correction_frame"] == 41
    assert c["forecast_integrated_steps"] == 120 and c["nonzero_requested_frames"] == 3
    assert not c["independent_search_qualified"]


@pytest.mark.parametrize(
    "bad",
    [
        "trigger",
        "side",
        "cancel",
        "counter",
        "negative_steps",
        "repeat_bank",
        "no_bank",
        "offset",
        "old_profile",
        "pd",
        "force",
        "name",
        "dtype",
        "caps",
    ],
)
def test_causal_or_physical_capture_mismatch_rejected(monkeypatch, bad):
    t, r = exit_trace(monkeypatch)
    if bad == "trigger":
        r["postcontact_exit"]["trigger_frame"] = 42
    elif bad == "side":
        r["postcontact_exit"]["measured_contact_side"] = 1
    elif bad == "cancel":
        r["postcontact_exit"]["cancelled"] = True
    elif bad == "counter":
        r["forecast_integrated_steps"] = 80
    elif bad == "negative_steps":
        t["postcontact_exit_forecast_steps"][0] = -40
    elif bad == "repeat_bank":
        t["postcontact_exit_forecast_steps"][80] = 120
    elif bad == "no_bank":
        t["postcontact_exit_forecast_steps"][:] = 0
    elif bad == "offset":
        t["postcontact_exit_requested_offset_nm"][90, 0, 0] = 0.1
    elif bad == "old_profile":
        t["postcontact_release_requested_offset_nm"] = np.zeros((300, 1, 29))
    elif bad == "pd":
        t["actual_actuator_force_nm"][70, 0, 0, 0] = 0.1
    elif bad == "force":
        t["force_n"][0, 0, 0] = -1
    elif bad == "name":
        r["postcontact_exit"]["selected_name"] = "undeclared"
    elif bad == "caps":
        t["postcontact_exit_requested_offset_nm"][41, 0, 0] = 100
    else:
        t["postcontact_exit_forecast_steps"] = t["postcontact_exit_forecast_steps"].astype(
            np.float64
        )
    with pytest.raises(ValueError):
        check_exit_trace(t, r)


def test_adaptive_scope_and_its_extra_bounds_cannot_be_relaxed():
    c = dict(
        scope(),
        profile="postcontact-exit-proxy-transfer18-v1",
        action_name="CURRENT_BODY_EXIT_PROXY",
        forecast_step_budget=10080,
        postcontact_exit_enabled=True,
        maximum_forecasts_per_episode=14,
    )
    require_scope(c)
    for k in c:
        broken = dict(c)
        broken.pop(k)
        with pytest.raises(ValueError):
            require_scope(broken)
    for k, v in (
        ("maximum_forecasts_per_episode", 15),
        ("postcontact_exit_enabled", 1),
        ("forecast_step_budget", 10081),
    ):
        with pytest.raises(ValueError):
            require_scope(dict(c, **{k: v}))
