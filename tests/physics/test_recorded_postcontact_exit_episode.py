"""Actual toy-body integration of exit fields; forecasts/NN NOT reexecuted."""

import numpy as np
import pytest

from tests.physics.test_recorded_early_episode import recorded  # noqa: F401


def exit_trace(trace):
    result = {k.replace("early_impact_feedback_", "postcontact_exit_"): v for k, v in trace.items()}
    result["postcontact_exit_forecast_steps"] = np.zeros((300, 1, 1), dtype=np.int64)
    frames = np.flatnonzero(np.any(result["postcontact_exit_requested_offset_nm"], axis=(1, 2)))
    if len(frames):
        result["postcontact_exit_forecast_steps"][frames[0]] = 80
    return result


@pytest.mark.parametrize("recorded", ["postcontact"], indirect=True)
def test_real_toy_postcontact_body_reintegrates_but_recorded_search_is_not_reexecuted(recorded):  # noqa: F811
    replay, trace = recorded
    r = replay.verify_postcontact_exit(exit_trace(trace))
    assert r["actual_physical_steps"] == 3000 and max(r["maximum_absolute_errors"].values()) == 0
    assert r["schema"] == "soccer.rsi.recorded_postcontact_exit_h300_physics.v1"
    assert r["recorded_forecast_steps_not_reexecuted"] == 80
    assert not r["independent_exit_search_qualified"] and not r["teacher_qualified"]
    assert (
        r["actual_postcontact_action_fields_consumed"]
        and not r["early_action_arrays_fabricated_or_aliased"]
    )
    with pytest.raises(ValueError, match="once-only"):
        replay.verify_postcontact_exit(exit_trace(trace))


@pytest.mark.parametrize("recorded", [True], indirect=True)
def test_no_contact_no_search_zero_body_still_preserved(recorded):  # noqa: F811
    replay, trace = recorded
    r = replay.verify_postcontact_exit(exit_trace(trace))
    assert r["actual_physical_steps"] == 3000
    assert r["recorded_forecast_steps_not_reexecuted"] == 0


@pytest.mark.parametrize(
    "bad", ["negative", "fraction", "count", "repeat", "old_profile", "missing", "shift", "dtype"]
)
@pytest.mark.parametrize("recorded", ["postcontact"], indirect=True)
def test_exit_forecast_profile_or_action_mismatch_rejects_before_steps(recorded, bad):  # noqa: F811
    replay, trace = recorded
    t = exit_trace(trace)
    steps = t["postcontact_exit_forecast_steps"]
    f = int(np.flatnonzero(steps)[0])
    if bad == "negative":
        steps[f] = -40
    elif bad == "fraction":
        steps[f] = 81
    elif bad == "count":
        steps[f] = 600
    elif bad == "repeat":
        steps[f + 1] = 80
    elif bad == "old_profile":
        t["postcontact_release_requested_offset_nm"] = np.zeros((300, 1, 29))
    elif bad == "missing":
        steps[:] = 0
    elif bad == "shift":
        steps[f] = 0
        steps[f + 1] = 80
    else:
        t["postcontact_exit_forecast_steps"] = steps.astype(np.float64)
    with pytest.raises(ValueError):
        replay.verify_postcontact_exit(t)
    assert replay.attempted_steps == 0


@pytest.mark.parametrize("recorded", ["postcontact_late"], indirect=True)
def test_late_fixed_action_cannot_be_mislabeled_as_complete_exit_forecast(recorded):  # noqa: F811
    replay, trace = recorded
    with pytest.raises(ValueError, match="no-replan"):
        replay.verify_postcontact_exit(exit_trace(trace))
    assert replay.attempted_steps == 0
