"""Actual toy post-touch physics, not real G1 skill/teacher qualification."""

import numpy as np
import pytest

from tests.physics.test_recorded_early_episode import recorded  # noqa: F401


def post_trace(trace):
    result = {
        k.replace("early_impact_feedback_", "postcontact_release_"): v for k, v in trace.items()
    }
    result["postcontact_release_forecast_steps"] = np.zeros((300, 1, 1), dtype=np.int64)
    return result


@pytest.mark.parametrize("recorded", ["postcontact", "postcontact_late"], indirect=True)
def test_nonzero_actual_measured_contact_and_late_h300_lease_reintegrate(recorded):  # noqa: F811
    replay, trace = recorded
    post = post_trace(trace)
    frames = np.flatnonzero(np.any(post["postcontact_release_requested_offset_nm"], axis=(1, 2)))
    assert len(frames) and np.any(post["force_n"][frames[0] - 1, 0, :2] > 1)
    if replay.course[0] > 2:
        assert frames[0] >= 295 and len(frames) < 3
    r = replay.verify_postcontact_release(post)
    assert r["schema"] == "soccer.rsi.recorded_postcontact_h300_physics.v1"
    assert r["actual_physical_steps"] == 3000
    assert max(r["maximum_absolute_errors"].values()) == 0
    assert r["actual_postcontact_action_fields_consumed"] is True
    assert r["early_action_arrays_fabricated_or_aliased"] is False
    assert not r["teacher_qualified"]
    with pytest.raises(ValueError, match="once-only"):
        replay.verify_postcontact_release(post)


def test_noncontact_offset_cannot_be_mislabeled_post_touch(recorded):  # noqa: F811
    replay, trace = recorded
    with pytest.raises(ValueError, match="completed foot force"):
        replay.verify_postcontact_release(post_trace(trace))
    assert replay.attempted_steps == 0


@pytest.mark.parametrize("bad", ["hybrid", "forecast", "dtype"])
@pytest.mark.parametrize("recorded", [True], indirect=True)
def test_distinct_scope_and_zero_forecast_admission_before_steps(recorded, bad):  # noqa: F811
    replay, trace = recorded
    post = post_trace(trace)
    if bad == "hybrid":
        post["early_impact_feedback_requested_offset_nm"] = np.zeros((300, 1, 29))
    elif bad == "forecast":
        post["postcontact_release_forecast_steps"][1] = 40
    else:
        post["postcontact_release_forecast_steps"] = post[
            "postcontact_release_forecast_steps"
        ].astype(np.float64)
    with pytest.raises(ValueError):
        replay.verify_postcontact_release(post)
    assert replay.attempted_steps == 0
