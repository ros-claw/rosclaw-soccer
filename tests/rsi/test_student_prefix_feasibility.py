import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.student_prefix_feasibility import (
    BOUNDARIES,
    CONTEXTS,
    SEEDS,
    prefix_feasibility_summary,
    verify_physical_prefix,
)


def source():
    return {
        k: np.zeros((100, 2), np.float32)
        for k in ("NN_obs", "NN_action", "qpos", "qvel", "forces", "torque")
    }


def test_later_teacher_action_is_allowed_but_actual_student_prefix_must_match():
    a, b = source(), source()
    b["NN_action"][40] = 1
    verify_physical_prefix(a, b, 40)
    with pytest.raises(ValueError):
        verify_physical_prefix(a, b, 80)
    a["qpos"] = a["qpos"][:39]
    with pytest.raises(ValueError):
        verify_physical_prefix(a, b, 40)


def records():
    return [
        dict(seed=s, context=c, prefix_controls=b, baseline_quality=c != 5, quality=c != 8)
        for s in SEEDS
        for c in CONTEXTS
        for b in BOUNDARIES
    ]


def test_complete_outcomes_preserve_teacher_regressions_and_actual_student_losses():
    rows = records()
    r = prefix_feasibility_summary(rows)
    assert r["worlds"] == 36 and r["source_contexts"] == 18
    assert r["rescues"] == 6 and r["regressions"] == 6
    assert r["all_observed_regressions_recoverable"] is True
    assert r["student_learning_proven"] is False
    assert r["calibrated_probability"] is False


@pytest.mark.parametrize("fault", ["missing", "duplicate", "baseline_change", "untyped"])
def test_no_dropped_failures_or_different_parent_for_same_pair(fault):
    r = copy.deepcopy(records())
    if fault == "missing":
        r.pop()
    elif fault == "duplicate":
        r[-1] = dict(r[0])
    elif fault == "baseline_change":
        r[0]["baseline_quality"] = not r[0]["baseline_quality"]
    else:
        r[0]["quality"] = 1
    with pytest.raises(ValueError):
        prefix_feasibility_summary(r)
