import numpy as np
import pytest

from rosclaw_soccer.training.guarded_episode_credit import (
    guarded_terminal_credit,
    joint_margin_penalty,
)


def test_bounded_monotonic_margin_credit_does_not_modify_measurements():
    margins = np.array([[0.10, 0.20], [0.04, 0.20], [0.0, 0.20], [-0.01, 0.20]])
    original = margins.copy()
    penalty = joint_margin_penalty(margins)
    np.testing.assert_allclose(penalty, [0, -0.025, -0.05, -0.05])
    np.testing.assert_array_equal(margins, original)
    assert penalty.dtype == np.float32


def test_undiscounted_task_score_cannot_compensate_failed_physics_or_contact():
    worst_margin = float(joint_margin_penalty(np.zeros((400, 29))).sum(dtype=np.float64))
    best_unsafe = guarded_terminal_credit(
        task_score=1e100, physically_safe=False, contact_qualified=True
    )
    worst_contact_failure = (
        guarded_terminal_credit(task_score=-1e100, physically_safe=True, contact_qualified=False)
        + worst_margin
    )
    best_contact_failure = guarded_terminal_credit(
        task_score=1e100, physically_safe=True, contact_qualified=False
    )
    worst_qualified = (
        guarded_terminal_credit(task_score=-1e100, physically_safe=True, contact_qualified=True)
        + worst_margin
    )
    assert best_unsafe < worst_contact_failure < best_contact_failure < worst_qualified


def test_discounted_unequal_horizons_are_not_a_safety_ordering_guarantee():
    # A concrete counterexample prevents treating reward strata as constrained RL.
    unsafe = guarded_terminal_credit(task_score=0.0, physically_safe=False, contact_qualified=True)
    qualified = guarded_terminal_credit(
        task_score=-20.0, physically_safe=True, contact_qualified=True
    )
    assert unsafe < qualified
    assert unsafe * 0.995**399 > qualified


@pytest.mark.parametrize("fault", ["nan", "integer", "empty", "long", "rank", "wide"])
def test_invalid_measurements_rejected(fault):
    values = np.zeros((2, 29))
    if fault == "nan":
        values[0, 0] = np.nan
    elif fault == "integer":
        values = values.astype(int)
    elif fault == "empty":
        values = values[:0]
    elif fault == "long":
        values = np.zeros((401, 29))
    elif fault == "rank":
        values = values[0]
    else:
        values = np.zeros((2, 257))
    with pytest.raises(ValueError):
        joint_margin_penalty(values)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(task_score=float("inf"), physically_safe=True, contact_qualified=True),
        dict(task_score=True, physically_safe=True, contact_qualified=True),
        dict(task_score=1.0, physically_safe=1, contact_qualified=True),
        dict(task_score=1.0, physically_safe=True, contact_qualified=0),
    ],
)
def test_invalid_outcome_labels_rejected(kwargs):
    with pytest.raises(ValueError):
        guarded_terminal_credit(**kwargs)
