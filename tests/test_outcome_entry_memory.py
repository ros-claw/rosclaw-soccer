import math

import numpy as np
import pytest

from rosclaw_soccer.training.outcome_entry_memory import fit_outcome_entry_memory


def memory():
    return fit_outcome_entry_memory([[0.0], [0.1], [1.0], [1.1]], [True, True, False, False])


def test_success_failure_and_ambiguity():
    fitted = memory()
    assert fitted.assess((0.0,)).suggested
    assert not fitted.assess((1.0,)).suggested
    assert not fitted.assess((0.55,)).suggested
    assert fitted.assess((9.0,)).reason == "outside_success_support"


def test_conflicting_identical_states_abstain():
    fitted = fit_outcome_entry_memory([[0.0], [0.0], [0.0], [0.0]], [True, True, False, False])
    assert fitted.assess((0.0,)).reason == "near_failure_or_ambiguous"


def test_copies_and_shared_metric():
    values = np.array([[0.0], [0.1], [1.0], [1.1]])
    labels = np.array([True, True, False, False])
    fitted = fit_outcome_entry_memory(values, labels)
    values[:] = 5
    labels[:] = False
    assert fitted.assess((0.0,)).suggested
    assert fitted.success.scales == fitted.failure.scales
    assert fitted.success.threshold == fitted.failure.threshold


@pytest.mark.parametrize("labels", [[1, 1, 0, 0], [True], [True] * 4, [True, False, False, False]])
def test_reject_invalid_labels(labels):
    with pytest.raises(ValueError):
        fit_outcome_entry_memory([[0.0], [0.1], [1.0], [1.1]], labels)


@pytest.mark.parametrize("ratio", [True, 0, 1.0, -0.1, math.nan, math.inf])
def test_invalid_ratio(ratio):
    with pytest.raises(ValueError):
        fit_outcome_entry_memory(
            [[0.0], [0.1], [1.0], [1.1]], [True, True, False, False], separation_ratio=ratio
        )


@pytest.mark.parametrize("query", [(math.nan,), (math.inf,), (11.0,), (), [0.0]])
def test_invalid_query(query):
    with pytest.raises(ValueError):
        memory().assess(query)


def test_feature_translation_preserves_distance():
    fitted = fit_outcome_entry_memory([[2.0], [2.1], [3.0], [3.1]], [True, True, False, False])
    assert fitted.assess((2.05,)).success_distance == pytest.approx(
        memory().assess((0.05,)).success_distance
    )
