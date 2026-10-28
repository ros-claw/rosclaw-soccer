import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_risk_cv import grouped_risk_selection


def test_censored_cases_are_not_scored_and_missing_classes_abstain():
    features = np.zeros((4, 98), np.float32)
    groups = np.array([0, 0, 1, 1], np.int64)
    base = np.array([True, False, False, False])
    cand = np.zeros(4, bool)
    observed = np.array([True, True, True, False])
    result = grouped_risk_selection(features, groups, base, cand, observed, steps=2)
    assert result["labeled"] == 3
    assert result["censored"] == 1
    assert result["predictions"][3] == -1
    assert result["folds"][0]["updates"] == 0
    assert result["predictions"][0] == 0
    assert not result["promotion_authorized"]


def test_rejects_nonfinite_and_unseparated_groups():
    features = np.zeros((4, 98), np.float32)
    groups = np.zeros(4, np.int64)
    flags = np.zeros(4, bool)
    with pytest.raises(ValueError):
        grouped_risk_selection(features, groups, flags, flags, flags)


def test_unobserved_predictions_never_train_or_enter_scored_outcomes():
    features = np.arange(6 * 98, dtype=np.float32).reshape(6, 98) / 1000
    groups = np.array([0, 0, 1, 1, 2, 2], np.int64)
    base = np.array([True, False, True, False, False, False])
    candidate = np.array([False, False, False, False, True, True])
    observed = np.array([True, True, True, True, False, False])
    first = grouped_risk_selection(
        features, groups, base, candidate, observed, steps=5, predict_unobserved=True
    )
    changed = candidate.copy()
    changed[~observed] = False
    second = grouped_risk_selection(
        features, groups, base, changed, observed, steps=5, predict_unobserved=True
    )
    assert first["predictions"] == second["predictions"]
    assert first["probabilities"] == second["probabilities"]
    assert first["selected_counterfactual_failures"] == second["selected_counterfactual_failures"]
    assert first["unobserved_predictions"] == 2
