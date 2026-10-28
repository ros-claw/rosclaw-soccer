import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_contextual_selector import (
    group_holdout_validation,
    selector_class,
    train_context_selector,
)


def test_selector_trains_deterministically_on_own_observations():
    x = np.zeros((6, 98), np.float32)
    x[:, 0] = np.arange(6, dtype=np.float32)
    labels = np.array([0, 0, 1, 1, 2, 2], np.int64)
    w, report = train_context_selector(x, labels, seed=4, steps=120)
    again, _ = train_context_selector(x, labels, seed=4, steps=120)
    assert report["training_accuracy"] == 1.0
    assert report["neural_weight_updates"] == 120
    assert report["promotion_authorized"] is False
    assert all(np.array_equal(w[k], again[k]) for k in w)
    assert [selector_class(row, w) for row in x] == labels.tolist()


def test_nonfinite_or_malformed_input_fails_closed():
    x = np.zeros((2, 98), np.float32)
    labels = np.array([0, 1], np.int64)
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        train_context_selector(x, labels, seed=0, steps=3)
    x[0, 0] = 0
    w, _ = train_context_selector(x, labels, seed=0, steps=3)
    w["w1"][0, 0] = np.nan
    with pytest.raises(ValueError):
        selector_class(x[0], w)


def test_group_validation_refits_without_held_out_group_and_rejects_single_group():
    x = np.zeros((6, 98), np.float32)
    x[:, 0] = np.arange(6, dtype=np.float32)
    labels = np.array([0, 0, 1, 1, 2, 2], np.int64)
    groups = np.array([0, 0, 1, 1, 2, 2], np.int64)
    result = group_holdout_validation(x, labels, groups, steps=10)
    assert len(result["folds"]) == 3
    assert all(f["train_rows"] == 4 and f["test_rows"] == 2 for f in result["folds"])
    assert result["held_out_contexts_excluded_from_fold_training"] is True
    assert result["physical_rollout"] is False
    with pytest.raises(ValueError, match="context groups"):
        group_holdout_validation(x, labels, np.zeros(6, np.int64), steps=10)
