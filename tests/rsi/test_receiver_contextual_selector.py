import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_contextual_selector import selector_class, train_context_selector


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
