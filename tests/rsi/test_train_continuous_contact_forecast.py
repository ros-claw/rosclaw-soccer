import numpy as np
import pytest

from scripts.rsi_train_continuous_contact_forecast import split_weights


def test_imbalance_weights_use_training_labels_only_and_are_finite():
    labels = np.array([0] * 20 + [1] * 5 + [2] * 2, dtype=np.int8)
    weights = split_weights(labels)
    assert np.isclose(weights.mean(), 1)
    assert weights[2] > weights[1] > weights[0]
    with pytest.raises(ValueError):
        split_weights(np.zeros(5, dtype=np.int8))
