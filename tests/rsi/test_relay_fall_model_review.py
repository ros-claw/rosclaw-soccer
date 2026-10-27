import numpy as np

from scripts.rsi_review_relay_fall_models import confusion


def test_confusion_preserves_both_error_types():
    prediction = np.array([True, False, True, False])
    truth = np.array([True, True, False, False])
    assert confusion(prediction, truth) == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}
