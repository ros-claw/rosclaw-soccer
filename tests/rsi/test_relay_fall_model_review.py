import numpy as np

from scripts.rsi_review_relay_fall_models import confusion, episode_detection


def test_confusion_preserves_both_error_types():
    prediction = np.array([True, False, True, False])
    truth = np.array([True, True, False, False])
    assert confusion(prediction, truth) == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}


def test_episode_detection_does_not_count_multiple_frames_as_falls():
    prediction = np.array([True, True, False, True])
    truth = np.array([True, True, False, False])
    episodes = np.array([10, 10, 11, 11])
    controls = np.array([2, 3, 0, 1])
    assert episode_detection(prediction, truth, episodes, controls) == {
        "positive_episodes": 1,
        "detected_positive_episodes": 1,
        "no_observed_b_failure_episodes": 1,
        "no_observed_b_failure_episodes_with_alarm": 1,
        "detected_warning_lead_controls": [2],
    }
