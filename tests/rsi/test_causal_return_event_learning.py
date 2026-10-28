import numpy as np

from rosclaw_soccer.rsi.causal_return_event_learning import (
    causal_features,
    classify_first,
    event_labels,
)


def test_features_do_not_use_future_and_label_after_event():
    q, v = np.zeros((9, 10, 12)), np.zeros((9, 10, 12))
    v[4, -1, 0] = 1
    f = np.zeros((9, 10, 2, 3))
    f[4, 2, 1, 0] = 2
    args = dict(start=4, ball_q=0, ball_v=0, passer_q=3, passer_v=3, returner_q=6, returner_v=6)
    x = causal_features(q, v, **args)
    v[6:, :, :] = 99
    assert np.array_equal(x[:3], causal_features(q, v, **args)[:3])
    labels, onset, censored = event_labels(f, start=4)
    assert onset == 1 and not censored and labels.tolist() == [0, 1, 1, 1, 1]


def test_event_censoring_and_early_detection():
    f = np.zeros((6, 10, 2, 3))
    f[-1, 0, 1, 0] = 2
    _, onset, censored = event_labels(f, start=4)
    assert onset is None and censored
    assert classify_first(np.array([True, False]), 1, False) == "EARLY_FALSE_ALARM"
    assert classify_first(np.array([False, True]), 1, False) == "TIMELY"
