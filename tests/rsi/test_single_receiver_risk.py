import numpy as np
import pytest

from rosclaw_soccer.rsi.single_receiver_risk import fit_single_receiver_risk


def test_censored_candidate_failure_is_negative_not_paired_success():
    x = np.arange(6 * 104, dtype=np.float32).reshape(6, 104) / 1000
    base = np.array([True, False, False, False, False, False])
    cand = np.array([False, False, True, False, False, False])
    complete = np.array([True, True, False, False, False, False])
    _, result = fit_single_receiver_risk(x, base, cand, complete, steps=3)
    assert result["admitted"] == 3
    assert result["added_candidate_failures"] == 1
    assert result["positives"] == 1
    assert result["optimizer_updates"] == 3
    assert not result["physical_rollout"]


def test_single_model_requires_positive_and_negative_examples():
    x = np.zeros((3, 104), np.float32)
    flags = np.zeros(3, bool)
    with pytest.raises(ValueError):
        fit_single_receiver_risk(x, flags, flags, np.ones(3, bool))
