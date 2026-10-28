import numpy as np
import pytest

from rosclaw_soccer.rsi.causal_ball_delta_trigger import CausalBallDeltaTrigger


def test_trigger_only_uses_successive_samples_and_latches():
    detector = CausalBallDeltaTrigger(np.zeros(3), threshold_mps=0.3)
    assert detector.observe(np.array([0.2, 0.0, 0.0]))[0] is False
    assert detector.observe(np.array([0.6, 0.0, 0.0]))[0] is True
    assert detector.observe(np.array([0.6, 0.0, 0.0]))[0] is True


def test_invalid_sample_does_not_mutate_history():
    detector = CausalBallDeltaTrigger(np.zeros(3), threshold_mps=0.3)
    with pytest.raises(ValueError):
        detector.observe(np.array([float("nan"), 0.0, 0.0]))
    assert detector.observe(np.array([0.2, 0.0, 0.0]))[0] is False
