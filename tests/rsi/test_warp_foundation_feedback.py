"""No GPU or physics: reject invalid feedback clocks/states before observation."""

import numpy as np
import pytest

from scripts.rsi_warp_foundation_feedback import FoundationFeedback


@pytest.mark.parametrize("control", [-1, 1, 84])
def test_no_skipped_repeated_or_extended_control_clock(control):
    feedback = FoundationFeedback.__new__(FoundationFeedback)
    feedback.control = 0
    with pytest.raises(ValueError, match="clock"):
        feedback.next(np.zeros((1, 43)), np.zeros((1, 41)), control)


def test_nonfinite_state_is_rejected_before_native_forward_or_inference():
    feedback = FoundationFeedback.__new__(FoundationFeedback)
    feedback.control = 0
    feedback.indices = [0]
    q = np.zeros((1, 43))
    q[0, 2] = np.nan
    with pytest.raises(ValueError, match="finite GPU state"):
        feedback.next(q, np.zeros((1, 41)), 0)
