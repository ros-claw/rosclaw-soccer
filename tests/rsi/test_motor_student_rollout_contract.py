import numpy as np
import pytest

from scripts.rsi_motor_student_rollout import exact_array, validate_student_recording


def test_exact_array_does_not_hide_dtype_or_signed_zero_change():
    a = np.array([0.0], dtype=np.float32)
    b = np.array([-0.0], dtype=np.float32)
    with pytest.raises(AssertionError):
        exact_array(a, b)
    with pytest.raises(AssertionError):
        exact_array(a, a.astype(np.float64))
    exact_array(a, a.copy())


def test_incomplete_trace_cannot_claim_neural_or_physical_verification():
    with pytest.raises(ValueError, match="complete NN/native"):
        validate_student_recording({}, nbody=46)
