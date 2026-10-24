"""OFFLINE labels: retain actual successes, correct precision/failure states.

Episode outcomes choose supervision ONLY. They are never policy inputs. Teacher
advice is hypothetical on this student's state, not a physically executed win.
"""

import numpy as np


def receiving_dagger_label(
    actual_command: np.ndarray,
    teacher_advice: np.ndarray,
    *,
    observed_safe_functional: bool,
    observed_strict: bool,
    plane_before_contact: bool,
) -> tuple[np.ndarray, str]:
    if (
        any(
            x.shape != (3,)
            or x.dtype != np.float32
            or not np.isfinite(x).all()
            or ((x < np.array([-0.5, -0.5, -1])) | (x > np.array([1.0, 0.5, 1.0]))).any()
            for x in (actual_command, teacher_advice)
        )
        or any(
            type(b) is not bool
            for b in (observed_safe_functional, observed_strict, plane_before_contact)
        )
        or (observed_strict and not observed_safe_functional)
    ):
        raise ValueError(
            "bounded actual/advice commands and consistent OFFLINE outcome labels required"
        )
    if observed_strict or (observed_safe_functional and not plane_before_contact):
        return actual_command.copy(), "ACTUAL_SUCCESS_SELF_RETENTION"
    if plane_before_contact:
        return teacher_advice.copy(), "CAUSAL_PLANE_ADVICE_NOT_EXECUTED"
    return teacher_advice.copy(), "CAUSAL_FAILURE_ADVICE_NOT_EXECUTED"
