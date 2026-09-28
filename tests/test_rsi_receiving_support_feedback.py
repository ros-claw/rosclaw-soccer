"""Boundary checks for the SIM_ONLY coupled receiving diagnostic."""

import math

import numpy as np
import pytest
from rsi_cpu_receiving_support_feedback import GAINS
from rsi_mjx_clean_touch_control_es import COURSES, FRESH8
from rsi_receiving_contact_dynamics_audit import _run


def test_support_probe_never_consumes_reserved_courses_as_training() -> None:
    assert len(COURSES) == 8
    assert len(FRESH8) == 8
    assert not set(COURSES) & set(FRESH8)
    assert GAINS == (-0.20, -0.10, 0.10, 0.20)


@pytest.mark.parametrize("gain", [-0.21, 0.21, math.nan, math.inf])
def test_unbounded_or_nonfinite_support_feedback_is_rejected(gain: float) -> None:
    with pytest.raises(ValueError, match="bounded SIM_ONLY coupled support"):
        _run(
            None,  # type: ignore[arg-type]
            {},
            np.zeros(80),
            privileged_teacher_lateral_sign=1.0,
            support_posture_gain=gain,
        )


def test_support_feedback_requires_explicit_sim_teacher() -> None:
    with pytest.raises(ValueError, match="bounded SIM_ONLY coupled support"):
        _run(None, {}, np.zeros(80), support_posture_gain=0.20)  # type: ignore[arg-type]
