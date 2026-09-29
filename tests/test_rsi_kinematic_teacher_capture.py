"""Specialist labels must match the old and new guarded action envelopes."""

import pytest
from rsi_r1_kinematic_teacher_capture_v205 import _new_fraction, _old_fraction


def test_specialist_precontact_phase_is_explicit() -> None:
    assert _old_fraction(15) == _new_fraction(15) == 0
    assert _old_fraction(19) == _new_fraction(19) == 1
    assert _old_fraction(30) == _new_fraction(30) == 1
    assert _old_fraction(35) == pytest.approx(0.5)
    assert _new_fraction(35) == 1
    assert _old_fraction(40) == 0
    assert _new_fraction(40) == 1
    assert _new_fraction(64) == pytest.approx(1 / 15)
