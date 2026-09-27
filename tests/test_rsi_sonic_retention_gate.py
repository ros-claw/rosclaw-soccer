"""Conservative regression gating remains explicit and bounded."""

import pytest
from rsi_sonic_retention_gate import RESERVED, _choose_threshold


def test_fresh_gate_courses_are_unique() -> None:
    assert len(RESERVED) == 6
    assert len(set(RESERVED)) == 6
    assert all(0.06 <= y <= 0.18 for _, y in RESERVED)


def test_last_harmful_course_forces_margin_before_candidate() -> None:
    dev = [
        {"ball_y_m": 0.08, "candidate_minus_parent_error_m": 0.26},
        {"ball_y_m": 0.12, "candidate_minus_parent_error_m": -0.02},
        {"ball_y_m": 0.16, "candidate_minus_parent_error_m": -0.65},
    ]
    assert _choose_threshold(dev) == pytest.approx(0.11)
    assert (
        _choose_threshold([{**row, "candidate_minus_parent_error_m": -0.02} for row in dev]) == 0.06
    )
    with pytest.raises(ValueError, match="no safe candidate"):
        _choose_threshold([{**row, "candidate_minus_parent_error_m": 0.26} for row in dev])
