"""Training diversity and sealed location separation are deterministic contracts."""

from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_course_catalog import (
    sample_training_courses,
    static_development_courses,
)


def test_static_extension_preserves_historical_eight() -> None:
    first = static_development_courses(8)
    extended = static_development_courses(16)
    assert extended[:8] == first
    assert len(set(extended)) == 16
    assert {course[0] for course in extended} == {2.3, 2.4, 2.6, 2.7}
    with pytest.raises(ValueError):
        static_development_courses(12)


def test_seeded_training_courses_are_diverse_and_sealed_disjoint() -> None:
    a = sample_training_courses(20260928)
    b = sample_training_courses(20260928)
    c = sample_training_courses(20260929)
    assert a == b and a != c
    assert len(set(a)) == 16
    assert all(2.2 <= x <= 2.8 and (x <= 2.4 or x >= 2.6) for x, _, _ in a)
    assert all(-0.16 <= y <= 0.16 and 0.25 <= abs(v) <= 0.7 for _, y, v in a)
    assert len({round(y, 2) for _, y, _ in a}) >= 12
    assert np.isfinite(a).all()
    with pytest.raises(ValueError):
        sample_training_courses(-1)
