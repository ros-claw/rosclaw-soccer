import numpy as np
import pytest

from rosclaw_soccer.rsi.walk_kick_retrieval import nearest_teacher


def test_excludes_ineligible_even_if_exact_match():
    courses = [[2.2, 0.0, 0.4], [2.21, 0.0, 0.4], [2.8, 0.2, -0.4]]
    assert nearest_teacher(courses[0], courses, np.array([False, True, True])) == 1


def test_rejects_empty_or_nonfinite_inputs():
    with pytest.raises(ValueError):
        nearest_teacher([0, 0, 0], [[0, 0, 0]], np.array([False]))
    with pytest.raises(ValueError):
        nearest_teacher([0, np.nan, 0], [[0, 0, 0]], np.array([True]))
