import numpy as np
import pytest

from rosclaw_soccer.rsi.walk_kick_retrieval import nearest_teacher, prospective_courses
from rosclaw_soccer.rsi.walk_kick_teacher_plan import search_course_bank
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses


def test_excludes_ineligible_even_if_exact_match():
    courses = [[2.2, 0.0, 0.4], [2.21, 0.0, 0.4], [2.8, 0.2, -0.4]]
    assert nearest_teacher(courses[0], courses, np.array([False, True, True])) == 1


def test_rejects_empty_or_nonfinite_inputs():
    with pytest.raises(ValueError):
        nearest_teacher([0, 0, 0], [[0, 0, 0]], np.array([False]))
    with pytest.raises(ValueError):
        nearest_teacher([0, np.nan, 0], [[0, 0, 0]], np.array([True]))


def test_prospective_bank_is_balanced_reproducible_and_disjoint():
    bank = prospective_courses()
    assert bank == prospective_courses()
    assert len(bank) == 8
    preflight_receiving_courses(bank)
    assert not {x.course for x in bank} & {
        x.course for x in search_course_bank(expanded_public=True)
    }
    assert [(x.course[0] >= 2.6, x.course[2] > 0) for x in bank] == [
        (False, False),
        (False, False),
        (False, True),
        (False, True),
        (True, False),
        (True, False),
        (True, True),
        (True, True),
    ]
