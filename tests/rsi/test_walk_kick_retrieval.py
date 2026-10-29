import numpy as np
import pytest

from rosclaw_soccer.rsi.walk_kick_retrieval import (
    course_in_training_domain,
    nearest_teacher,
    outcome_model_teacher,
    prospective_courses,
)
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


def test_extended_bank_has_ten_per_class_and_no_existing_cases():
    bank = prospective_courses(extended=True)
    assert len(bank) == 40
    assert bank == prospective_courses(extended=True)
    preflight_receiving_courses(bank)
    assert not {x.course for x in bank} & {
        x.course for x in (*prospective_courses(), *search_course_bank(expanded_public=True))
    }
    assert [(x.course[0] >= 2.6, x.course[2] > 0) for x in bank] == (
        [(False, False)] * 10 + [(False, True)] * 10 + [(True, False)] * 10 + [(True, True)] * 10
    )


def test_outcome_model_ranks_fixed_donors_without_claiming_safety():
    scores = np.array([[[0.8, 0.1], [0.6, 0.7]]] * 3)
    assert outcome_model_teacher(scores, np.array([3, 9])) == 9
    scores[0, 1, 1] = np.nan
    with pytest.raises(ValueError):
        outcome_model_teacher(scores, np.array([3, 9]))


def test_out_of_domain_team_pass_is_not_silently_extrapolated():
    assert course_in_training_domain([2.3, 0.1, -0.5])
    assert not course_in_training_domain([2.3, 0.1, -3.1])
    assert not course_in_training_domain([2.5, 0.1, -0.5])
    assert not course_in_training_domain([2.3, 0.17, -0.5])
    with pytest.raises(ValueError, match="outside declared"):
        nearest_teacher([2.3, 0.1, -3.1], [[2.3, 0.1, -0.5]], np.array([True]))
