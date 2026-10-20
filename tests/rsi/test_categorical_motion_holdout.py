from collections import Counter
from dataclasses import replace

import pytest

from rosclaw_soccer.training.categorical_motion_holdout import categorical_motion_holdout_courses
from rosclaw_soccer.training.motion_portfolio_holdout import motion_portfolio_holdout_courses
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses


def test_entire96_balanced_population_disjoint_from_consumed192():
    courses = categorical_motion_holdout_courses()
    assert courses == categorical_motion_holdout_courses()
    assert len(courses) == 96
    assert len(set(c.course for c in courses)) == 96
    assert sorted(Counter(c.stratum for c in courses).values()) == [16] * 6
    assert not set(c.course for c in courses) & {
        c.course for c in motion_portfolio_holdout_courses()
    }
    assert preflight_receiving_courses(courses).startswith("sha256:")


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "out_of_domain", "nan", "bool", "altered"]
)
def test_any_invalid_course_rejects_entire_bank(kind):
    courses = list(categorical_motion_holdout_courses())
    if kind == "missing":
        courses.pop()
    elif kind == "duplicate":
        courses[1] = courses[0]
    elif kind == "out_of_domain":
        courses[0] = replace(courses[0], course=(0.2, -0.1, 0.15))
    elif kind == "nan":
        courses[0] = replace(courses[0], course=(float("nan"), -0.1, 0.15))
    elif kind == "bool":
        courses[0] = replace(courses[0], seed=True)
    else:
        courses[0] = replace(courses[0], course_index=1)
    with pytest.raises(ValueError):
        preflight_receiving_courses(tuple(courses))
