from collections import Counter

from rosclaw_soccer.rsi.teacher_coverage_cohort import teacher_coverage_indices
from rosclaw_soccer.training.categorical_motion_holdout import categorical_motion_holdout_courses
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses


def test_diagnostic_subset_fixed_before_teacher_outcomes_balanced_and_valid():
    courses = categorical_motion_holdout_courses()
    assert preflight_receiving_courses(courses)
    selected = teacher_coverage_indices()
    assert len(selected) == len(set(selected)) == 24
    assert selected == tuple(16 * g + k for g in range(6) for k in (0, 5, 10, 15))
    assert sorted(Counter(courses[i].stratum for i in selected).values()) == [4] * 6
    # Eight independent exec jobs cover all24, never eight replicates/case.
    assert sorted(i for p in range(8) for i in selected[p::8]) == list(selected)
