from collections import Counter
from dataclasses import replace

import pytest

from rosclaw_soccer.training.motion_portfolio_holdout import motion_portfolio_holdout_courses
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses


def test_all192_preflight_and_six_strata_without_old_bank_reuse():
    from rosclaw_soccer.training.expanded_entry_courses import expanded_entry_courses

    courses = motion_portfolio_holdout_courses()
    assert len(courses) == 192 and len({c.course for c in courses}) == 192
    assert set(Counter(c.stratum for c in courses).values()) == {32}
    assert not {c.course for c in courses} & {c.course for c in expanded_entry_courses()}
    assert preflight_receiving_courses(courses) == preflight_receiving_courses(
        motion_portfolio_holdout_courses()
    )


def test_bad_holdout_rejects_whole_declaration_not_clips_or_drops():
    courses = motion_portfolio_holdout_courses()
    bad = (replace(courses[0], course=(0.299, -0.1, -0.15)), *courses[1:])
    with pytest.raises(ValueError):
        preflight_receiving_courses(bad)
    with pytest.raises(ValueError):
        preflight_receiving_courses(courses[:-1])
