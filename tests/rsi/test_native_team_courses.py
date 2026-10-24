from dataclasses import replace

import pytest

from rosclaw_soccer.training.native_team_courses import native_team_course_bank
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses


def test_entire_native_team_course_bank_is_preflighted_without_legacy_coercion():
    bank = native_team_course_bank()
    assert len(bank) == 5
    assert preflight_receiving_courses(bank).startswith("sha256:")
    assert preflight_receiving_courses(bank) == preflight_receiving_courses(
        native_team_course_bank()
    )


@pytest.mark.parametrize("mutation", ["nonfinite", "too_fast", "too_far", "duplicate"])
def test_one_invalid_course_rejects_the_entire_declaration(mutation):
    bank = native_team_course_bank()
    bad = {
        "nonfinite": replace(bank[-1], ball_xy=(float("nan"), 0.0)),
        "too_fast": replace(bank[-1], ball_velocity_xy=(1.0, 0.0)),
        "too_far": replace(bank[-1], ball_xy=(2.1, 0.0)),
        "duplicate": bank[0],
    }[mutation]
    with pytest.raises(ValueError):
        preflight_receiving_courses((*bank[:-1], bad))
