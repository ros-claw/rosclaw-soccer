from dataclasses import replace

import pytest

from rosclaw_soccer.training.close_contact_courses import close_contact_perturbations
from rosclaw_soccer.training.receiving_course_preflight import (
    FirstTouchCourse,
    preflight_receiving_courses,
)


def test_complete_close_bank_distinct_and_four_clusters():
    bank = close_contact_perturbations()
    assert len(bank) == len({x.course for x in bank}) == len({x.seed for x in bank}) == 16
    assert [sum(x.parent_index == p for x in bank) for p in range(4)] == [4] * 4
    assert preflight_receiving_courses(bank) == preflight_receiving_courses(
        close_contact_perturbations()
    )
    assert all(x.course[2] == 0 for x in bank)


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "order", "coordinate", "velocity", "bool", "seed"]
)
def test_any_declaration_change_rejects_whole_bank(kind):
    bank = list(close_contact_perturbations())
    if kind == "missing":
        bank.pop()
    elif kind == "duplicate":
        bank[1] = bank[0]
    elif kind == "order":
        bank[0], bank[1] = bank[1], bank[0]
    elif kind == "coordinate":
        bank[0] = replace(bank[0], course=(0.0, -0.1, 0.0))
    elif kind == "velocity":
        bank[0] = replace(bank[0], course=(bank[0].course[0], bank[0].course[1], 0.01))
    elif kind == "bool":
        bank[0] = replace(bank[0], parent_index=False)
    else:
        bank[0] = replace(bank[0], seed=0)
    with pytest.raises(ValueError):
        preflight_receiving_courses(tuple(bank))


def test_new_profile_never_widens_original_rsi_domain():
    close = close_contact_perturbations()[0]
    with pytest.raises(ValueError, match="declared domain"):
        preflight_receiving_courses((FirstTouchCourse("tracker", 1, close.course),))
    original = (FirstTouchCourse("tracker", 1, (2.3, 0.0, 0.3)),)
    assert preflight_receiving_courses(original) != preflight_receiving_courses(
        close_contact_perturbations()
    )


def test_mixed_profiles_rejected():
    bank = list(close_contact_perturbations())
    bank[0] = FirstTouchCourse("tracker", 1, (2.3, 0.0, 0.3))
    with pytest.raises(ValueError):
        preflight_receiving_courses(tuple(bank))
