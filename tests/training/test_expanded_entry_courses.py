from dataclasses import replace

import pytest

from rosclaw_soccer.training.expanded_entry_courses import (
    expanded_entry_courses,
    preflight_expanded_entry_bank,
)
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses


def test_exact40_distinct_positions_and_balanced_explicit_rolling():
    bank = expanded_entry_courses()
    assert len(bank) == len({c.course[:2] for c in bank}) == len({c.seed for c in bank}) == 40
    assert [sum(c.course[2] == speed for c in bank) for speed in (0, -0.15, 0.15)] == [20, 10, 10]
    assert [c.course_index for c in bank] == list(range(40))
    assert preflight_expanded_entry_bank(bank) == preflight_receiving_courses(bank)


@pytest.mark.parametrize("bad", ["missing", "duplicate", "order", "speed", "position", "type"])
def test_reject_entire_declaration_not_silently_repair(bad):
    bank = list(expanded_entry_courses())
    if bad == "missing":
        bank.pop()
    elif bad == "duplicate":
        bank[1] = bank[0]
    elif bad == "order":
        bank[0], bank[1] = bank[1], bank[0]
    elif bad == "speed":
        bank[0] = replace(bank[0], course=(0.30, -0.24, 0.151))
    elif bad == "position":
        bank[0] = replace(bank[0], course=(3.0, -0.24, 0.0))
    else:
        bank[0] = replace(bank[0], seed=True)
    with pytest.raises(ValueError):
        preflight_receiving_courses(tuple(bank))
