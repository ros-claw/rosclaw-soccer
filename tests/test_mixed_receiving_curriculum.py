from collections import Counter
from dataclasses import replace

import pytest

from rosclaw_soccer.training.mixed_receiving_curriculum import mixed_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ROSTER, ReceivingCourse


def reference():
    return tuple(
        ReceivingCourse(agent, i, 0.75, lateral)
        for i, (lateral, agent) in enumerate((side, a) for side in (-0.04, 0.04) for a in ROSTER)
    )


def build(courses=None, **kwargs):
    return mixed_receiving_courses(
        reference() if courses is None else courses,
        **{"first_seed": 1000, "hard_first_seed": 2000, **kwargs},
    )


def test_every_old_geometry_is_preserved_but_has_fresh_seed():
    original = reference()
    result = build()
    assert len(result) == 80 and len({c.seed for c in result}) == 80
    assert result[:16] == tuple(replace(c, seed=1000 + i) for i, c in enumerate(original))
    assert Counter(c.agent_id for c in result) == {a: 10 for a in ROSTER}
    assert original == reference() and result == build()


def test_equal_counts_without_equal_geometry_is_rejected():
    courses = list(reference())
    courses[0] = replace(courses[0], lateral_m=0.12)
    with pytest.raises(ValueError, match="geometry coverage"):
        build(tuple(courses))


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "seed_bool", "speed_nan", "side_inf", "role"]
)
def test_invalid_reference_cannot_silently_drop_rehearsal(change):
    courses = list(reference())
    if change == "missing":
        courses.pop()
    elif change == "duplicate":
        courses[1] = courses[0]
    elif change == "seed_bool":
        courses[0] = replace(courses[0], seed=True)
    elif change == "speed_nan":
        courses[0] = replace(courses[0], speed_mps=float("nan"))
    elif change == "side_inf":
        courses[0] = replace(courses[0], lateral_m=float("inf"))
    else:
        courses[0] = replace(courses[0], agent_id="red.unknown")
    with pytest.raises(ValueError):
        build(tuple(courses))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"first_seed": True},
        {"hard_first_seed": -1},
        {"first_seed": 2**32 - 15},
        {"hard_first_seed": 2**32 - 63},
        {"first_seed": 0},
        {"hard_first_seed": 1000},
        {"hard_speeds_mps": ()},
        {"hard_speeds_mps": (0.4,)},
        {"hard_speeds_mps": (1.0, 1.0)},
        {"hard_speeds_mps": (True,)},
        {"hard_lateral_m": (float("nan"),)},
        {"hard_lateral_m": (0.31,)},
    ],
)
def test_seed_and_grid_bounds(kwargs):
    with pytest.raises(ValueError):
        build(**kwargs)


def test_last_uint32_seed_is_allowed_without_wraparound():
    courses = build(hard_first_seed=2**32 - 64)
    assert courses[-1].seed == 2**32 - 1
