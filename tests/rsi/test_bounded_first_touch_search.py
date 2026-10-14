from dataclasses import asdict

import pytest

from rosclaw_soccer.rsi.bounded_first_touch_control import BOUNDS
from rosclaw_soccer.rsi.bounded_first_touch_search import (
    initial_population,
    refine_population,
    select_multistate_elites,
)


def evaluations():
    return [
        dict(
            candidate=i,
            state=s,
            outcome=dict(safe=True, task_pass=i == 1, training_reward=float(i)),
        )
        for i in range(4)
        for s in (0, 4, 8, 12)
    ]


def test_reproducible_bounded_proposals():
    a = initial_population(seed=42, count=4)
    assert a == initial_population(seed=42, count=4)
    b = refine_population(a, (1, 2), seed=43, count=8)
    assert b[0] == a[1]
    for p in b:
        assert all(lo <= asdict(p)[n] <= hi for n, (lo, hi) in BOUNDS.items())


def test_one_unsafe_state_vetoes_best_quality_candidate():
    population = initial_population(seed=42, count=4)
    records = evaluations()
    records[4]["outcome"].update(safe=False, task_pass=False)
    assert select_multistate_elites(
        population, state_ids=(0, 4, 8, 12), records=records, elite_count=2
    ) == (3, 2)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "nonfinite", "bool_index"])
def test_incomplete_or_invalid_evaluations_fail_closed(kind):
    records = evaluations()
    if kind == "missing":
        records.pop()
    elif kind == "duplicate":
        records.append(records[0])
    elif kind == "nonfinite":
        records[0]["outcome"]["training_reward"] = float("nan")
    else:
        records[0]["candidate"] = False
    with pytest.raises(ValueError):
        select_multistate_elites(
            initial_population(seed=42, count=4),
            state_ids=(0, 4, 8, 12),
            records=records,
            elite_count=2,
        )


def test_all_unsafe_stops_instead_of_recycling_failures_as_elites():
    population = initial_population(seed=42, count=4)
    records = evaluations()
    for row in records:
        row["outcome"].update(safe=False, task_pass=False)
    elites = select_multistate_elites(
        population, state_ids=(0, 4, 8, 12), records=records, elite_count=2
    )
    assert elites == ()
    with pytest.raises(ValueError, match="stop search"):
        refine_population(population, elites, seed=43, count=4)
