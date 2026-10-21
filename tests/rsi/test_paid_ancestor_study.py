import copy

import numpy as np
import pytest
from test_paid_fresh_protocol import source_cases

from rosclaw_soccer.rsi.paid_ancestor_study import SCHEDULES, ancestor_train_cases, case_condition
from rosclaw_soccer.rsi.paid_fresh_protocol import fresh_lifecycle_cases
from rosclaw_soccer.rsi.paid_lifecycle_bundle import require_same_prefix
from rosclaw_soccer.sim.contracts import hash_json


def sources():
    cases = source_cases()
    names = [
        "001_right",
        "002_left",
        "003_left",
        "004_right",
        "005_right",
        "006_right",
        "008_left",
        "009_right",
    ]
    for i, c in enumerate(cases):
        c["source"] = "soccer-standard-" + names[i // 2] + ".npz"
    return cases


def test_new_train_is_fixed20_distinct_from_consumed_fresh_and_never_copies_outcomes():
    original = sources()
    before = copy.deepcopy(original)
    rows = ancestor_train_cases(original)
    assert original == before and rows == ancestor_train_cases(original)
    assert len(rows) == 20 and [r["case"] for r in rows] == list(range(20))
    assert len({r["source"] for r in rows}) == 8
    assert sum(r["cohort"] == "source009_coverage" for r in rows) == 4
    assert all(
        r["bank"] == "train" and r["not_fresh"] and "obsolete_outcome" not in r for r in rows
    )
    old = {hash_json(case_condition(c)) for c in fresh_lifecycle_cases(original)}
    assert not old & {hash_json(case_condition(c)) for c in rows}
    assert all(np.linalg.norm(c["velocity_world_initial"]) <= 0.6 for c in rows)
    assert SCHEDULES["grown"][:2] == SCHEDULES["ancestor_third"][:2]


@pytest.mark.parametrize("fault", ["missing", "mass", "source"])
def test_bad_train_source_is_rejected_without_replacing_conditions(fault):
    original = sources()
    if fault == "missing":
        original.pop()
    elif fault == "mass":
        original[0]["ball_mass_kg"] = 1
    else:
        original[-1]["source"] = "another-right.npz"
    with pytest.raises(ValueError):
        ancestor_train_cases(original)


def prefix():
    keys = [
        "world-initial",
        "kick0",
        "kick-terminal0",
        "stand0",
        "between0-1-drop",
        "between0-1-settle",
        "between0-1-boundary",
        "kick1",
        "kick-terminal1",
        "stand1",
        "between1-2-drop",
        "between1-2-settle",
        "between1-2-boundary",
        "kick2",
    ]
    return {key: dict(measured=np.zeros(3)) for key in keys}


def test_counterfactual_can_change_third_but_not_any_previous_body_or_action():
    a = prefix()
    b = copy.deepcopy(a)
    b["kick2"]["measured"] += 1
    require_same_prefix(a, b, before_kick=2)
    b["between1-2-boundary"]["measured"] += 0.01
    with pytest.raises(ValueError):
        require_same_prefix(a, b, before_kick=2)


def test_missing_prefix_cannot_be_a_same_state_proof():
    with pytest.raises(ValueError):
        require_same_prefix({}, {}, before_kick=2)
    a = prefix()
    b = copy.deepcopy(a)
    del a["between1-2-boundary"]
    del b["between1-2-boundary"]
    with pytest.raises(ValueError):
        require_same_prefix(a, b, before_kick=2)
