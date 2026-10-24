"""Declaration tests only; no new physical teacher success is implied."""

import copy

import pytest

from rosclaw_soccer.rsi.broad_receiving_role import (
    MODES,
    STATES,
    config,
    sampling_seed,
    select_courses,
)


def courses():
    return [
        dict(
            state_id=s,
            task=t,
            context=(1, 4, 20, 21)[s // 10],
            assisted=False,
            target=[s, 0, 0.11],
            side=-0.55 if s % 2 == 0 else 0.55,
        )
        for s in range(40)
        for t in (0, 1)
    ]


def test_fixed_sixteen_real_course_selection_not_best_outcome_selection():
    original = courses()
    selected = select_courses(original[::-1])
    assert [c["state_id"] for c in selected] == list(STATES)
    assert len({c["state_id"] for c in selected}) == 16
    assert {c["context"] for c in selected} == {1, 4, 20, 21}
    for context in (1, 4, 20, 21):
        assert sorted(c["side"] for c in selected if c["context"] == context) == [
            -0.55,
            -0.55,
            0.55,
            0.55,
        ]
    assert all(c["task"] == 0 for c in selected)
    selected[0]["target"][0] = -100
    assert original[0]["target"][0] == 0
    cfg = config()
    assert cfg["maximum_worlds_per_side"] == 2 * len(selected)
    assert cfg["formal_controls"] == 300
    assert cfg["recovery_controls"] == 100
    assert cfg["optimizer_updates"] == 0
    assert MODES == ("pursuit-v1", "projected-capture-v3")
    assert len({sampling_seed(s) for s in STATES}) == 16


@pytest.mark.parametrize("fault", ["missing", "duplicate", "assist", "context", "task", "bool"])
def test_incomplete_mutated_or_teacher_corrected_original_bank_refused(fault):
    rows = copy.deepcopy(courses())
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif fault == "assist":
        rows[10]["assisted"] = True
    elif fault == "context":
        rows[10]["context"] = 20
    elif fault == "task":
        rows[10]["task"] = 3
    else:
        rows[0]["state_id"] = False
    with pytest.raises(ValueError):
        select_courses(rows)


@pytest.mark.parametrize("state", [26, 40, -1, True, 0.0])
def test_non_declared_noise_identity_refused(state):
    with pytest.raises(ValueError):
        sampling_seed(state)


def test_global_expert_control_changes_only_A_reference_not_courses_noise_or_budget():
    local, control = config(), config(global_reference=True)
    changed = {k for k in local.keys() | control.keys() if local.get(k) != control.get(k)}
    assert changed == {"protocol", "A_fixed_generation", "A_fixed_actor_steps", "A_reference"}
    assert control["A_reference"] == "original_global_expert_2048_not_local_theta12"
    assert control["A_fixed_generation"] is None
    assert control["maximum_worlds_per_side"] == 32
    with pytest.raises(ValueError):
        config(global_reference="yes")
