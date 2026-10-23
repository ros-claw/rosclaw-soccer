import copy

import pytest

from rosclaw_soccer.rsi.single_mode_pass_curriculum import single_mode_pass_curriculum
from rosclaw_soccer.rsi.team_pass_consolidation import RETAINED


def records():
    out = [
        dict(index=i, role="retained_neural17", source_context=c, task_success=True, bc_weight=1)
        for i, c in enumerate(RETAINED)
    ]
    for state in range(40):
        for task in (0, 1):
            for role in ("new_unassisted", "new_teacher"):
                good = (state % 4 == 0) if role == "new_unassisted" else state % 3 == 0
                out.append(
                    dict(
                        index=len(out),
                        role=role,
                        task_success=good,
                        bc_weight=int(good),
                        source_course=dict(
                            state_id=state,
                            task=task,
                            assisted=role == "new_teacher",
                            target=[1.0, 2.0, 0.11],
                        ),
                    )
                )
    for c in RETAINED:
        out.append(
            dict(
                index=len(out),
                role="current_acquired17",
                source_context=c,
                task_success=True,
                bc_weight=1,
            )
        )
    for state in range(40):
        for task in (0, 1):
            good = state % 2 == 0
            out.append(
                dict(
                    index=len(out),
                    role="own_state_expert",
                    task_success=good,
                    bc_weight=int(good),
                    source_course=dict(
                        state_id=state, task=task, assisted=True, target=[1.0, 2.0, 0.11]
                    ),
                )
            )
    return out


def test_one_plan_per_task_preserves_raw_labels_and_already_successful_behavior():
    rows = records()
    before = copy.deepcopy(rows)
    p = single_mode_pass_curriculum(rows)
    assert rows == before and p == single_mode_pass_curriculum(rows)
    assert len(p["current17_anchor_indices"]) == 17
    assert len({(r["state_id"], r["task"]) for r in p["task_choices"]}) == len(p["task_choices"])
    assert all(rows[i]["task_success"] for i in p["fit_indices"])
    assert all(rows[i]["role"] == "new_unassisted" for i in p["selected_autonomous_indices"])
    for r in p["task_choices"]:
        if r["state_id"] % 4 == 0:
            assert r["role"] == "new_unassisted"
        elif r["state_id"] % 2 == 0:
            assert r["role"] == "own_state_expert"
        else:
            assert r["role"] == "new_teacher"
    assert set(p["all_negative_indices"]) == {r["index"] for r in rows if not r["task_success"]}
    assert not p["task_success_labels_modified"] and p["records_removed"] == 0


@pytest.mark.parametrize("fault", ["goal", "duplicate_task", "missing"])
def test_unsupported_or_incompletely_matched_task_cannot_enter_selection(fault):
    rows = records()
    if fault == "goal":
        rows[-1]["source_course"]["target"][0] = 99.0
    elif fault == "duplicate_task":
        rows[18]["source_course"]["state_id"] = 1
    else:
        rows.pop()
    with pytest.raises(ValueError):
        single_mode_pass_curriculum(rows)
