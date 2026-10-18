"""Synthetic source-path selection, not successful physical trajectories."""

import copy

import pytest

from rosclaw_soccer.rsi.actual_winner_supervision import select_actual_winner_paths

PARENT = "sha256:" + "a" * 64
GROUPS = (12, 13, 28, 29, 76, 77, 92, 93)


def bank():
    def rows(role, passed):
        return [
            dict(
                group=g,
                role=role,
                receipt_hash=f"{role}-{g}",
                outcome=dict(task_pass=g in passed, safe=not (role == "zero" and g == 77)),
            )
            for g in GROUPS
        ]

    return dict(
        parent_model_hash=PARENT,
        zeros=rows("zero", (12, 13, 92, 93)),
        teachers=rows("teacher", (12, 13, 76, 93)),
        zero_nonfoot={g: g == 77 for g in GROUPS},
        teacher_nonfoot={g: False for g in GROUPS},
    )


def test_baseline_success_is_positive_zero_action_supervision_not_missing_data():
    kw = bank()
    original = copy.deepcopy(kw)
    result = select_actual_winner_paths(**kw)
    assert kw == original
    assert result["selected_train_paths"] == 3
    assert result["selected_zero_train_anchors"] == 2
    assert result["selected_corrective_train_paths"] == 1
    assert result["selected_train_sources"] == [20261077, 20261093]
    assert result["selected_validation_paths"] == 2
    assert len(result["rows"]) == 16 and len(result["choices"]) == 8
    selected = [
        (r["group"], r["role"], r["label_kind"])
        for r in result["rows"]
        if r["experimental_actual_path_supervision"]
    ]
    assert selected == [
        (12, "zero", "ACTUAL_NAV_ZERO_ACTIONS"),
        (13, "zero", "ACTUAL_NAV_ZERO_ACTIONS"),
        (76, "teacher", "ACTUAL_IK_ACTIONS"),
        (92, "zero", "ACTUAL_NAV_ZERO_ACTIONS"),
        (93, "zero", "ACTUAL_NAV_ZERO_ACTIONS"),
    ]
    assert result["successful_path_union_is_not_one_executable_policy"]
    assert not result["teacher_qualified"] and not result["neural_training_authorized"]
    assert (
        not result["dataset_admission_authorized"]
        and not result["source_file_authentication_performed_here"]
    )


def test_actual_primitive_pass_cannot_hide_dense_nonfoot_and_failures_are_retained():
    kw = bank()
    kw["teacher_nonfoot"][76] = True
    result = select_actual_winner_paths(**kw)
    assert result["selected_corrective_train_paths"] == 0
    assert len(result["rows"]) == 16
    row = next(r for r in result["rows"] if r["group"] == 76 and r["role"] == "teacher")
    assert row["original_outcome"]["task_pass"] is True
    assert not row["experimental_actual_path_supervision"]


def test_validation_changes_cannot_change_selected_training_paths():
    kw = bank()
    before = select_actual_winner_paths(**kw)
    for r in kw["zeros"] + kw["teachers"]:
        if r["group"] in (92, 93):
            r["outcome"] = dict(task_pass=False, safe=True)
    after = select_actual_winner_paths(**kw)
    assert [r for r in before["rows"] if r["partition"] == "train"] == [
        r for r in after["rows"] if r["partition"] == "train"
    ]
    assert after["selected_validation_paths"] == 0


@pytest.mark.parametrize(
    "tamper",
    ["partial", "order", "role", "duplicate", "bool", "unsafe_pass", "dense", "dense_type"],
)
def test_rejects_selected_success_prefix_or_bad_primitive_safety(tamper):
    kw = bank()
    if tamper == "partial":
        kw["zeros"].pop()
    elif tamper == "order":
        kw["teachers"].reverse()
    elif tamper == "role":
        kw["zeros"][0]["role"] = "teacher"
    elif tamper == "duplicate":
        kw["zeros"][0]["receipt_hash"] = kw["teachers"][0]["receipt_hash"]
    elif tamper == "bool":
        kw["zeros"][0]["outcome"]["task_pass"] = 1
    elif tamper == "unsafe_pass":
        kw["zeros"][0]["outcome"]["safe"] = False
    elif tamper == "dense":
        kw["zero_nonfoot"].pop(77)
    else:
        kw["zero_nonfoot"][77] = 1
    with pytest.raises(ValueError):
        select_actual_winner_paths(**kw)
