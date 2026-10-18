"""Pure metadata/paired labels; no actual student or physical improvement."""

import copy
import json

import numpy as np
import pytest
from rosclaw.growth.causal_residual_memory import initial_parameters

from rosclaw_soccer.rsi.reference_student_exam_scope import assess_exam, exam_scope, zero_model_like
from rosclaw_soccer.rsi.reference_student_execution import make_model, own_model

PARENT = "sha256:" + "a" * 64


def test_fixed_study_is_not_thirteen_independent_courses_or_fresh():
    s = exam_scope(PARENT)
    assert s["new_native_bodies"] == 13 and s["combined_body_substeps"] == 78000
    assert s["distinct_derived_course_count"] == 10 and s["source_scene_clusters"] == 6
    assert s["jobs"][0]["role"] == "off"
    assert len([j for j in s["jobs"] if j["plan"] == "positive19" and j["role"] == "student"]) == 8
    assert len([j for j in s["jobs"] if j["plan"] == "original18"]) == 4
    assert not s["fresh_exam"] and not s["original_52_case_exam_completed"]


def test_zero_controls_preserve_the_same_normalization_not_old_probe_defaults():
    p = initial_parameters(135, 12, seed=2)
    p["head_bias"][4] = 0.2
    mean, scale = np.full(135, 0.2), np.full(135, 2.0)
    mean[-1], scale[-1] = 0, 1
    student = make_model(
        p, mean=mean, scale=scale, world_hash="sha256:" + "b" * 64, parent_model_hash=PARENT
    )
    zero = zero_model_like(student, seed=2)
    assert own_model(zero) == zero
    assert zero["mean"] == student["mean"] and zero["scale"] == student["scale"]
    assert zero["parameters"]["head_bias"] == [0.0] * 12
    expected = copy.deepcopy(student["parameters"])
    expected["head_bias"][4] = 0
    assert zero["parameters"] == expected


def data():
    s = exam_scope(PARENT)
    receipt = [dict(j, outcome=dict(task_pass=True, safe=True)) for j in s["jobs"]]
    teacher = {
        g: dict(task_pass=i % 2 == 0, safe=True)
        for i, g in enumerate((12, 13, 28, 29, 76, 77, 92, 93))
    }
    old = {g: dict(task_pass=True, safe=True) for g in (4, 32)}
    return s, receipt, teacher, old


def test_development_rescues_do_not_hide_retention_regression():
    s, r, t, old = data()
    r[-1]["outcome"] = dict(task_pass=False, safe=False)
    result = assess_exam(s, r, teachers=t, original_retention=old)
    assert result["positive_teacher_vs_student"]["rescued_pairs"] == 4
    assert result["retention_zero_vs_student"]["newly_unsafe_pairs"] == 1
    assert result["retention_legacy_vs_student"]["regressed_pairs"] == 1
    assert result["legacy_comparison_confounded_by_approach_change"]
    assert not result["promotion_authorized"]


def test_scope_and_pairing_survive_real_json_artifact_roundtrip():
    s, r, t, old = data()
    serialized = json.loads(json.dumps(s))
    assert assess_exam(serialized, r, teachers=t, original_retention=old) == assess_exam(
        s, r, teachers=t, original_retention=old
    )


def test_sorted_json_mapping_order_cannot_change_declared_pair_rows():
    s, r, t, old = data()
    serialized = json.loads(json.dumps({str(k): v for k, v in old.items()}, sort_keys=True))
    changed = {int(k): v for k, v in serialized.items()}
    assert list(changed) == [32, 4]
    assert assess_exam(s, r, teachers=t, original_retention=changed) == assess_exam(
        s, r, teachers=t, original_retention=old
    )


@pytest.mark.parametrize(
    "tamper", ["partial", "order", "duplicate", "teacher_failure", "old_success", "scope"]
)
def test_no_partial_bank_retagging_or_deleted_baseline_failures(tamper):
    s, r, t, old = data()
    if tamper == "partial":
        r.pop()
    elif tamper == "order":
        r.reverse()
    elif tamper == "duplicate":
        r[2] = r[1]
    elif tamper == "teacher_failure":
        t.pop(13)
    elif tamper == "old_success":
        old[4]["task_pass"] = False
    else:
        s["source_scene_clusters"] = 13
    with pytest.raises(ValueError):
        assess_exam(s, r, teachers=t, original_retention=old)
