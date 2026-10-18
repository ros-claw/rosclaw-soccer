"""Synthetic proposed aggregation; counterfactual rows are NOT admitted data."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.reference_state_aggregation import (
    GROUPS,
    LABEL_CEILING,
    prepare_proposed_aggregation,
)
from rosclaw_soccer.rsi.reference_student_exam_scope import exam_scope
from rosclaw_soccer.rsi.reference_student_imitation import fit_numeric_reference_student

PARENT = "sha256:" + "a" * 64


def inputs():
    rng = np.random.default_rng(20261019)
    x = rng.normal(0, 0.1, (29, 300, 135))
    x[:, :, -1] = (300 - np.arange(300)) / 300
    y = np.zeros((29, 300, 12))
    y[:, 30:, 4] = 0.08
    receipts, meta = [], []
    for i in range(16):
        eligible = i in (1, 3, 9, 15)
        receipts.append(
            dict(
                role="teacher" if i % 2 else "control",
                group=GROUPS[i // 2],
                receipt_hash=f"original-{i}",
                outcome=dict(task_pass=eligible, safe=True),
            )
        )
        meta.append(
            dict(
                eligible=eligible,
                source_seed=11 + i // 4,
                partition="validation" if i >= 12 else "train",
            )
        )
    own_hashes = [f"own-{i}" for i in range(13)]
    kw = dict(
        original_receipts=receipts,
        original_metadata=meta,
        own_jobs=exam_scope(PARENT)["jobs"],
        own_receipt_hashes=own_hashes,
        query_receipt_hashes=own_hashes.copy(),
        parent_model_hash=PARENT,
        acknowledged_label_ceiling=LABEL_CEILING,
    )
    return [x[:16], y[:16], x[16:], y[16:]], kw


def test_complete_29_sources_keep_failed_and_retention_rows_but_exclude_gradients():
    arrays, kw = inputs()
    b, r = prepare_proposed_aggregation(*arrays, **kw)
    assert b["original_observations"].shape == (29, 300, 135)
    assert int(b["train"].sum()) == 6 and int(b["validation"].sum()) == 2
    assert r["proposed_train_positive_weight_frames"] == 1620
    assert r["proposed_train_source_seeds"] == [11, 13]
    assert r["proposed_validation_source_seeds"] == [14]
    assert len(r["rows"]) == 29 and r["total_recorded_frames"] == 8700
    assert not b["train"][-4:].any()  # raw retention bodies cannot be examples.
    assert not b["weights"][~b["train"]].any()
    assert not r["dataset_admission_authorized"] and not r["teacher_qualified"]
    assert not r["expert_success_from_student_states_verified"]


def test_oracle_targets_do_not_replace_actual_behaviour_policy_inputs():
    arrays, kw = inputs()
    b, _ = prepare_proposed_aggregation(*arrays, **kw)
    changed = copy.deepcopy(arrays)
    changed[3][1:, 100:, 0] = 0.3
    after, _ = prepare_proposed_aggregation(*changed, **kw)
    np.testing.assert_array_equal(after["original_observations"], b["original_observations"])
    assert not np.array_equal(after["targets"], b["targets"])


def test_validation_and_ineligible_own_failures_cannot_change_normalization():
    arrays, kw = inputs()
    b, _ = prepare_proposed_aggregation(*arrays, **kw)
    changed = copy.deepcopy(arrays)
    changed[0][~b["train"][:16], :, :-1] = 100
    changed[2][~b["train"][16:], :, :-1] = 100
    after, _ = prepare_proposed_aggregation(*changed, **kw)
    np.testing.assert_array_equal(after["mean"], b["mean"])
    np.testing.assert_array_equal(after["scale"], b["scale"])


@pytest.mark.parametrize(
    "tamper",
    [
        "ceiling",
        "partial",
        "job",
        "query_order",
        "duplicate",
        "wrong_teacher",
        "eligibility",
        "split",
        "train_only",
        "shape",
        "nan",
        "clock",
        "label_prefix",
        "label_cap",
    ],
)
def test_rejects_truncated_or_relabelled_unqualified_banks(tamper):
    arrays, kw = inputs()
    if tamper == "ceiling":
        kw["acknowledged_label_ceiling"] = "PHYSICALLY_QUALIFIED"
    elif tamper == "partial":
        kw["original_receipts"].pop()
    elif tamper == "job":
        kw["own_jobs"][-1]["group"] = 4
    elif tamper == "query_order":
        kw["query_receipt_hashes"].reverse()
    elif tamper == "duplicate":
        kw["own_receipt_hashes"][-1] = kw["own_receipt_hashes"][0]
        kw["query_receipt_hashes"] = kw["own_receipt_hashes"].copy()
    elif tamper == "wrong_teacher":
        kw["original_receipts"][1]["outcome"]["safe"] = False
    elif tamper == "eligibility":
        kw["original_metadata"][0]["eligible"] = True
    elif tamper == "split":
        kw["original_metadata"][1]["partition"] = "validation"
    elif tamper == "train_only":
        for m in kw["original_metadata"]:
            m["partition"] = "train"
    elif tamper == "shape":
        arrays[2] = arrays[2][:12]
    elif tamper == "nan":
        arrays[3][-1, 299, 0] = np.nan
    elif tamper == "clock":
        arrays[2][-1, 299, -1] = 0
    elif tamper == "label_prefix":
        arrays[3][0, 29, 0] = 0.1
    else:
        arrays[3][0, 40, 0] = 0.36
    with pytest.raises(ValueError):
        prepare_proposed_aggregation(*arrays, **kw)


def test_existing_core_learner_can_take_two_numeric_steps_on_six_aggregated_sequences():
    arrays, kw = inputs()
    batch, _ = prepare_proposed_aggregation(*arrays, **kw)
    fit = fit_numeric_reference_student(
        batch,
        world_hash="sha256:" + "b" * 64,
        parent_model_hash=PARENT,
        seed=2026101902,
        steps=2,
        ignore_past_reference=True,
    )
    assert fit["optimized_episode_count"] == 6
    assert fit["input_episode_count"] == 29
    assert fit["learning_receipt"]["completed_optimizer_steps"] == 2
    assert fit["learning_receipt"]["original_episode_count"] == 6
    assert fit["past_reference_input_ablation"]
    assert not fit["teacher_qualified"]
    assert not fit["numeric_model"]["physics_qualified"]
