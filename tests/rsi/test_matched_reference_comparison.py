"""Synthetic matched-cohort decisions only; no physical or training claims."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.matched_reference_comparison import compare_matched_references
from rosclaw_soccer.rsi.reference_exam_profiles import assess_reference
from rosclaw_soccer.rsi.reference_student_exam_diagnostics import measured_readout
from rosclaw_soccer.rsi.reference_student_exam_scope import zero_model_like
from rosclaw_soccer.rsi.reference_student_execution import make_model
from tests.rsi.test_reference_positive_zero import dense as zero_dense
from tests.rsi.test_reference_positive_zero import zero_cohort
from tests.rsi.test_reference_student_exam_diagnostics import cohort, dense


def matched():
    result = []
    for make, make_dense in ((zero_cohort, zero_dense), (cohort, dense)):
        cfg, d, c, a = make()
        _, _, replay, metrics = make_dense()
        result.append(
            dict(config=cfg, declaration=d, complete=c, audit=a, replay=replay, metrics=metrics)
        )
    return result


def refresh(c):
    d, complete, audit = (c[k] for k in ("declaration", "complete", "audit"))
    for receipt, row, replay, metric in zip(
        complete["results"], audit["rows"], c["replay"]["rows"], c["metrics"]["rows"], strict=True
    ):
        for target, key in (
            (row, "outcome"),
            (replay, "original_outcome"),
            (metric, "original_outcome"),
        ):
            target[key] = copy.deepcopy(receipt["outcome"])
    pairs = assess_reference(
        c["config"]["exam_profile"],
        d["scope"],
        complete["results"],
        teachers={int(k): v["outcome"] for k, v in d["reused_teachers"].items()},
        original_retention={int(k): v["outcome"] for k, v in d["reused_legacy_retention"].items()},
    )
    complete["paired_results"] = pairs
    audit["paired_results"] = copy.deepcopy(pairs)
    c["measured"] = measured_readout(d, complete, c["replay"], c["metrics"])
    return c


def zero_fails(c):
    for r in c["complete"]["results"]:
        if r["role"] == "zero":
            r["outcome"] = dict(task_pass=False, safe=True)


def test_navigation_already_stronger_does_not_motivate_blind_imitation():
    z, s = matched()
    report = compare_matched_references(refresh(z), refresh(s))
    assert report["comparisons"]["zero_vs_teacher"]["net_success_gain"] == -4
    assert report["comparisons"]["zero_vs_student"]["net_success_gain"] == 0
    assert report["student_disposition"].startswith("NO_POSITIVE_NET")
    assert report["next_research"].startswith("IMPROVE_EXPERT")
    assert not report["expert_advantage_training_sources"]
    assert not report["neural_training_authorized"] and report["optimizer_updates"] == 0


def test_actual_winner_norm_change_needs_explicit_zero_equivalence_not_fake_same_model():
    z, s = matched()
    marker = dict(fit_configuration="test", fit_configuration_file_hash="test", fit_unit="test")
    s["config"]["actual_winner_fit"] = marker
    s["declaration"]["actual_winner_fit"] = marker
    models = s["declaration"]["models"]
    old = models["student"]
    mean = np.asarray(old["mean"]).copy()
    mean[0] += 1.0
    models["student"] = make_model(
        old["parameters"],
        mean=mean,
        scale=np.asarray(old["scale"]),
        world_hash=old["world_hash"],
        parent_model_hash=old["parent_model_hash"],
    )
    models["zero"] = zero_model_like(models["student"], seed=2026101902)
    for receipt, audit in zip(s["complete"]["results"], s["audit"]["rows"], strict=True):
        m = models[receipt["role"]]
        if m is not None:
            receipt["reference_student_model_hash"] = audit["model_hash"] = m["model_hash"]
    refresh(z)
    refresh(s)
    with pytest.raises(ValueError):
        compare_matched_references(z, s)
    result = compare_matched_references(z, s, allow_zero_normalization_reuse=True)
    assert result["recorded_zero_control_equivalence"]["hidden_states_may_differ"]
    assert result["zero_normalization_reuse_is_not_nonzero_student_equivalence"]
    del s["config"]["actual_winner_fit"]
    with pytest.raises(ValueError):
        compare_matched_references(z, s, allow_zero_normalization_reuse=True)


def test_two_or_more_train_sources_with_expert_gain_motivates_research_not_admission():
    z, s = matched()
    zero_fails(z)
    report = compare_matched_references(refresh(z), refresh(s))
    assert report["expert_advantage_training_sources"] == [20261077, 20261078, 20261093]
    assert report["teacher_minus_zero_train_net"] == 3
    assert report["next_research"].startswith("BOUNDED_SAME_STATE")
    assert report["comparisons"]["zero_vs_student"]["net_success_gain"] == 8
    assert report["student_disposition"].startswith("LOCAL_MATCHED")
    assert (
        not report["promotion_authorized"] and not report["counterfactual_query_success_verified"]
    )


def test_one_scene_advantage_is_not_reproducible_train_source_support():
    z, s = matched()
    for r in z["complete"]["results"]:
        if r["role"] == "zero" and r["group"] in (12, 13):
            r["outcome"] = dict(task_pass=False, safe=True)
    report = compare_matched_references(refresh(z), refresh(s))
    assert report["expert_advantage_training_sources"] == [20261077]
    assert report["next_research"].startswith("IMPROVE_EXPERT")


def test_legacy_loss_shared_by_zero_and_student_identifies_approach_scope():
    z, s = matched()
    zero_fails(z)
    for r in s["complete"]["results"]:
        if r["plan"] == "original18" and r["group"] == 32:
            r["outcome"] = dict(task_pass=False, safe=True)
    report = compare_matched_references(refresh(z), refresh(s))
    assert report["student_disposition"].startswith("REVIEW_APPROACH_APPLICABILITY")
    assert report["retention_zero_vs_student"]["regressed_pairs"] == 0
    assert report["retention_legacy_vs_student"]["regressed_pairs"] == 1


@pytest.mark.parametrize("validation_pass", [False, True])
def test_validation_cannot_select_next_training_family(validation_pass):
    z, s = matched()
    for c in (z, s):
        for r in c["complete"]["results"]:
            if r["plan"] == "positive19" and r["group"] in (92, 93):
                r["outcome"] = dict(task_pass=validation_pass, safe=True)
    report = compare_matched_references(refresh(z), refresh(s))
    assert report["next_research"].startswith("IMPROVE_EXPERT")
    assert report["teacher_minus_zero_train_net"] == -3
    assert report["validation_used_for_research_family_choice"] is False


@pytest.mark.parametrize("failure", ["unsafe", "nonfoot", "retention"])
def test_ball_gain_does_not_excuse_unsafe_or_nonfoot_or_retention_regression(failure):
    z, s = matched()
    zero_fails(z)
    if failure == "unsafe":
        s["complete"]["results"][1]["outcome"] = dict(task_pass=False, safe=False)
    elif failure == "nonfoot":
        s["metrics"]["rows"][1]["metrics"]["first_nonfoot_robot_contact_substep"] = 1400
    else:
        s["complete"]["results"][10]["outcome"] = dict(task_pass=False, safe=True)
    report = compare_matched_references(refresh(z), refresh(s))
    assert report["student_disposition"].startswith("REJECT_")


@pytest.mark.parametrize(
    "tamper",
    [
        "approach",
        "normalization",
        "teachers",
        "drop",
        "dense_order",
        "dense_receipt",
        "dense_outcome",
    ],
)
def test_no_mismatched_source_or_selected_success_comparison(tamper):
    z, s = matched()
    refresh(z)
    refresh(s)
    if tamper == "approach":
        s["declaration"]["candidate"] = dict(different="approach")
    elif tamper == "normalization":
        s["declaration"]["models"]["zero"]["mean"][0] = 1
    elif tamper == "teachers":
        s["declaration"]["reused_teachers"]["12"]["outcome"]["task_pass"] = False
    elif tamper == "drop":
        z["measured"]["rows"].pop()
    elif tamper == "dense_order":
        z["measured"]["rows"].reverse()
    elif tamper == "dense_receipt":
        z["measured"]["rows"][1]["source_receipt_hash"] = "borrowed"
    elif tamper == "dense_outcome":
        z["measured"]["rows"][1]["original_outcome"] = dict(task_pass=False, safe=True)
    with pytest.raises(ValueError):
        compare_matched_references(z, s)
