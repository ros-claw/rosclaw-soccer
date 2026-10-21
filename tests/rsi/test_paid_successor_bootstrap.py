import copy

import pytest

from scripts.rsi_search_paid_live_contact import qualified_successor_bootstrap


def proofs():
    isolated = {k: dict(quality=16, stable=16, regressions=[]) for k in ("old16", "dev16")}
    candidate = dict(
        training_receipt_hash="train",
        single_recurrent_actor=True,
        isolated=isolated,
        continuous=dict(
            stable=16,
            results=[dict(first_quality=True, world_stable=True, second={}) for _ in range(16)],
        ),
    )
    report = dict(
        status="PAID_LIVE_DISTILLED_CPU_FULL_WORLD_EXAM_CLOSED",
        entry_controller="candidate",
        all_whole_neural_native_repeats_exact=True,
        IK_executed=False,
        candidates=[candidate],
    )
    audit = dict(
        status="PAID_LIVE_DISTILLED_ALL_PRIMITIVE_AUDIT_PASS", input_file_hashes={"report": "hash"}
    )
    return report, audit


def test_task_specific_bootstrap_requires_safe_complete_successor_not_second_goal_success():
    report, audit = proofs()
    assert (
        qualified_successor_bootstrap(
            report, audit, report_path="report", report_hash="hash", training_hash="train"
        )
        is report["candidates"][0]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "old_regression",
        "missing_cohort",
        "body",
        "first_quality",
        "mixed_entry",
        "audit",
        "wrong_training",
    ],
)
def test_failed_partial_or_unbound_bootstrap_rejected(fault):
    report, audit = copy.deepcopy(proofs())
    c = report["candidates"][0]
    if fault == "old_regression":
        c["isolated"]["old16"]["regressions"] = [5]
    elif fault == "missing_cohort":
        c["isolated"].pop("dev16")
    elif fault == "body":
        c["continuous"]["stable"] = 15
    elif fault == "first_quality":
        c["continuous"]["results"][0]["first_quality"] = False
    elif fault == "mixed_entry":
        report["entry_controller"] = "qualified-prior"
    elif fault == "audit":
        audit["input_file_hashes"]["report"] = "other"
    else:
        c["training_receipt_hash"] = "other"
    with pytest.raises(ValueError):
        qualified_successor_bootstrap(
            report, audit, report_path="report", report_hash="hash", training_hash="train"
        )
