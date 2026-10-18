import copy

import pytest

from rosclaw_soccer.rsi.contact_demonstration_inventory import ENDPOINTS, inventory


def rows():
    return [
        dict(
            case_id=case,
            source_receipt_hash=f"receipt-{i}",
            original_outcome=dict(
                task_pass=i % 2 == 0,
                safe=True,
                quality_observed=True,
                first_foot_contact_frame=74,
            ),
            metrics=dict(
                schema="soccer.physics.contact_exit_metrics.v1",
                first_foot_contact_substep=745,
                first_nonfoot_robot_contact_substep=None,
                new_physical_steps=0,
                new_optimizer_updates=0,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        for i, case in enumerate(ENDPOINTS)
    ]


def test_whole_historical_table_retains_failure_without_authorizing_learning():
    source = rows()
    before = copy.deepcopy(source)
    result = inventory(source, dense_report_hash="dense")
    assert source == before
    assert result["historical_endpoint_count"] == 18
    assert result["independent_course_count"] == 6
    assert result["provisional_success_trace_count"] == 9
    assert sum(r["failure_retained"] for r in result["rows"]) == 9
    for key in (
        "hindsight_union_is_causal_selector",
        "hindsight_union_is_generalization",
        "local_perturbation_reproducibility_verified",
        "teacher_qualified",
        "dataset_admission_authorized",
        "new_training_authorized",
        "promotion_authorized",
        "hardware_authorized",
    ):
        assert result[key] is False


def test_previously_passed_but_late_nonfoot_contact_is_not_a_success_example():
    source = rows()
    source[0]["metrics"]["first_nonfoot_robot_contact_substep"] = 2900
    result = inventory(source, dense_report_hash="dense")
    assert not result["rows"][0]["provisional_success_trace"]
    assert result["rows"][0]["original_outcome"]["task_pass"]
    assert result["rows"][0]["failure_retained"]


@pytest.mark.parametrize(
    "tamper", ["partial", "duplicate", "receipt", "retag", "unsafe", "quality", "clock", "bool"]
)
def test_rejects_partial_scope_and_inconsistent_physics(tamper):
    source = rows()
    if tamper == "partial":
        source.pop()
    elif tamper == "duplicate":
        source[1]["case_id"] = source[0]["case_id"]
    elif tamper == "receipt":
        source[1]["source_receipt_hash"] = source[0]["source_receipt_hash"]
    elif tamper == "retag":
        source[0]["case_id"] = "new-course"
    elif tamper == "unsafe":
        source[0]["original_outcome"]["safe"] = False
    elif tamper == "quality":
        source[0]["original_outcome"]["quality_observed"] = False
    elif tamper == "clock":
        source[0]["metrics"]["first_foot_contact_substep"] = 755
    else:
        source[0]["original_outcome"]["task_pass"] = 1
    with pytest.raises(ValueError):
        inventory(source, dense_report_hash="dense")


def test_perfect_hindsight_union_still_cannot_claim_a_deployable_teacher():
    source = rows()
    for row in source:
        row["original_outcome"]["task_pass"] = True
    result = inventory(source, dense_report_hash="dense")
    assert result["hindsight_union_covered_courses"] == 6
    assert result["teacher_qualified"] is False
    assert result["new_training_authorized"] is False
