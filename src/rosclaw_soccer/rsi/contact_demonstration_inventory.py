"""Inventory consumed public traces; an oracle union is NOT a deployable teacher.

This read-only accounting never admits a dataset or selects an action. A passed
episode can be a provisional example while its global controller is rejected.
Nearby-state reproducibility and a causal runtime selector remain unproved.
"""

from typing import Any

from rosclaw_soccer.sim.contracts import hash_json

COURSES = ("raw-4", "raw-8", "raw-32", "raw-56", "context-0", "context-1")
FAMILIES = ("none", "body-origin", "mesh-surface")
ENDPOINTS = {
    "none_raw_4": ("none", "raw-4"),
    "none_raw_8": ("none", "raw-8"),
    "none_raw_32": ("none", "raw-32"),
    "none_raw_56": ("none", "raw-56"),
    "screen_none_context0": ("none", "context-0"),
    "none_context_1": ("none", "context-1"),
    "catalog_option4_raw8": ("body-origin", "raw-8"),
    "catalog_option4_raw56": ("body-origin", "raw-56"),
    "screen_option4_context0": ("body-origin", "context-0"),
    "baseline_context_1": ("body-origin", "context-1"),
    "baseline_raw_4": ("body-origin", "raw-4"),
    "baseline_raw_32": ("body-origin", "raw-32"),
    **{"surface_" + course.replace("-", "_"): ("mesh-surface", course) for course in COURSES},
}


def inventory(rows: list[dict[str, Any]], *, dense_report_hash: str) -> dict[str, Any]:
    """Require the whole 6x3 table, including failures, before counting coverage.

    Caller authenticates complete original studies, witness bytes, original
    primitive labels and independent review alignment. Hashes do not authenticate
    a hostile issuer; this function does not claim to do so.
    """
    if (
        type(rows) is not list
        or len(rows) != 18
        or len({r["case_id"] for r in rows}) != 18
        or {r["case_id"] for r in rows} != set(ENDPOINTS)
        or len({r["source_receipt_hash"] for r in rows}) != 18
    ):
        raise ValueError("complete distinct eighteen endpoints including all failures required")
    indexed = {r["case_id"]: r for r in rows}
    result_rows = []
    for case, (family, course) in ENDPOINTS.items():
        row = indexed[case]
        outcome, metrics = row["original_outcome"], row["metrics"]
        if (
            any(type(outcome.get(k)) is not bool for k in ("task_pass", "safe", "quality_observed"))
            or metrics.get("schema") != "soccer.physics.contact_exit_metrics.v1"
            or metrics.get("new_physical_steps") != 0
            or metrics.get("new_optimizer_updates") != 0
            or metrics.get("hardware_authorized") is not False
            or metrics.get("promotion_authorized") is not False
        ):
            raise ValueError(
                "typed unchanged primitive outcomes and read-only contact metrics required"
            )
        foot, nonfoot = (
            metrics["first_foot_contact_substep"],
            metrics["first_nonfoot_robot_contact_substep"],
        )
        for value in (foot, nonfoot):
            if value is not None and (type(value) is not int or not 0 <= value < 3000):
                raise ValueError("complete dense clock required")
        original_frame = outcome["first_foot_contact_frame"]
        if (None if foot is None else foot // 10) != original_frame:
            raise ValueError("actual dense foot-contact clock must match original primitive label")
        provisional = (
            outcome["task_pass"]
            and outcome["safe"]
            and outcome["quality_observed"]
            and nonfoot is None
        )
        if outcome["task_pass"] and (not outcome["safe"] or not outcome["quality_observed"]):
            raise ValueError("success cannot override unsafe or unobserved original outcome")
        result_rows.append(
            dict(
                case_id=case,
                family=family,
                course=course,
                source_receipt_hash=row["source_receipt_hash"],
                original_outcome=outcome,
                first_nonfoot_robot_contact_over_1n_substep=nonfoot,
                provisional_success_trace=provisional,
                failure_retained=not provisional,
                dataset_admitted=False,
                reproducible_local_teacher_qualified=False,
            )
        )
    coverage = {
        course: [
            r["family"]
            for r in result_rows
            if r["course"] == course and r["provisional_success_trace"]
        ]
        for course in COURSES
    }
    result = dict(
        schema="soccer.rsi.contact_demonstration_inventory.v1",
        dense_report_hash=dense_report_hash,
        rows=result_rows,
        independent_course_count=6,
        historical_endpoint_count=18,
        consumed_public_courses=True,
        family_success_trace_counts={
            family: sum(
                r["provisional_success_trace"] for r in result_rows if r["family"] == family
            )
            for family in FAMILIES
        },
        provisional_success_trace_count=sum(r["provisional_success_trace"] for r in result_rows),
        per_course_success_trace_families=coverage,
        hindsight_union_covered_courses=sum(bool(families) for families in coverage.values()),
        hindsight_union_is_causal_selector=False,
        hindsight_union_is_generalization=False,
        local_perturbation_reproducibility_verified=False,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        new_training_authorized=False,
        new_physical_steps=0,
        new_optimizer_updates=0,
        promotion_authorized=False,
        hardware_authorized=False,
        next_required_work="qualify bounded state perturbations before student dataset admission",
    )
    return dict(result, report_hash=hash_json(result))
