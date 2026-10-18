"""Read-only next-research handoff, not another score or promotion authority.

The caller authenticates whole input seals/files. A missing historical dense
endpoint remains missing; three dense pairs must never be advertised as four.
"""

import math
from typing import Any


def contact_exit_pilot_handoff(
    summary: dict[str, Any],
    complete: dict[str, Any],
    dense: dict[str, Any],
    baselines: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind every current endpoint and classify evidence for the next agent turn."""
    surface = summary.get("schema") == "soccer.rsi.mesh_surface_measured_summary.v1"
    cases, bodies = (6, 7) if surface else (4, 4)
    dense_count = len(dense.get("rows", []))
    if (
        summary.get("schema")
        != (
            "soccer.rsi.mesh_surface_measured_summary.v1"
            if surface
            else "soccer.rsi.directional_compliance_measured_summary.v1"
        )
        or complete.get("schema")
        != (
            "soccer.rsi.mesh_surface_swing_pilot_complete.v1"
            if surface
            else "soccer.rsi.directional_contact_compliance_pilot_complete.v1"
        )
        or dense.get("schema") != "soccer.rsi.contact_exit_metrics_readback.v1"
        or summary.get("complete_hash") != complete.get("report_hash")
        or summary.get("paired_development") != complete.get("paired_development")
        or len(summary.get("rows", [])) != cases
        or len(complete.get("results", [])) != bodies
        or len(baselines) != cases
        or dense_count not in ((10, 13) if surface else (8,))
        or summary.get("teacher_qualified") is not False
        or summary.get("new_optimizer_updates") != 0
        or complete.get("new_optimizer_updates") != 0
        or dense.get("new_optimizer_updates") != 0
        or summary.get("decision")
        not in ("REJECT_LOCAL_TEACHER", "LOCAL_PILOT_PASS_EXPANSION_REQUIRED")
    ):
        raise ValueError(
            "whole same-study four-pair audited summary and eight dense endpoints required"
        )
    if (
        len({b["pair_id"] for b in baselines}) != cases
        or len({b["receipt_hash"] for b in baselines}) != cases
        or {b["receipt_hash"] for b in baselines} & {r["receipt_hash"] for r in complete["results"]}
    ):
        raise ValueError("distinct original baseline receipts and four unique pairs required")
    by_identity = {}
    for row in dense["rows"]:
        identity = row["source_receipt_hash"]
        if identity in by_identity:
            raise ValueError("duplicate dense endpoint is not an independent case")
        by_identity[identity] = row
    records, paired_count, used = [], 0, set()
    for baseline, current, summarized in zip(
        baselines,
        complete["results"][1:] if surface else complete["results"],
        summary["rows"],
        strict=True,
    ):
        if (
            baseline["pair_id"] != current["pair_id"]
            or summarized["pair_id"] != current["pair_id"]
            or summarized["evaluated"]["outcome"] != current["outcome"]
            or current["receipt_hash"] not in by_identity
            or current["receipt_hash"] in used
        ):
            raise ValueError("every aligned current pair needs its exact dense receipt identity")
        used.add(current["receipt_hash"])
        new = by_identity[current["receipt_hash"]]
        old = by_identity.get(baseline["receipt_hash"])
        if new["original_outcome"] != current["outcome"] or (
            old is not None and old["original_outcome"] != summarized["baseline"]["outcome"]
        ):
            raise ValueError("dense primitive outcomes must match the whole audited policy outcome")
        paired_count += old is not None
        measurements: list[dict[str, Any] | None] = []
        for row in (old, new):
            if row is None:
                measurements.append(None)
                continue
            m = row["metrics"]
            event = m["first_nonfoot_robot_contact_substep"]
            if (event is not None and (type(event) is not int or not 0 <= event < 3000)) or any(
                type(m[k]) not in (float, int) or not math.isfinite(m[k]) or m[k] < 0
                for k in (
                    "total_robot_downward_impulse_n_s",
                    "maximum_linear_newton_residual_n",
                    "summed_active_foot_duration_s",
                )
            ):
                raise ValueError(
                    "finite dense diagnostic measurements and exact contact clock required"
                )
            if m["maximum_linear_newton_residual_n"] > 1e-7:
                raise ValueError("force/momentum inconsistency blocks research handoff")
            measurements.append(
                dict(
                    first_nonfoot_robot_contact_substep=event,
                    robot_downward_impulse_n_s=m["total_robot_downward_impulse_n_s"],
                    active_foot_duration_s=m["summed_active_foot_duration_s"],
                    first_ball_reversal_substep=m[
                        "first_forward_to_backward_crossing_after_initial_20ms_substep"
                    ],
                )
            )
        records.append(
            dict(
                pair_id=current["pair_id"],
                current_receipt_hash=current["receipt_hash"],
                baseline_dense_available=old is not None,
                baseline=measurements[0],
                evaluated=measurements[1],
                audited_outcome=current["outcome"],
            )
        )
    if surface and dense_count == 13 and paired_count != 6:
        raise ValueError("expanded dense evidence must cover all six original baseline pairs")
    nonfoot = [
        r["pair_id"]
        for r in records
        if r["evaluated"]["first_nonfoot_robot_contact_substep"] is not None
    ]
    task_failures = [r["pair_id"] for r in records if not r["audited_outcome"]["task_pass"]]
    # Do not overwrite M0 scoring. This separate all-body diagnostic only withholds
    # qualification, which even a successful four-case pilot cannot grant.
    local_pass = summary["decision"] == "LOCAL_PILOT_PASS_EXPANSION_REQUIRED"
    return dict(
        schema="soccer.rsi.mesh_surface_research_handoff.v1"
        if surface
        else "soccer.rsi.directional_compliance_research_handoff.v1",
        complete_hash=complete["report_hash"],
        summary_hash=summary["report_hash"],
        dense_readback_hash=dense["report_hash"],
        original_m0_score_unchanged=True,
        rows=records,
        dense_pair_count=paired_count,
        dense_single_endpoint_count=dense_count - 2 * paired_count,
        **{
            (
                "all_six_current_endpoints_have_dense_contact_review"
                if surface
                else "all_four_current_endpoints_have_dense_contact_review"
            ): True
        },
        dense_current_endpoint_count=cases,
        current_nonfoot_ball_contact_cases=nonfoot,
        original_task_failure_cases=task_failures,
        next_agent_action=(
            "REVIEW_ALL_BODY_CONTACT_BEFORE_EXPANSION"
            if local_pass and nonfoot
            else "DECLARE_EXPANDED_RETENTION_AND_PERTURBATION_STUDY"
            if local_pass
            else "REJECT_LOCAL_TEACHER_ANALYZE_CONTACT_AND_SUPPORT_REDESIGN"
        ),
        automatic_same_setting_retry_authorized=False,
        automatic_scientific_successor_launched=False,
        agent_reasoning_still_required=True,
        teacher_qualified=False,
        new_physical_steps=0,
        new_optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
