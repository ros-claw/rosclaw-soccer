"""Whole six-body pulse metadata contract before recorded-torque diagnostics."""

import math
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.planner_navigation_pulse_evidence import require_pulse_physical_parity
from rosclaw_soccer.rsi.planner_navigation_pulse_scope import (
    assess_pulse_scope,
    pulse_scope,
    require_pulse_scope,
)
from rosclaw_soccer.sim.contracts import hash_json


def require_pulse_cohort(
    producer: dict[str, Any],
    declaration: dict[str, Any],
    complete: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Callers authenticate source files, units, receipts, audits and witnesses."""
    require_pulse_scope(producer)
    scope = pulse_scope(producer["behavior_model_hash"], profile=producer["exam_profile"])
    if (
        declaration.get("schema") != "soccer.rsi.pre_planner_pulse_six_declaration.v1"
        or complete.get("schema") != "soccer.rsi.pre_planner_pulse_six_complete.v1"
        or audit.get("schema") != "soccer.rsi.pre_planner_pulse_six_terminal_readback.v1"
        or any(declaration.get(k) != v for k, v in producer.items())
        or hash_json(declaration["scope"]) != hash_json(scope)
        or complete.get("declaration_hash") != declaration["report_hash"]
        or audit.get("complete_hash") != complete["report_hash"]
        or len(complete.get("results", [])) != 6
        or len(audit.get("rows", [])) != 6
        or any(
            type(complete.get(k)) is not int or complete[k] != n
            for k, n in (
                ("new_native_bodies", 6),
                ("new_independent_cpu_bodies", 6),
                ("native_substeps", 18000),
                ("independent_cpu_substeps", 18000),
                ("new_optimizer_updates", 0),
            )
        )
        or any(
            type(audit.get(k)) is not int or audit[k] != n
            for k, n in (
                ("source_native_bodies", 6),
                ("zero_reference_frames_reconstructed", 1800),
                ("new_physical_steps", 0),
                ("new_foundation_inferences", 0),
                ("optimizer_updates", 0),
            )
        )
        or any(
            doc.get(k) is not False
            for doc in (complete, audit)
            for k in ("teacher_qualified", "hardware_authorized", "promotion_authorized")
        )
        or audit.get("dataset_admission_authorized") is not False
    ):
        raise ValueError(
            "whole original six-body declaration, physical budget and zero-update audit required"
        )
    expected_model = declaration["zero_reference_model"]["model_hash"]
    for job, r, a in zip(scope["jobs"], complete["results"], audit["rows"], strict=True):
        if (
            any(r.get(k) != v for k, v in job.items())
            or a.get("job") != job
            or a.get("receipt_hash") != r["receipt_hash"]
            or a.get("outcome") != r["outcome"]
            or a.get("alignment") != r["qualification_alignment"]
            or r.get("candidate") != declaration["candidate"]
            or r.get("reference_student_model_hash") != expected_model
            or any(
                type(r.get(k)) is not int or r[k] != n
                for k, n in (
                    ("physical_substeps", 3000),
                    ("foundation_calls", 300),
                    ("sampling_draws", 270),
                    ("reference_student_inference_calls", 300),
                    ("neural_optimizer_updates", 0),
                )
            )
            or any(
                r.get(k) is not False
                for k in (
                    "hardware_authorized",
                    "promotion_authorized",
                    "teacher_qualified",
                    "private_fresh_accessed",
                )
            )
        ):
            raise ValueError(
                "all ordered original outcomes, zero models and physical receipt counts required"
            )
    assessment = assess_pulse_scope(scope, complete["results"])
    if assessment != complete.get("assessment") or assessment != audit.get("assessment"):
        raise ValueError(
            "whole original pair assessment must reconstruct including unsafe failures"
        )
    return assessment


def fixed_six_contact_rows(
    declaration: dict[str, Any],
    complete: dict[str, Any],
    replay: dict[str, Any],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Shared six-endpoint file/metric alignment, without model or scope admission."""
    jobs, receipts = declaration["scope"]["jobs"], complete["results"]
    if (
        replay.get("schema") != "soccer.rsi.fixed_control_contact_replay_complete.v1"
        or metrics.get("schema") != "soccer.rsi.contact_exit_metrics_readback.v1"
        or any(
            type(replay.get(k)) is not int or replay[k] != 18000
            for k in ("integrated_substeps", "attempted_step_calls")
        )
        or any(
            len(rows) != 6
            for rows in (jobs, receipts, replay.get("rows", []), metrics.get("rows", []))
        )
        or any(
            type(metrics.get(k)) is not int or metrics[k] != 0
            for k in ("new_physical_steps", "new_neural_inferences", "new_optimizer_updates")
        )
    ):
        raise ValueError("whole six-endpoint fixed 18000-step diagnostic required")
    indexed = {r["source_receipt_hash"]: r for r in metrics["rows"]}
    if len(indexed) != 6 or set(indexed) != {r["receipt_hash"] for r in receipts}:
        raise ValueError("all six original distinct endpoint metrics required")
    rows = []
    for job, receipt, row in zip(jobs, receipts, replay["rows"], strict=True):
        dense = indexed[receipt["receipt_hash"]]
        if (
            any(receipt.get(k) != v for k, v in job.items())
            or row["case_id"] != job["job"].replace("-", "_")
            or dense["case_id"] != row["case_id"]
            or row["source_receipt_hash"] != receipt["receipt_hash"]
            or dense["replay_result_hash"] != row["report_hash"]
            or row["original_outcome"] != receipt["outcome"]
            or dense["original_outcome"] != receipt["outcome"]
        ):
            raise ValueError(
                "original ordered receipts, replay rows and unchanged outcomes required"
            )
        m = dense["metrics"]
        for key in ("maximum_linear_newton_residual_n", "total_robot_downward_impulse_n_s"):
            if type(m.get(key)) not in (int, float) or not math.isfinite(m[key]) or m[key] < 0:
                raise ValueError("finite nonnegative actual contact diagnostics required")
        if m["maximum_linear_newton_residual_n"] > 1e-7:
            raise ValueError("original Newton consistency threshold cannot be relaxed")
        rows.append(
            dict(
                job=job,
                source_receipt_hash=receipt["receipt_hash"],
                original_outcome=receipt["outcome"],
                metrics=m,
            )
        )
    return rows


def pulse_contact_readout(
    declaration: dict[str, Any],
    complete: dict[str, Any],
    replay: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Whole fixed-torque diagnostics, not six further closed-loop trials."""
    rows = fixed_six_contact_rows(declaration, complete, replay, metrics)
    receipts = complete["results"]
    unsafe = sum(not r["original_outcome"]["safe"] for r in rows if r["job"]["role"] == "pulse")
    nonfoot = sum(
        r["metrics"]["first_nonfoot_robot_contact_substep"] is not None
        for r in rows
        if r["job"]["role"] == "pulse"
    )
    assessment = assess_pulse_scope(
        pulse_scope(
            declaration["behavior_model_hash"],
            profile=declaration.get("exam_profile", "pre-planner-navigation-six-v1"),
        ),
        receipts,
    )
    if assessment != complete["assessment"]:
        raise ValueError("all paired outcomes including failures must reconstruct")
    return dict(
        rows=rows,
        assessment=assessment,
        pulse_unsafe_endpoints=unsafe,
        pulse_endpoints_with_nonfoot_over1n=nonfoot,
        research_decision=(
            "UNSAFE_OR_NONFOOT_PULSE_NOT_ADMITTED"
            if unsafe or nonfoot
            else "MEASURED_AUTHORITY_DIAGNOSTIC_REQUIRES_MECHANISM_REVIEW_NOT_PROMOTION"
        ),
        original_task_labels_unchanged=True,
        new_closed_loop_policy_trials=0,
        new_optimizer_updates=0,
        derived_courses=3,
        source_scene_clusters=3,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        neural_training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )


def pulse_motion_readout(off: dict[str, Any], pulse: dict[str, Any]) -> dict[str, Any]:
    """Reuse measured reference/body channels, not a second physical experiment."""
    from rosclaw_soccer.rsi.reference_feedback_divergence import reference_feedback_divergence

    prefix = require_pulse_physical_parity(off, pulse, frames=100)
    feedback = reference_feedback_divergence(off, pulse)
    for t in (off, pulse):
        for key in ("requested", "applied"):
            if np.any(t[f"reference_student_{key}_reference_increment_rad"] != 0):
                raise ValueError("both arms must retain whole300 exact zero joint references")
    left, right = (np.asarray(t["navigation_command"]) for t in (off, pulse))
    if any(
        a.shape != (300, 1, 3) or a.dtype != np.float64 or not np.isfinite(a).all()
        for a in (left, right)
    ):
        raise ValueError("whole finite navigation command channels required")
    delta = np.abs(left - right).reshape(300, -1).max(axis=1)
    result = dict(
        schema="soccer.rsi.pre_planner_pulse_motion_readout.v1",
        before_action_prefix=prefix,
        measured_reference_body_ball_channels=feedback,
        maximum_navigation_component_difference_m_s=float(delta.max()),
        first_navigation_difference_frame=int(np.flatnonzero(delta > 1e-8)[0])
        if np.any(delta > 1e-8)
        else None,
        both_arms_exact_zero_joint_reference_all300=True,
        action_authority_is_not_task_improvement=True,
        first_difference_is_not_proven_causal_order=True,
        new_physical_steps=0,
        optimizer_updates=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))
