"""Whole fitted-student cohort and dense contact readout, never promotion.

Callers authenticate file bytes and original source/process ownership. This
module preserves every primitive outcome; a newly observed dense collision
rejects interpretation without relabelling old receipts or changing the exam.
"""

import math

from rosclaw_soccer.rsi.reference_student_exam_scope import assess_exam, exam_scope, zero_model_like
from rosclaw_soccer.rsi.reference_student_execution import own_model
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_fit_pre_parent_reference import FIT_SEED
from scripts.rsi_reference_student_exam import require_scope


def require_cohort(config, declaration, complete, audit):
    require_scope(config)
    scope = exam_scope(config["behavior_model_hash"])
    if (
        declaration.get("schema") != "soccer.rsi.reference_student_exam_declaration.v1"
        or complete.get("schema") != "soccer.rsi.reference_student_exam_complete.v1"
        or audit.get("schema") != "soccer.rsi.reference_student_exam_terminal_readback.v1"
        or complete.get("declaration_hash") != declaration["report_hash"]
        or audit.get("complete_hash") != complete["report_hash"]
        or any(declaration.get(k) != v for k, v in config.items())
        or hash_json(declaration["scope"]) != hash_json(scope)
        or len(complete.get("results", [])) != 13
        or len(audit.get("rows", [])) != 13
        or any(
            type(complete.get(k)) is not int or complete[k] != n
            for k, n in (
                ("new_native_bodies", 13),
                ("new_independent_cpu_bodies", 13),
                ("native_substeps", 39000),
                ("independent_cpu_substeps", 39000),
                ("new_optimizer_updates", 0),
            )
        )
        or any(
            type(audit.get(k)) is not int or audit[k] != n
            for k, n in (
                ("source_native_bodies", 13),
                ("causal_reference_frames_reconstructed", 3600),
                ("new_physical_steps", 0),
                ("new_optimizer_updates", 0),
            )
        )
        or any(
            audit.get(k) is not True
            for k in (
                "whole_input_and_witness_files_verified",
                "shared_payload_logical_seals_rebuilt",
                "original_off_all33_arrays_bit_exact",
                "all_pre_action_prefixes_bit_exact",
            )
        )
        or any(
            doc.get(k) is not False
            for doc in (complete, audit)
            for k in (
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
                "fresh_exam",
            )
        )
        or audit.get("official_52_case_exam_completed") is not False
        or audit.get("successor_readiness_qualified") is not False
    ):
        raise ValueError("whole original thirteen-body own-student audit required")
    models = declaration["models"]
    if set(models) != {"off", "zero", "student"} or models["off"] is not None:
        raise ValueError("exact three original reference model roles required")
    for role in ("zero", "student"):
        model = own_model(models[role])
        if (
            model["world_hash"] != config["mesh_task_contact_world_hash"]
            or model["parent_model_hash"] != config["behavior_model_hash"]
        ):
            raise ValueError("unchanged world and full original parent identity required")
    if models["zero"] != zero_model_like(models["student"], seed=FIT_SEED):
        raise ValueError(
            "zero control must retain fitted normalization and original initialization"
        )
    if len({r["receipt_hash"] for r in complete["results"]}) != 13:
        raise ValueError("thirteen distinct original receipts required")
    for job, receipt, row in zip(scope["jobs"], complete["results"], audit["rows"], strict=True):
        model = models[job["role"]]
        model_hash = None if model is None else model["model_hash"]
        if (
            any(receipt.get(k) != v for k, v in job.items())
            or receipt["candidate"] != declaration["candidate"]
            or row["job"] != job
            or row["receipt_hash"] != receipt["receipt_hash"]
            or row["outcome"] != receipt["outcome"]
            or row["model_hash"] != model_hash
            or receipt.get("reference_student_model_hash") != model_hash
        ):
            raise ValueError("whole ordered original jobs, model identities and failures required")
    paired = assess_exam(
        scope,
        complete["results"],
        teachers={int(k): v["outcome"] for k, v in declaration["reused_teachers"].items()},
        original_retention={
            int(k): v["outcome"] for k, v in declaration["reused_legacy_retention"].items()
        },
    )
    if paired != complete["paired_results"] or paired != audit["paired_results"]:
        raise ValueError("all positive and legacy/zero retention pairings must reconstruct")
    return paired


def measured_readout(declaration, complete, replay, metrics):
    scope = declaration["scope"]
    if (
        replay.get("schema") != "soccer.rsi.fixed_control_contact_replay_complete.v1"
        or metrics.get("schema") != "soccer.rsi.contact_exit_metrics_readback.v1"
        or any(
            type(replay.get(k)) is not int or replay[k] != 39000
            for k in ("integrated_substeps", "attempted_step_calls")
        )
        or any(
            len(v) != 13
            for v in (
                scope["jobs"],
                complete["results"],
                replay.get("rows", []),
                metrics.get("rows", []),
            )
        )
    ):
        raise ValueError("whole thirteen-endpoint fixed 39000-step diagnostic required")
    indexed = {r["source_receipt_hash"]: r for r in metrics["rows"]}
    if len(indexed) != 13 or set(indexed) != {r["receipt_hash"] for r in complete["results"]}:
        raise ValueError(
            "all original receipt memberships, not a selected success prefix, required"
        )
    rows, unsafe, nonfoot = [], 0, 0
    for job, receipt, row in zip(scope["jobs"], complete["results"], replay["rows"], strict=True):
        dense = indexed[receipt["receipt_hash"]]
        if (
            row["case_id"] != job["job"].replace("-", "_")
            or dense["case_id"] != row["case_id"]
            or row["source_receipt_hash"] != receipt["receipt_hash"]
            or row["original_outcome"] != receipt["outcome"]
            or dense["original_outcome"] != receipt["outcome"]
        ):
            raise ValueError("exact original dense case order and outcomes required")
        m = dense["metrics"]
        for key in ("maximum_linear_newton_residual_n", "total_robot_downward_impulse_n_s"):
            if type(m.get(key)) not in (int, float) or not math.isfinite(m[key]) or m[key] < 0:
                raise ValueError("finite nonnegative actual contact diagnostics required")
        if m["maximum_linear_newton_residual_n"] > 1e-7:
            raise ValueError("original Newton consistency threshold cannot be relaxed")
        if job["role"] == "student":
            unsafe += not receipt["outcome"]["safe"]
            nonfoot += m["first_nonfoot_robot_contact_substep"] is not None
        rows.append(
            dict(
                job=job,
                source_receipt_hash=receipt["receipt_hash"],
                original_outcome=receipt["outcome"],
                metrics=m,
            )
        )
    pairs = complete["paired_results"]
    positive = pairs["positive_teacher_vs_student"]
    retention = [pairs[k] for k in ("retention_zero_vs_student", "retention_legacy_vs_student")]
    if (
        unsafe
        or nonfoot
        or positive["newly_unsafe_pairs"]
        or any(r["newly_unsafe_pairs"] for r in retention)
    ):
        decision = "REJECT_STUDENT_UNSAFE_OR_DENSE_NONFOOT_RETAIN_ALL_ORIGINAL_LABELS"
    elif any(r["regressed_pairs"] for r in retention):
        decision = "RETENTION_REGRESSION_REQUIRES_REVIEW_NOT_PROMOTION"
    elif positive["evaluated_passes"] < positive["baseline_passes"]:
        decision = "STUDENT_BELOW_TEACHER_REQUIRES_OWN_STATE_DATA_REVIEW"
    elif positive["regressed_pairs"]:
        decision = "MIXED_POSITIVE_RESCUE_REGRESSION_REQUIRES_DOMAIN_REVIEW"
    elif positive["net_success_gain"] > 0:
        decision = "LOCAL_STUDENT_GAIN_REQUIRES_THREE_SEEDS_FRESH_AND_SKILL_CHAIN"
    else:
        decision = "NO_NET_STUDENT_SKILL_GROWTH_REQUIRES_OWN_STATE_DATA_REVIEW"
    return dict(
        rows=rows,
        research_decision=decision,
        student_unsafe_endpoints=unsafe,
        student_endpoints_with_nonfoot_over1n=nonfoot,
        paired_results=pairs,
        distinct_derived_course_count=10,
        source_scene_clusters=6,
        original_task_labels_unchanged=True,
        diagnostic_is_not_a_new_exam_gate=True,
        naturalness_or_successor_readiness_qualified=False,
        neural_training_authorized=False,
        dataset_admission_authorized=False,
        teacher_qualified=False,
        fresh_exam=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
