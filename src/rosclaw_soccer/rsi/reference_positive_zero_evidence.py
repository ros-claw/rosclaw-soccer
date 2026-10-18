"""Whole nine-body NAV-only baseline evidence; never a learned-skill claim."""

import math

from rosclaw_soccer.rsi.reference_exam_profiles import (
    ZERO,
    assess_reference,
    positive_zero_scope,
    profile_info,
)
from rosclaw_soccer.rsi.reference_student_exam_scope import zero_model_like
from rosclaw_soccer.rsi.reference_student_execution import own_model
from rosclaw_soccer.sim.contracts import hash_json


def require_zero_cohort(cfg, declaration, complete, audit):
    info = profile_info(ZERO)
    scope = positive_zero_scope(cfg["behavior_model_hash"])
    if (
        cfg["exam_profile"] != ZERO
        or declaration.get("schema") != info["declaration_schema"]
        or complete.get("schema") != info["complete_schema"]
        or audit.get("schema") != info["readback_schema"]
        or any(declaration.get(k) != v for k, v in cfg.items())
        or hash_json(declaration["scope"]) != hash_json(scope)
        or complete["declaration_hash"] != declaration["report_hash"]
        or audit["complete_hash"] != complete["report_hash"]
        or len(complete["results"]) != 9
        or len(audit["rows"]) != 9
        or any(
            type(complete.get(k)) is not int or complete[k] != n
            for k, n in (
                ("new_native_bodies", 9),
                ("new_independent_cpu_bodies", 9),
                ("native_substeps", 27000),
                ("independent_cpu_substeps", 27000),
                ("new_optimizer_updates", 0),
            )
        )
        or any(
            type(audit.get(k)) is not int or audit[k] != n
            for k, n in (
                ("source_native_bodies", 9),
                ("causal_reference_frames_reconstructed", 2400),
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
                "matched_zero_all300_reference_increments_exactly_zero",
            )
        )
        or audit.get("fitted_student_policy_evaluated") is not False
        or audit.get("official_52_case_exam_completed") is not False
        or audit.get("successor_readiness_qualified") is not False
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
    ):
        raise ValueError("whole nine-body zero-only baseline and independent readback required")
    models = declaration["models"]
    if set(models) != {"off", "zero", "student"} or models["off"] is not None:
        raise ValueError("unchanged explicit three model identities required")
    for role in ("zero", "student"):
        model = own_model(models[role])
        if (
            model["world_hash"] != cfg["mesh_task_contact_world_hash"]
            or model["parent_model_hash"] != cfg["behavior_model_hash"]
        ):
            raise ValueError("unchanged parent and world required for matched zero controls")
    if models["zero"] != zero_model_like(models["student"], seed=2026101902):
        raise ValueError("zero head with same fitted normalization and original seed required")
    if len({r["receipt_hash"] for r in complete["results"]}) != 9:
        raise ValueError("nine original distinct physical receipts required")
    for j, r, a in zip(scope["jobs"], complete["results"], audit["rows"], strict=True):
        m = models[j["role"]]
        mh = None if m is None else m["model_hash"]
        if (
            any(r.get(k) != v for k, v in j.items())
            or r["candidate"] != declaration["candidate"]
            or a["job"] != j
            or a["receipt_hash"] != r["receipt_hash"]
            or a["outcome"] != r["outcome"]
            or a["model_hash"] != mh
            or r.get("reference_student_model_hash") != mh
        ):
            raise ValueError("whole ordered zero/OFF roles, identities and outcomes required")
    paired = assess_reference(
        ZERO,
        scope,
        complete["results"],
        teachers={int(k): v["outcome"] for k, v in declaration["reused_teachers"].items()},
        original_retention={},
    )
    if paired != complete["paired_results"] or paired != audit["paired_results"]:
        raise ValueError("whole original teacher-to-zero pairing must reconstruct")
    return paired


def zero_measured_readout(declaration, complete, replay, metrics):
    if (
        replay.get("schema") != "soccer.rsi.fixed_control_contact_replay_complete.v1"
        or metrics.get("schema") != "soccer.rsi.contact_exit_metrics_readback.v1"
        or any(
            type(replay.get(k)) is not int or replay[k] != 27000
            for k in ("integrated_substeps", "attempted_step_calls")
        )
        or any(
            len(x) != 9
            for x in (
                declaration["scope"]["jobs"],
                complete["results"],
                replay.get("rows", []),
                metrics.get("rows", []),
            )
        )
    ):
        raise ValueError("all nine original zero/OFF recorded-control 27000-step replays required")
    index = {r["source_receipt_hash"]: r for r in metrics["rows"]}
    if len(index) != 9 or set(index) != {r["receipt_hash"] for r in complete["results"]}:
        raise ValueError("all nine original receipt memberships required")
    rows, nonfoot = [], 0
    for j, r, p in zip(
        declaration["scope"]["jobs"], complete["results"], replay["rows"], strict=True
    ):
        dense = index[r["receipt_hash"]]
        if (
            p["case_id"] != j["job"].replace("-", "_")
            or dense["case_id"] != p["case_id"]
            or p["source_receipt_hash"] != r["receipt_hash"]
            or p["original_outcome"] != r["outcome"]
            or dense["original_outcome"] != r["outcome"]
        ):
            raise ValueError("original zero-control case order and outcomes required")
        m = dense["metrics"]
        for key in ("maximum_linear_newton_residual_n", "total_robot_downward_impulse_n_s"):
            if type(m.get(key)) not in (int, float) or not math.isfinite(m[key]) or m[key] < 0:
                raise ValueError("finite original zero-control Newton/contact metrics required")
        if m["maximum_linear_newton_residual_n"] > 1e-7:
            raise ValueError("original Newton threshold cannot be relaxed")
        if j["role"] == "zero":
            nonfoot += m["first_nonfoot_robot_contact_substep"] is not None
        rows.append(
            dict(
                job=j,
                source_receipt_hash=r["receipt_hash"],
                original_outcome=r["outcome"],
                metrics=m,
            )
        )
    return dict(
        rows=rows,
        paired_results=complete["paired_results"],
        research_decision="MATCHED_ZERO_CONTROLS_CLOSED_NOT_LEARNING_REQUIRES_STUDENT_COMPARISON",
        zero_endpoints_with_nonfoot_over1n=nonfoot,
        distinct_derived_course_count=8,
        source_scene_clusters=4,
        fitted_student_policy_evaluated=False,
        original_task_labels_unchanged=True,
        diagnostic_is_not_a_new_exam_gate=True,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        neural_training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
