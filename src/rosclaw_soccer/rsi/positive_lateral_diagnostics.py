"""Whole prospective cohort contract before fixed-control diagnostics.

Caller authenticates original file bytes and seals. This pure module neither
executes physics nor admits a teacher, dataset, model or hardware action.
"""

import math

from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs
from rosclaw_soccer.rsi.positive_lateral_episode_inputs import prospective_split
from rosclaw_soccer.rsi.positive_lateral_receding_scope import build_scope
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_positive_lateral_physical_pilot import require_scope


def require_cohort(config, declaration, complete, audit):
    require_scope(config)
    scope = build_scope(config["behavior_model_hash"])
    jobs = scope["proposed_jobs"]
    split = prospective_split(scope)
    if (
        declaration.get("schema") != "soccer.rsi.positive_lateral_physical_declaration.v1"
        or complete.get("schema") != "soccer.rsi.positive_lateral_physical_complete.v1"
        or audit.get("schema") != "soccer.rsi.positive_lateral_physical_terminal_readback.v1"
        or complete.get("declaration_hash") != declaration["report_hash"]
        or audit.get("complete_hash") != complete["report_hash"]
        or declaration.get("jobs") != jobs
        or hash_json(declaration.get("source_plan")) != hash_json(scope["source_plan"])
        or declaration.get("scope_hash") != scope["scope_hash"]
        or any(doc.get("prospective_split") != split for doc in (declaration, complete, audit))
        or any(
            doc.get("source_plan_hash") != scope["source_plan"]["plan_hash"]
            for doc in (complete, audit)
        )
        or any(
            doc.get("course_cluster_contract") != scope["cluster_accounting"]
            for doc in (declaration, audit)
        )
        or any(
            type(doc.get(k)) is not int or doc[k] != n
            for doc in (declaration, complete, audit)
            for k, n in (("distinct_derived_course_count", 8), ("source_scene_clusters", 4))
        )
        or any(
            type(complete.get(k)) is not int or complete[k] != n
            for k, n in (
                ("new_native_bodies", 16),
                ("new_independent_cpu_bodies", 16),
                ("native_substeps", 48000),
                ("independent_cpu_substeps", 48000),
                ("new_optimizer_updates", 0),
            )
        )
        or len(complete.get("results", [])) != 16
        or len(audit.get("rows", [])) != 16
        or type(audit.get("source_native_bodies")) is not int
        or audit["source_native_bodies"] != 16
        or any(
            audit.get(k) is not True
            for k in (
                "whole_input_and_witness_files_verified",
                "shared_payload_logical_seals_rebuilt",
                "prior_independent_replays_authenticated",
                "generated_seed_binding_in_all_bodies",
            )
        )
        or any(
            doc.get(k) is not False
            for doc in (complete, audit)
            for k in (
                "teacher_qualified",
                "dataset_admission_authorized",
                "neural_training_authorized",
                "fresh_exam",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or any(
            doc.get("old_global_failures_retained_not_relabelled") is not True
            for doc in (complete, audit)
        )
        or len(declaration.get("options", [])) != 8
        or declaration["options"][0] is not None
    ):
        raise ValueError("whole original sixteen-body/four-cluster development audit required")
    controls, evaluations = {}, {}
    for job, receipt, row in zip(jobs, complete["results"], audit["rows"], strict=True):
        if (
            any(receipt.get(k) != v for k, v in job.items())
            or any(row[k] != receipt[k] for k in ("job", "receipt_hash", "outcome"))
            or row["world_hash"] != config["mesh_task_contact_world_hash"]
            or receipt["candidate"] != declaration["options"][job["option_index"]]
        ):
            raise ValueError("all original paired endpoint identities and failures required")
        (controls if job["role"] == "control" else evaluations)[job["pair_id"]] = receipt["outcome"]
    paired = assess_pairs(controls, evaluations)
    if any(doc.get("paired_development") != paired for doc in (complete, audit)):
        raise ValueError("all eight paired physical outcomes must reconstruct")
    return paired


def measured_contact_readout(declaration, complete, replay, metrics):
    """Inspect every reused control trace without manufacturing training authority."""
    jobs = declaration["jobs"]
    if (
        replay.get("schema") != "soccer.rsi.fixed_control_contact_replay_complete.v1"
        or replay.get("integrated_substeps") != 48000
        or replay.get("attempted_step_calls") != 48000
        or metrics.get("schema") != "soccer.rsi.contact_exit_metrics_readback.v1"
        or any(
            len(rows) != 16
            for rows in (jobs, complete["results"], replay.get("rows", []), metrics.get("rows", []))
        )
    ):
        raise ValueError("whole sixteen-endpoint 48000-step contact diagnostic required")
    indexed = {r["source_receipt_hash"]: r for r in metrics["rows"]}
    if len(indexed) != 16 or set(indexed) != {r["receipt_hash"] for r in complete["results"]}:
        raise ValueError("distinct whole original receipt membership required")
    scope_rows = {r["group"]: r for r in declaration["prospective_split"]["rows"]}
    selected, successful, train_seeds, validation_seeds = [], [], set(), set()
    unsafe, nonfoot = 0, 0
    for job, receipt, row in zip(jobs, complete["results"], replay["rows"], strict=True):
        dense = indexed[receipt["receipt_hash"]]
        if (
            row["case_id"] != job["job"].replace("-", "_")
            or dense["case_id"] != row["case_id"]
            or row["source_receipt_hash"] != receipt["receipt_hash"]
            or row["original_outcome"] != receipt["outcome"]
            or dense["original_outcome"] != receipt["outcome"]
        ):
            raise ValueError("all replay outcomes and paired course identities must remain exact")
        m = dense["metrics"]
        for key in ("total_robot_downward_impulse_n_s", "maximum_linear_newton_residual_n"):
            if type(m[key]) not in (float, int) or not math.isfinite(m[key]) or m[key] < 0:
                raise ValueError("finite nonnegative physical contact audit required")
        if m["maximum_linear_newton_residual_n"] > 1e-7:
            raise ValueError("Newton consistency threshold cannot be relaxed")
        split = scope_rows[job["group"]]
        selected.append(
            dict(
                job=job,
                partition=split["partition"],
                source_seed=split["source_seed"],
                dense_row=dense,
            )
        )
        if job["role"] == "teacher":
            unsafe += not receipt["outcome"]["safe"]
            nonfoot += m["first_nonfoot_robot_contact_substep"] is not None
            if (
                receipt["outcome"]["task_pass"]
                and receipt["outcome"]["safe"]
                and m["first_nonfoot_robot_contact_substep"] is None
            ):
                successful.append(receipt["receipt_hash"])
                (train_seeds if split["partition"] == "train" else validation_seeds).add(
                    split["source_seed"]
                )
    paired = complete["paired_development"]
    if unsafe or nonfoot or paired["regressed_pairs"] or paired["newly_unsafe_pairs"]:
        decision = "REJECT_DECLARED_TEACHER_RETAIN_ALL_FAILURES"
    elif paired["net_success_gain"] <= 0 or len(train_seeds) < 2 or not validation_seeds:
        decision = "INSUFFICIENT_CROSS_SOURCE_REFERENCE_EVIDENCE_REQUIRES_MECHANISM_REVIEW"
    else:
        decision = "LOCAL_REFERENCE_EVIDENCE_REQUIRES_LABEL_BUILD_AND_OWN_STUDENT_CLOSED_LOOP"
    return dict(
        rows=selected,
        research_decision=decision,
        successful_candidate_receipt_hashes=successful,
        successful_train_source_seeds=sorted(train_seeds),
        successful_validation_source_seeds=sorted(validation_seeds),
        unsafe_candidate_endpoints=unsafe,
        candidate_endpoints_with_nonfoot_over1n=nonfoot,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        neural_training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
        fresh_exam=False,
    )
