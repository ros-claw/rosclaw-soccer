"""Whole prospective cohort contract before fixed-control diagnostics.

Caller authenticates original file bytes and seals. This pure module neither
executes physics nor admits a teacher, dataset, model or hardware action.
"""

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
