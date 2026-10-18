"""Whole six-case diagnostic contracts; no physics, learning or admission."""

import math

from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs
from rosclaw_soccer.rsi.receding_teacher_scope import SOURCE_GROUPS
from rosclaw_soccer.rsi.surface_receding_evidence import surface_receding_jobs


def require_endpoints(config, declaration, complete, audit):
    jobs = surface_receding_jobs()
    if (
        config.get("mode") != "SIM_ONLY"
        or config.get("groups") != list(SOURCE_GROUPS)
        or config.get("selected_option_index") != 2
        or config.get("surface_profile") != "receding-six"
        or config.get("use_mesh_surface_swing") is not True
        or any(
            config.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "neural_training_authorized",
                "teacher_auto_admission",
                "suppress_postcontact_swing_release",
                "generated_seed_episode_binding",
            )
        )
        or any(
            type(config.get(k)) is not int or config[k] != 6
            for k in ("max_attempts", "new_physical_budget")
        )
        or declaration.get("schema") != "soccer.rsi.mesh_surface_receding_pilot_declaration.v1"
        or complete.get("schema") != "soccer.rsi.mesh_surface_receding_pilot_complete.v1"
        or audit.get("schema") != "soccer.rsi.mesh_surface_receding_terminal_readback.v1"
        or declaration.get("jobs") != jobs
        or complete.get("declaration_hash") != declaration["report_hash"]
        or audit.get("complete_hash") != complete["report_hash"]
        or any(
            type(complete.get(k)) is not int or complete[k] != n
            for k, n in (
                ("new_native_bodies", 6),
                ("native_substeps", 18000),
                ("independent_cpu_substeps", 18000),
                ("new_optimizer_updates", 0),
                ("paired_endpoint_count", 12),
            )
        )
        or any(
            type(doc.get(k)) is not int or doc[k] != n
            for doc in (declaration, complete, audit)
            for k, n in (("distinct_derived_course_count", 6), ("source_scene_clusters", 3))
        )
        or audit.get("source_native_bodies") != 6
        or any(
            audit.get(k) is not True
            for k in (
                "whole_input_and_witness_files_verified",
                "shared_payload_logical_seals_rebuilt",
                "prior_independent_replays_authenticated",
                "all_original_pre_activation_prefixes_bit_equal",
                "prior_disabled_noninterference_reused_not_rerun",
                "geometry_and_ik_reconstruction_uses_same_pinned_implementation",
            )
        )
        or any(
            doc.get(k) is not False
            for doc in (complete, audit)
            for k in ("teacher_qualified", "hardware_authorized", "promotion_authorized")
        )
        or audit.get("fresh_exam") is not False
        or audit.get("independent_ik_implementation") is not False
        or any(
            len(complete.get(k, [])) != 6
            for k in ("results", "reused_endpoints", "reused_original_none_endpoints")
        )
        or len(audit.get("rows", [])) != 6
        or any(
            complete[k] != declaration.get(k)
            for k in ("reused_endpoints", "reused_original_none_endpoints")
        )
        or declaration.get("course_cluster_contract") != audit.get("course_cluster_contract")
    ):
        raise ValueError("whole six-case material-point audit and three source clusters required")
    controls, none, evaluations = {}, {}, {}
    for job, receipt, row, baseline, original in zip(
        jobs,
        complete["results"],
        audit["rows"],
        complete["reused_endpoints"],
        complete["reused_original_none_endpoints"],
        strict=True,
    ):
        if (
            any(receipt.get(k) != v for k, v in job.items())
            or any(row[k] != receipt[k] for k in ("job", "receipt_hash", "outcome"))
            or row["world_hash"] != config["mesh_task_contact_world_hash"]
            or any(
                r["pair_id"] != job["pair_id"]
                or r["role"] != "control"
                or r["receipt"]["group"] != job["group"]
                for r in (baseline, original)
            )
            or original["receipt"]["candidate"] is not None
            or baseline["receipt"]["candidate"] != declaration["options"][2]
            or receipt["candidate"] != declaration["options"][2]
        ):
            raise ValueError("all original baseline and new endpoint identities/outcomes required")
        controls[job["pair_id"]] = baseline["receipt"]["outcome"]
        none[job["pair_id"]] = original["receipt"]["outcome"]
        evaluations[job["pair_id"]] = receipt["outcome"]
    paired, retention = assess_pairs(controls, evaluations), assess_pairs(none, evaluations)
    if (
        paired != complete["paired_development"]
        or paired != audit["paired_development"]
        or retention != complete["original_none_context_retention"]
        or retention != audit["original_none_context_retention"]
    ):
        raise ValueError("both complete paired outcome tables must reconstruct")
    return paired, retention


def scoped_metrics(complete, old_replays, replay, metrics):
    """28 endpoints = 10 derived courses/5 clusters; select the scoped 6x3."""
    if (
        [len(r.get("rows", [])) for r in old_replays] != [10, 12]
        or [r.get("integrated_substeps") for r in old_replays] != [30000, 36000]
        or replay.get("schema") != "soccer.rsi.fixed_control_contact_replay_complete.v1"
        or replay.get("integrated_substeps") != 18000
        or replay.get("attempted_step_calls") != 18000
        or len(replay.get("rows", [])) != 6
        or metrics.get("schema") != "soccer.rsi.contact_exit_metrics_readback.v1"
        or len(metrics.get("rows", [])) != 28
    ):
        raise ValueError("exact six new replays and two whole reused replay sets required")
    expected = {
        r["source_receipt_hash"]: r["original_outcome"] for old in old_replays for r in old["rows"]
    }
    if len(expected) != 22:
        raise ValueError("twenty-two distinct reused endpoints required")
    for job, receipt, row in zip(
        surface_receding_jobs(), complete["results"], replay["rows"], strict=True
    ):
        if (
            row["case_id"] != job["job"].replace("-", "_")
            or row["source_receipt_hash"] != receipt["receipt_hash"]
            or row["original_outcome"] != receipt["outcome"]
            or receipt["receipt_hash"] in expected
        ):
            raise ValueError("exact new material-point replay membership required")
        expected[receipt["receipt_hash"]] = receipt["outcome"]
    indexed = {r["source_receipt_hash"]: r for r in metrics["rows"]}
    if (
        len(indexed) != 28
        or indexed.keys() != expected.keys()
        or any(indexed[h]["original_outcome"] != outcome for h, outcome in expected.items())
    ):
        raise ValueError(
            "whole dense union must preserve all success/failure labels without duplication"
        )
    selected = []
    for job, current, baseline, original in zip(
        surface_receding_jobs(),
        complete["results"],
        complete["reused_endpoints"],
        complete["reused_original_none_endpoints"],
        strict=True,
    ):
        for family, receipt in (
            ("none", original["receipt"]),
            ("body_origin_catalog2", baseline["receipt"]),
            ("material_point_catalog2", current),
        ):
            row = indexed[receipt["receipt_hash"]]
            m = row["metrics"]
            for key in ("total_robot_downward_impulse_n_s", "maximum_linear_newton_residual_n"):
                value = m[key]
                if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                    raise ValueError("finite physical contact diagnostics required")
            if m["maximum_linear_newton_residual_n"] > 1e-7:
                raise ValueError("dense Newton consistency must remain verified")
            selected.append(dict(pair_id=job["pair_id"], family=family, dense_row=row))
    if len({r["dense_row"]["source_receipt_hash"] for r in selected}) != 18:
        raise ValueError("six courses x three distinct controller endpoints required")
    return selected


def research_decision(paired, selected):
    """Explain the mechanism evidence; never auto-admit a teacher or dataset."""
    current = [r["dense_row"] for r in selected if r["family"] == "material_point_catalog2"]
    if len(current) != 6:
        raise ValueError("all six current outcomes required before research decision")
    unsafe = sum(not r["original_outcome"]["safe"] for r in current)
    nonfoot = sum(r["metrics"]["first_nonfoot_robot_contact_substep"] is not None for r in current)
    if paired["regressed_pairs"] or paired["newly_unsafe_pairs"] or unsafe or nonfoot:
        decision = "REJECT_DECLARED_TEACHER_RETAIN_LOCAL_EVIDENCE_AND_FAILURES"
    elif paired["net_success_gain"] <= 0:
        decision = "NO_NET_GAIN_REQUIRES_MECHANISM_REVIEW_NOT_MORE_SAME_SETTING_TRIALS"
    else:
        decision = "MECHANISM_GAIN_REQUIRES_CROSS_SOURCE_REPLICATION_AND_STUDENT_CLOSED_LOOP"
    return dict(
        decision=decision,
        unsafe_current_endpoints=unsafe,
        current_endpoints_with_nonfoot_over1n=nonfoot,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        neural_training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
