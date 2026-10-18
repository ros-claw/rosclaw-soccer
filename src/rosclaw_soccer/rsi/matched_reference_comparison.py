"""Whole same-approach zero/teacher/student comparison, not admission or fitting.

Callers authenticate both original physical cohorts, dense measurements and
process/source/file ownership. All public failures and the held-out source are
reported, but only predeclared training sources can motivate more supervision.
"""

from collections.abc import Callable
from typing import Any, cast

from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs
from rosclaw_soccer.rsi.positive_lateral_receding_scope import build_scope
from rosclaw_soccer.rsi.reference_exam_profiles import STUDENT, ZERO
from rosclaw_soccer.rsi.reference_student_exam_diagnostics import require_cohort as _require_cohort
from rosclaw_soccer.sim.contracts import hash_json

VALIDATION_SEED = 20261096
require_cohort = cast(Callable[..., dict[str, Any]], _require_cohort)


def compare_matched_references(zero: dict[str, Any], student: dict[str, Any]) -> dict[str, Any]:
    """Each cohort has config/declaration/complete/audit/measured, all whole.

    This is a research decision aid. A candidate with unsafe or dense nonfoot
    contacts is rejected regardless of its aggregate ball-outcome gains. A
    recommendation for new expert research never executes optimization.
    """
    for c in (zero, student):
        require_cohort(c["config"], c["declaration"], c["complete"], c["audit"])
    zc, sc = zero["config"], student["config"]
    zd, sd = zero["declaration"], student["declaration"]
    if (
        zc["exam_profile"] != ZERO
        or sc["exam_profile"] != STUDENT
        or any(zc[k] != sc[k] for k in ("behavior_model_hash", "mesh_task_contact_world_hash"))
        or zd["candidate"] != sd["candidate"]
        or zd["models"] != sd["models"]
        or zd["reused_teachers"] != sd["reused_teachers"]
        or hash_json(zd["scope"]["source_plans"]["positive19"])
        != hash_json(sd["scope"]["source_plans"]["positive19"])
    ):
        raise ValueError(
            "same parent/world/models/normalization/approach and original teachers required"
        )
    scope = build_scope(zc["behavior_model_hash"])
    groups = [c["source_group"] for c in scope["contexts"]]
    teachers = {g: sd["reused_teachers"][str(g)]["outcome"] for g in groups}
    selected, dense_by_role = {}, {}
    for role, c in (("zero", zero), ("student", student)):
        receipts = c["complete"]["results"]
        dense = c["measured"]["rows"]
        if len(dense) != len(receipts):
            raise ValueError(
                "whole original dense cohort required, not selected positive successes"
            )
        for job, r, d in zip(c["declaration"]["scope"]["jobs"], receipts, dense, strict=True):
            if (
                d["job"] != job
                or d["source_receipt_hash"] != r["receipt_hash"]
                or d["original_outcome"] != r["outcome"]
            ):
                raise ValueError("all ordered original dense receipts/outcomes required")
        chosen = [r for r in receipts if r["role"] == role and r["plan"] == "positive19"]
        if [r["group"] for r in chosen] != groups:
            raise ValueError("all eight same-approach positive courses required")
        selected[role] = {r["group"]: r["outcome"] for r in chosen}
        dense_by_role[role] = {
            d["job"]["group"]: d["metrics"]["first_nonfoot_robot_contact_substep"] is not None
            for d in dense
            if d["job"]["role"] == role and d["job"]["plan"] == "positive19"
        }
    comparisons = dict(
        zero_vs_teacher=assess_pairs(selected["zero"], teachers),
        zero_vs_student=assess_pairs(selected["zero"], selected["student"]),
        teacher_vs_student=assess_pairs(teachers, selected["student"]),
    )
    source_rows = []
    for seed in dict.fromkeys(c["source_seed"] for c in scope["contexts"]):
        gs = [c["source_group"] for c in scope["contexts"] if c["source_seed"] == seed]
        zs, ts, ss = (
            {g: m[g] for g in gs} for m in (selected["zero"], teachers, selected["student"])
        )
        source_rows.append(
            dict(
                source_seed=seed,
                partition="validation" if seed == VALIDATION_SEED else "train",
                groups=gs,
                zero_vs_teacher=assess_pairs(zs, ts),
                zero_vs_student=assess_pairs(zs, ss),
                zero_nonfoot_endpoints=sum(dense_by_role["zero"][g] for g in gs),
                student_nonfoot_endpoints=sum(dense_by_role["student"][g] for g in gs),
            )
        )
    training = [r for r in source_rows if r["partition"] == "train"]
    expert_support = [
        r["source_seed"]
        for r in training
        if r["zero_vs_teacher"]["net_success_gain"] > 0
        and r["zero_vs_teacher"]["newly_unsafe_pairs"] == 0
    ]
    train_net = sum(r["zero_vs_teacher"]["net_success_gain"] for r in training)
    expert_motivated = len(expert_support) >= 2 and train_net > 0
    retention = student["complete"]["paired_results"]["retention_zero_vs_student"]
    legacy = student["complete"]["paired_results"]["retention_legacy_vs_student"]
    student_unsafe = sum(
        not r["outcome"]["safe"] for r in student["complete"]["results"] if r["role"] == "student"
    )
    student_nonfoot = sum(
        r["metrics"]["first_nonfoot_robot_contact_substep"] is not None
        for r in student["measured"]["rows"]
        if r["job"]["role"] == "student"
    )
    # This family choice deliberately does not use validation success/failure.
    if expert_motivated:
        next_research = "BOUNDED_SAME_STATE_EXPERT_SUPERVISION_RESEARCH_HAS_TRAIN_SOURCE_SUPPORT"
    else:
        next_research = "IMPROVE_EXPERT_OR_APPROACH_BEFORE_MORE_SAME_TEACHER_IMITATION"
    if student_unsafe or student_nonfoot:
        disposition = "REJECT_UNSAFE_OR_NONFOOT_STUDENT"
    elif retention["regressed_pairs"]:
        disposition = "REJECT_RETENTION_REGRESSION"
    elif legacy["regressed_pairs"]:
        disposition = "REVIEW_APPROACH_APPLICABILITY_LEGACY_RETENTION_REGRESSION"
    elif comparisons["zero_vs_student"]["net_success_gain"] <= 0:
        disposition = "NO_POSITIVE_NET_LEARNED_REFERENCE_GAIN_VERSUS_MATCHED_NAV"
    else:
        disposition = "LOCAL_MATCHED_REFERENCE_GAIN_REQUIRES_THREE_SEEDS_FRESH_AND_CHAIN"
    return dict(
        schema="soccer.rsi.matched_reference_three_way.v1",
        comparisons=comparisons,
        source_rows=source_rows,
        expert_advantage_training_sources=expert_support,
        teacher_minus_zero_train_net=train_net,
        next_research=next_research,
        student_disposition=disposition,
        student_unsafe_endpoints=student_unsafe,
        student_nonfoot_endpoints=student_nonfoot,
        retention_zero_vs_student=retention,
        retention_legacy_vs_student=legacy,
        validation_used_for_research_family_choice=False,
        source_cluster_count=4,
        positive_derived_courses=8,
        independent_training_seeds_completed=1,
        counterfactual_query_success_verified=False,
        report_only=True,
        optimizer_updates=0,
        new_physical_steps=0,
        new_foundation_inferences=0,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        neural_training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
        fresh_exam=False,
    )
