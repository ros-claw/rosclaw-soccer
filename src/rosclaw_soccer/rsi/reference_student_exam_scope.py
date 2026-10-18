"""Fixed own-body student exam, distinct from numeric or teacher-state replay.

Eight prospective development courses plus two historically successful raw
courses. Reused teacher/original evidence is not a new trial. No Fresh claim,
control execution, source authentication, fitting or automatic promotion here.
"""

from rosclaw.growth.causal_residual_memory import initial_parameters

from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs
from rosclaw_soccer.rsi.positive_lateral_receding_scope import build_scope
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.reference_student_execution import make_model, own_model
from rosclaw_soccer.sim.contracts import hash_json


def exam_scope(parent_model_hash):
    positive = build_scope(parent_model_hash)
    original = PublicFirstTouchTrainingPlan(tuple(range(20261060, 20261070)), parent_model_hash, 18)
    jobs = [
        dict(
            job="reference-exam-off-positive-c0",
            role="off",
            plan="positive19",
            group=12,
            comparison="original_positive_teacher_full33_parity",
            use_generated_binding=True,
        )
    ]
    jobs.extend(
        dict(
            job=f"reference-exam-student-positive-c{c['context_index']}",
            role="student",
            plan="positive19",
            group=c["source_group"],
            comparison="positive_teacher",
            use_generated_binding=True,
        )
        for c in positive["contexts"]
    )
    jobs.extend(
        dict(
            job=f"reference-exam-{role}-retention-g{group}",
            role=role,
            plan="original18",
            group=group,
            comparison="retention_navigation_control",
            use_generated_binding=False,
        )
        for group in (4, 32)
        for role in ("zero", "student")
    )
    value = dict(
        schema="soccer.rsi.reference_student_own_body_exam_scope.v1",
        source_plans=dict(positive19=positive["source_plan"], original18=original.contract()),
        positive_scope_hash=positive["scope_hash"],
        jobs=jobs,
        new_native_bodies=13,
        new_independent_cpu_bodies=13,
        combined_body_substeps=78000,
        new_optimizer_updates=0,
        distinct_derived_course_count=10,
        source_scene_clusters=6,
        positive_development_courses=8,
        positive_source_scene_clusters=4,
        historical_retention_courses=2,
        historical_source_scene_clusters=2,
        original_52_case_exam_completed=False,
        fresh_exam=False,
        original_teacher_outcomes_all_reused_including_failures=True,
        off_case_must_reproduce_all33_arrays_before_any_student_body=True,
        zero_model_keeps_fitted_normalization_and_original_initialization=True,
        raw_legacy_comparison_changes_approach_not_pure_gru_ablation=True,
        raw_zero_vs_student_comparison_is_learned_reference_ablation=True,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    if len(jobs) != 13 or len({j["job"] for j in jobs}) != 13:
        raise ValueError("all thirteen fixed distinct own-body jobs required")
    return dict(value, scope_hash=hash_json(value))


def zero_model_like(student, *, seed):
    model = own_model(student)
    return make_model(
        initial_parameters(135, 12, hidden_dimension=64, seed=seed),
        mean=model["mean"],
        scale=model["scale"],
        world_hash=model["world_hash"],
        parent_model_hash=model["parent_model_hash"],
    )


def assess_exam(scope, receipts, *, teachers, original_retention):
    """Whole declared results only; the caller authenticates primitive labels."""
    if (
        hash_json(scope)
        != hash_json(exam_scope(scope["source_plans"]["positive19"]["behavior_model_hash"]))
        or len(receipts) != 13
        or set(teachers) != {12, 13, 28, 29, 76, 77, 92, 93}
        or set(original_retention) != {4, 32}
        or any(
            r.get(k) != v
            for j, r in zip(scope["jobs"], receipts, strict=True)
            for k, v in j.items()
        )
    ):
        raise ValueError(
            "whole fixed exam, all teacher failures and both legacy successes required"
        )
    positive, zero, retention = {}, {}, {}
    for r in receipts:
        if r["role"] == "student":
            (positive if r["plan"] == "positive19" else retention)[r["group"]] = r["outcome"]
        elif r["role"] == "zero":
            zero[r["group"]] = r["outcome"]
    if any(r["task_pass"] is not True for r in original_retention.values()):
        raise ValueError("both predeclared raw retention controls must be historical successes")
    return dict(
        positive_teacher_vs_student=assess_pairs(teachers, positive),
        retention_zero_vs_student=assess_pairs(zero, retention),
        retention_legacy_vs_student=assess_pairs(original_retention, retention),
        legacy_comparison_confounded_by_approach_change=True,
        zero_comparison_is_learned_reference_ablation=True,
        distinct_derived_course_count=10,
        source_scene_clusters=6,
        official_52_case_exam_completed=False,
        fresh_exam=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
