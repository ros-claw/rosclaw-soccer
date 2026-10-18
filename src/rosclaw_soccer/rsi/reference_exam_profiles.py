"""Two explicit bounded SIM exam profiles; no arbitrary count overrides."""

from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs
from rosclaw_soccer.rsi.positive_lateral_receding_scope import build_scope
from rosclaw_soccer.rsi.reference_student_exam_scope import assess_exam, exam_scope
from rosclaw_soccer.sim.contracts import hash_json

STUDENT = "reference-own-body-thirteen-v1"
ZERO = "reference-positive-zero-nine-v1"


def profile_info(profile):
    if profile == STUDENT:
        return dict(
            bodies=13,
            steps=39000,
            reference_frames=3600,
            courses=10,
            clusters=6,
            declaration_schema="soccer.rsi.reference_student_exam_declaration.v1",
            complete_schema="soccer.rsi.reference_student_exam_complete.v1",
            readback_schema="soccer.rsi.reference_student_exam_terminal_readback.v1",
            replay_profile="reference-student-thirteen",
            preflight="THIRTEEN_FIXED_OWN_BODY_EXAMS_ZERO_FULL_MODELS_OR_PHYSICS_ALLOCATIONS",
        )
    if profile == ZERO:
        return dict(
            bodies=9,
            steps=27000,
            reference_frames=2400,
            courses=8,
            clusters=4,
            declaration_schema="soccer.rsi.reference_positive_zero_declaration.v1",
            complete_schema="soccer.rsi.reference_positive_zero_complete.v1",
            readback_schema="soccer.rsi.reference_positive_zero_terminal_readback.v1",
            replay_profile="reference-positive-zero-nine",
            preflight="NINE_MATCHED_ZERO_BASELINES_ZERO_FULL_MODELS_OR_PHYSICS_ALLOCATIONS",
        )
    raise ValueError("explicit fixed thirteen-student or nine-zero SIM profile required")


def positive_zero_scope(parent_model_hash):
    positive = build_scope(parent_model_hash)
    jobs = [
        dict(
            job="reference-zero-off-positive-c0",
            role="off",
            plan="positive19",
            group=12,
            comparison="original_positive_teacher_full33_parity",
            use_generated_binding=True,
        )
    ]
    jobs.extend(
        dict(
            job=f"reference-zero-positive-c{c['context_index']}",
            role="zero",
            plan="positive19",
            group=c["source_group"],
            comparison="matched_navigation_only",
            use_generated_binding=True,
        )
        for c in positive["contexts"]
    )
    value = dict(
        schema="soccer.rsi.reference_positive_zero_scope.v1",
        source_plans=dict(positive19=positive["source_plan"]),
        positive_scope_hash=positive["scope_hash"],
        jobs=jobs,
        new_native_bodies=9,
        new_independent_cpu_bodies=9,
        combined_body_substeps=54000,
        new_optimizer_updates=0,
        distinct_derived_course_count=8,
        source_scene_clusters=4,
        all_positive_teacher_failures_retained=True,
        zero_is_navigation_not_original_IK_teacher=True,
        off_must_reproduce_all33_arrays_before_zero_bodies=True,
        fitted_student_policy_executed=False,
        official_52_case_exam_completed=False,
        fresh_exam=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(value, scope_hash=hash_json(value))


def reference_scope(profile, parent_model_hash):
    profile_info(profile)
    return (
        positive_zero_scope(parent_model_hash) if profile == ZERO else exam_scope(parent_model_hash)
    )


def assess_reference(profile, scope, receipts, *, teachers, original_retention):
    if profile == STUDENT:
        return assess_exam(
            scope, receipts, teachers=teachers, original_retention=original_retention
        )
    if profile != ZERO:
        profile_info(profile)
    expected = positive_zero_scope(scope["source_plans"]["positive19"]["behavior_model_hash"])
    groups = [j["group"] for j in expected["jobs"][1:]]
    if (
        hash_json(scope) != hash_json(expected)
        or len(receipts) != 9
        or set(teachers) != set(groups)
        or any(
            r.get(k) != v
            for j, r in zip(expected["jobs"], receipts, strict=True)
            for k, v in j.items()
        )
    ):
        raise ValueError(
            "all nine matched zero/OFF bodies and all eight original teacher outcomes required"
        )
    paired = assess_pairs(
        {g: teachers[g] for g in groups},
        {g: r["outcome"] for g, r in zip(groups, receipts[1:], strict=True)},
    )
    return dict(
        positive_teacher_vs_zero=paired,
        fitted_student_policy_evaluated=False,
        zero_is_navigation_with_same_approach_and_parent=True,
        distinct_derived_course_count=8,
        source_scene_clusters=4,
        official_52_case_exam_completed=False,
        teacher_qualified=False,
        fresh_exam=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )


def require_zero_navigation_trace(trace):
    import numpy as np

    from rosclaw_soccer.rsi.contact_reference_labels import reference_labels

    for name in ("requested_reference_increment_rad", "applied_reference_increment_rad"):
        a = trace["reference_student_" + name]
        if a.shape != (300, 1, 12) or a.dtype != np.float64 or np.any(a != 0):
            raise ValueError(
                "every original 300-frame requested and applied reference must be zero"
            )
    navigation = reference_labels(trace)["navigation_joint_target_rad"]
    if trace["pre_motor_joint_target_rad"].tobytes() != navigation.tobytes():
        raise ValueError("whole zero-control pre-parent target must be navigation byte-exactly")
