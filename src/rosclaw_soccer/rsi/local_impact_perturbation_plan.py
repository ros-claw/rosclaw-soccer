"""Outcome-independent LOCAL high-x rolling-away impact robustness pilot.

Three source-course clusters, two new courses per cluster and two heading
bindings per course are not twelve independent worlds. This declaration never
qualifies a teacher, opens Fresh, admits training data or alters an exam.
"""

from typing import Any

from rosclaw_soccer.rsi.first_touch_mechanism_expansion import validate_public_course
from rosclaw_soccer.sim.contracts import hash_json

# Fixed before any new outcome. No random rejection, clipping or replacement.
PERTURBATIONS = ((0.01, -0.015, 0.02), (-0.01, 0.015, -0.02))


def declare_local_impact_perturbations(
    courses: tuple[tuple[float, float, float], ...],
) -> dict[str, Any]:
    if type(courses) is not tuple or len(courses) != 6:
        raise ValueError("all six ordered original heading bindings required")
    for i, course in enumerate(courses):
        if validate_public_course(course) != 3:
            raise ValueError("local pilot covers only high-x positive rolling-away balls")
        if i % 2 and course != courses[i - 1]:
            raise ValueError("each OFF/ON heading pair must share its source course")
    if len(set(courses)) != 3:
        raise ValueError("three distinct source-course clusters required")
    jobs: list[dict[str, Any]] = []
    for i, course in enumerate(courses):
        for variant, offset in enumerate(PERTURBATIONS):
            changed = (
                float(course[0] + offset[0]),
                float(course[1] + offset[1]),
                float(course[2] + offset[2]),
            )
            if validate_public_course(changed) != 3:
                raise ValueError("entire perturbation declaration must retain its public stratum")
            jobs.append(
                dict(
                    source_index=i,
                    source_cluster=i // 2,
                    perturbation_index=variant,
                    course=list(changed),
                    paired_arms=["original_control", "prospective_early_impact"],
                    same_original_policy_seed_in_both_arms=True,
                )
            )
    unique = {tuple(job["course"]) for job in jobs}
    if len(unique) != 6 or unique & set(courses):
        raise ValueError("six distinct new physical courses required; no duplicate old courses")
    d = dict(
        schema="soccer.rsi.local_impact_perturbation_plan.v1",
        partition="CONSUMED_PUBLIC_DEVELOPMENT_LOCAL_PERTURBATION",
        original_courses=[list(c) for c in courses],
        perturbations=[list(v) for v in PERTURBATIONS],
        jobs=jobs,
        original_source_clusters=3,
        new_unique_physical_courses=6,
        paired_policy_course_contexts=12,
        new_rollouts=24,
        baseline_source_rollouts_reused=12,
        official_horizon_frames=300,
        post_contact_frames=60,
        new_native_step_budget=72000,
        forecast_step_budget=21600,
        combined_step_budget=93600,
        all_courses_preflighted=True,
        all_success_failure_censored_and_unsafe_records_required=True,
        optimizer_updates=0,
        private_fresh_accessed=False,
        global_four_strata_generalization_claimed=False,
        teacher_qualified=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return dict(d, report_hash=hash_json(d))
