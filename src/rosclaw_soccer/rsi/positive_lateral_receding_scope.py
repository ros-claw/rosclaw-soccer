"""Bounded metadata-only curriculum for a prospective local physics teacher.

Existing C6/C7 positives share one source scene. This declaration obtains four
different source seeds using the ORIGINAL public generator/lane rotation. No
task results, private exams, optimizer or physical execution are accessed.
"""

from typing import Any

from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from rosclaw_soccer.rsi.public_course_cluster_accounting import cluster_contract
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.public_physics_option_planning import (
    PERTURBATIONS,
    PublicPlanningContext,
    validate_context_source,
)
from rosclaw_soccer.sim.contracts import hash_json

FIRST_SEED = 20261070
MAX_EXAMINED_SEEDS = 2048
SOURCE_POSITIONS = (0, 1, 4, 5)


def _in_domain(course: tuple[float, float, float]) -> bool:
    x, y, vx = course
    return 2.60 <= x <= 2.72 and 0.08 <= y <= 0.16 and 0.25 <= vx <= 0.60


def build_scope(behavior_model_hash: str) -> dict[str, Any]:
    """First four metadata matches in a fixed bounded search, no result filter.

    Seed order determines the old balanced generator's lane. Keep that mapping
    rather than relabel an inconvenient lane or clip an out-of-domain course.
    Examined metadata and rejections remain part of the declaration.
    """
    selected: list[int] = []
    registry: list[int] = []
    examined: list[dict[str, Any]] = []
    for seed in range(FIRST_SEED, FIRST_SEED + MAX_EXAMINED_SEEDS):
        position = len(registry)
        lane = 9 + 2 * (position % 4)
        base = sample_training_courses(seed)[lane]
        derived = [(base[0] + dx, base[1] * sy, base[2] * sv) for dx, sy, sv in PERTURBATIONS]
        eligible = all(_in_domain(course) for course in derived)
        filler = position in (2, 3)
        examined.append(
            dict(
                seed=seed,
                lane=lane,
                base_course=list(base),
                derived_courses=[list(c) for c in derived],
                metadata_eligible=eligible,
                source_registry_position=position,
                source_registry_filler_not_a_declared_physical_trial=filler,
            )
        )
        if filler:
            # These lanes lie outside the prospective local x domain. Keep
            # their original source registration; do not relabel their lane or
            # pretend a full source-plan bank was collected. Only positions
            # 0,1,4,5 below are declared as the prospective physical study.
            registry.append(seed)
            continue
        if eligible:
            selected.append(seed)
            registry.append(seed)
            if len(selected) == 4:
                break
    if len(selected) != 4:
        raise ValueError(
            "bounded metadata pool lacks four scene clusters; no implicit domain expansion"
        )
    plan = PublicFirstTouchTrainingPlan(tuple(registry), behavior_model_hash, 19)
    draws = plan.draws()
    contexts = []
    for index in range(8):
        cluster_index, perturbation = divmod(index, 2)
        position = SOURCE_POSITIONS[cluster_index]
        group = position * 16 + 12 + perturbation
        draw = draws[group]
        if draw.lane != 9 + 2 * (position % 4) or draw.seed != selected[cluster_index]:
            raise ValueError("original balanced source lane rotation must remain exact")
        dx, sy, sv = PERTURBATIONS[perturbation]
        course = (draw.course[0] + dx, draw.course[1] * sy, draw.course[2] * sv)
        context = PublicPlanningContext(
            index,
            group,
            draw.context_id,
            draw.seed,
            draw.lane,
            draw.sampling_seed,
            course,
            perturbation,
        )
        validate_context_source(context, plan, group=group)
        if not _in_domain(context.course):
            raise ValueError("whole local curriculum must remain inside its original fixed domain")
        contexts.append(context)
    context_tuple = tuple(contexts)
    cluster = cluster_contract(plan, context_tuple)
    if cluster["source_context_clusters"] != 4 or cluster["distinct_source_seeds"] != 4:
        raise ValueError("eight perturbations cannot impersonate four distinct source seeds")
    value = dict(
        schema="soccer.rsi.positive_lateral_receding_scope.v1",
        source_plan=plan.contract(),
        source_seed_search=dict(
            first_seed=FIRST_SEED,
            maximum_examined_seeds=MAX_EXAMINED_SEEDS,
            examined_metadata=examined,
            selected_seeds=selected,
            source_registry_seeds=registry,
            target_source_positions=list(SOURCE_POSITIONS),
            selection=(
                "first metadata match at positions0/1/4/5; preserve lane13/15 registry fillers"
            ),
        ),
        domain=dict(x_m=[2.60, 2.72], y_m=[0.08, 0.16], vx_m_s=[0.25, 0.60]),
        contexts=[
            dict(
                context_hash=c.context_hash,
                source_group=c.source_group,
                source_context_id=c.source_context_id,
                source_seed=c.source_seed,
                source_lane=c.source_lane,
                context_index=c.context_index,
                sampling_seed=c.sampling_seed,
                perturbation_index=c.perturbation_index,
                course=list(c.course),
            )
            for c in contexts
        ],
        cluster_accounting=cluster,
        proposed_jobs=[
            dict(
                job=f"{role}-positive-local-c{c.context_index}",
                role=role,
                pair_id=f"positive-local-{c.context_hash}",
                group=c.source_group,
                context_hash=c.context_hash,
                context_index=c.context_index,
                option_index=0 if role == "control" else 2,
            )
            for c in contexts
            for role in ("control", "teacher")
        ],
        proposed_new_native_bodies=16,
        proposed_independent_cpu_bodies=16,
        proposed_combined_body_substeps=96000,
        old_c6_c7_reproduction_and_contacts_must_be_authenticated_and_reused=True,
        original_catalog2_parameters_unchanged=True,
        proposed_generated_seed_episode_binding=True,
        original_receding_full_audit_and_contact_diagnosis_required_before_execution=True,
        scoped_development_only=True,
        fresh_exam=False,
        source_plan_is_metadata_registry_not_a_full_bank_collection_request=True,
        current_results_for_new_courses_accessed=False,
        global_ancestry_separation_proven=False,
        all_future_success_failure_and_censored_outcomes_required=True,
        execution_authorized=False,
        dataset_admission_authorized=False,
        neural_training_authorized=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(value, scope_hash=hash_json(value))
