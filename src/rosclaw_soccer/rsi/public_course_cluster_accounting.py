"""Keep distinct perturbed courses separate from source-scene cluster counts.

Pure metadata accounting. Paired controllers, perturbations, replay bodies and
adjacent frames cannot inflate independent source-context sample counts. Even
different source clusters are not a proof of statistical independence.
"""

from collections import defaultdict
from typing import Any

from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.public_physics_option_planning import (
    PublicPlanningContext,
    validate_context_source,
)
from rosclaw_soccer.sim.contracts import hash_json


def cluster_contract(
    plan: PublicFirstTouchTrainingPlan, contexts: tuple[PublicPlanningContext, ...]
) -> dict[str, Any]:
    if type(contexts) is not tuple or not 1 <= len(contexts) <= 32:
        raise ValueError("bounded explicit derived public courses required")
    for context in contexts:
        if type(context) is not PublicPlanningContext:
            raise ValueError("typed original derived public context required")
        validate_context_source(context, plan, group=context.source_group)
    if len({c.context_hash for c in contexts}) != len(contexts) or len(
        {c.course for c in contexts}
    ) != len(contexts):
        raise ValueError(
            "deduplicate courses before counting; pairs and replays are not new courses"
        )
    groups: dict[tuple[int, int, int], list[PublicPlanningContext]] = defaultdict(list)
    for context in contexts:
        groups[(context.source_context_id, context.source_seed, context.source_lane)].append(
            context
        )
    clusters = [
        dict(
            source_context_id=key[0],
            source_seed=key[1],
            source_lane=key[2],
            derived_context_hashes=[c.context_hash for c in values],
            source_groups=[c.source_group for c in values],
            perturbation_indices=[c.perturbation_index for c in values],
        )
        for key, values in sorted(groups.items())
    ]
    value = dict(
        schema="soccer.rsi.public_derived_course_cluster_accounting.v1",
        source_plan_hash=plan.contract()["plan_hash"],
        distinct_derived_courses=len(contexts),
        source_context_clusters=len(groups),
        distinct_source_seeds=len({c.source_seed for c in contexts}),
        clusters=clusters,
        statistical_independence_proven=False,
        same_cluster_must_share_training_validation_split=True,
        uncertainty_unit="source context/seed/lane cluster, not frame, perturbation or controller",
        outcome_labels_read=False,
        new_physical_steps=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(value, report_hash=hash_json(value))
