"""Fixed public receding-ball research scope; no global teacher or promotion."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.rsi.public_physics_option_planning import (
    PERTURBATIONS,
    PublicPlanningContext,
    validate_context_source,
)
from rosclaw_soccer.sim.contracts import hash_json

SOURCE_GROUPS = (76, 77, 108, 109, 140, 141)
CONTEXT_INDICES = (6, 7, 0, 1, 2, 3)


def receding_contexts(plan: PublicFirstTouchTrainingPlan) -> tuple[PublicPlanningContext, ...]:
    if (
        type(plan) is not PublicFirstTouchTrainingPlan
        or plan.seeds != tuple(range(20261060, 20261070))
        or plan.stream != 18
    ):
        raise ValueError("original complete public source plan required")
    draws = plan.draws()
    contexts = []
    for index, group in zip(CONTEXT_INDICES, SOURCE_GROUPS, strict=True):
        draw = draws[group]
        dx, sy, sv = PERTURBATIONS[draw.sample_index]
        context = PublicPlanningContext(
            index,
            group,
            draw.context_id,
            draw.seed,
            draw.lane,
            draw.sampling_seed,
            (draw.course[0] + dx, draw.course[1] * sy, draw.course[2] * sv),
            draw.sample_index,
        )
        validate_context_source(context, plan, group=group)
        if not (2.6 <= context.course[0] <= 2.8 and 0.25 <= context.course[2] <= 0.7):
            raise ValueError("whole scope must remain high-x positive-velocity public domain")
        contexts.append(context)
    if len({c.context_hash for c in contexts}) != 6:
        raise ValueError("six distinct course identities required")
    return tuple(contexts)


def receding_jobs() -> list[dict[str, Any]]:
    indices = dict(zip(SOURCE_GROUPS, CONTEXT_INDICES, strict=True))
    return [
        dict(
            job=f"{role}-receding-g{group}",
            role=role,
            pair_id=f"receding-g{group}",
            group=group,
            context_index=indices[group],
            option_index=2 if role == "teacher" else 0,
        )
        for group, role in (
            (77, "teacher"),  # Reproduce the archived zero-layer success first.
            (76, "teacher"),
            *((g, role) for g in SOURCE_GROUPS[2:] for role in ("control", "teacher")),
        )
    ]


def reproduce_zero_control(
    trace: dict[str, NDArray[np.generic]], old_trace: dict[str, NDArray[np.generic]]
) -> dict[str, Any]:
    """Compare *every* old non-cerebellar array, not just qpos or the task score.

    The caller authenticates original zero outputs, complete studies, receipts,
    models and source. Here no data is repaired and no numeric tolerance added.
    """
    old_public = {
        k: v
        for k, v in old_trace.items()
        if not k.startswith("cerebellar_")
        and k
        not in ("requested_cerebellar_torque_offset_nm", "applied_cerebellar_torque_offset_nm")
    }
    if (
        not old_public
        or set(trace) != set(old_public)
        or any(
            a.shape != old_public[k].shape
            or a.dtype != old_public[k].dtype
            or a.ndim == 0
            or a.shape[0] != 300
            or a.tobytes() != old_public[k].tobytes()
            for k, a in trace.items()
        )
    ):
        raise ValueError("all original300-frame non-cerebellar arrays must reproduce byte-exactly")
    return dict(
        schema="soccer.rsi.archived_zero_control_numeric_reproduction.v1",
        original_public_array_names=sorted(old_public),
        compared_frames=300,
        whole_original_arrays_bit_equal=True,
        added_cerebellar_layer_excluded_not_called=True,
        new_policy_learning_claimed=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )


def scope_contract(plan: PublicFirstTouchTrainingPlan) -> dict[str, Any]:
    contexts = receding_contexts(plan)
    result = dict(
        schema="soccer.rsi.receding_catalog2_scope.v1",
        source_plan_hash=plan.contract()["plan_hash"],
        contexts=[
            dict(context_hash=c.context_hash, course=list(c.course), source_group=c.source_group)
            for c in contexts
        ],
        jobs=receding_jobs(),
        new_native_bodies=10,
        independent_cpu_bodies=10,
        combined_body_substeps=60000,
        independent_courses=6,
        reused_none_controls=2,
        archived_expert_reproduction_required=True,
        all_four_new_none_controls_required=True,
        single_high_x_receding_stratum_only=True,
        fresh_exam=False,
        new_optimizer_updates=0,
        teacher_qualified=False,
        training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(result, scope_hash=hash_json(result))
