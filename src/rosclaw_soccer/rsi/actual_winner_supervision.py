"""Select actual successful paths, including NAV zero-action stability anchors.

This is not the rejected all-successful-IK/counterfactual imitation objective.
Every label must come from the selected controller's OWN recorded state. No
cross-trajectory action transplant, fitting, admission or policy promotion.
Callers authenticate the full original cohorts, traces and dense contacts.
"""

from collections.abc import Callable
from typing import Any, cast

from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs
from rosclaw_soccer.rsi.positive_lateral_episode_inputs import (
    prospective_split as _prospective_split,
)
from rosclaw_soccer.rsi.positive_lateral_receding_scope import build_scope
from rosclaw_soccer.sim.contracts import hash_json

prospective_split = cast(Callable[[dict[str, Any]], dict[str, Any]], _prospective_split)


def select_actual_winner_paths(
    *,
    parent_model_hash: str,
    zeros: list[dict[str, Any]],
    teachers: list[dict[str, Any]],
    zero_nonfoot: dict[int, bool],
    teacher_nonfoot: dict[int, bool],
) -> dict[str, Any]:
    """One actual successful controller per course; prefer NAV if both pass.

    Zero is an observed successful action target, not absence of data. Failed
    controllers remain in the bank with no positive supervision. This function
    selects a proposed experimental dataset, never authorizes training.
    """
    scope = build_scope(parent_model_hash)
    split = prospective_split(scope)
    groups = [c["source_group"] for c in scope["contexts"]]
    if (
        len(zeros) != 8
        or len(teachers) != 8
        or [r.get("group") for r in zeros] != groups
        or [r.get("group") for r in teachers] != groups
        or any(r.get("role") != "zero" for r in zeros)
        or any(r.get("role") != "teacher" for r in teachers)
        or len({r["receipt_hash"] for r in zeros + teachers}) != 16
        or any(
            set(v) != set(groups) or any(type(x) is not bool for x in v.values())
            for v in (zero_nonfoot, teacher_nonfoot)
        )
    ):
        raise ValueError("whole distinct eight NAV/IK pairs and dense safety labels required")
    paired = assess_pairs(
        {r["group"]: r["outcome"] for r in zeros}, {r["group"]: r["outcome"] for r in teachers}
    )
    rows, choices = [], []
    for meta, z, t in zip(split["rows"], zeros, teachers, strict=True):
        group = z["group"]
        zgood = z["outcome"]["task_pass"] and z["outcome"]["safe"] and not zero_nonfoot[group]
        tgood = t["outcome"]["task_pass"] and t["outcome"]["safe"] and not teacher_nonfoot[group]
        selected = "zero" if zgood else "teacher" if tgood else None
        choices.append(
            dict(
                group=group,
                partition=meta["partition"],
                source_seed=meta["source_seed"],
                selected=selected,
                zero_success=zgood,
                teacher_success=tgood,
            )
        )
        for receipt, nonfoot in ((z, zero_nonfoot[group]), (t, teacher_nonfoot[group])):
            chosen = receipt["role"] == selected
            rows.append(
                dict(
                    group=group,
                    source_seed=meta["source_seed"],
                    partition=meta["partition"],
                    role=receipt["role"],
                    source_receipt_hash=receipt["receipt_hash"],
                    original_outcome=receipt["outcome"],
                    dense_nonfoot_over1n=nonfoot,
                    experimental_actual_path_supervision=chosen,
                    label_kind=(
                        "ACTUAL_NAV_ZERO_ACTIONS" if selected == "zero" else "ACTUAL_IK_ACTIONS"
                    )
                    if chosen
                    else "NO_POSITIVE_SUPERVISION",
                    zero_weight_is_not_negative_training=not chosen,
                    actual_same_body_state_labels_required=True,
                    counterfactual_labels=False,
                )
            )
    selected_train = [
        r for r in rows if r["partition"] == "train" and r["experimental_actual_path_supervision"]
    ]
    result = dict(
        schema="soccer.rsi.proposed_actual_winner_supervision.v1",
        scope_hash=scope["scope_hash"],
        prospective_split_hash=split["split_hash"],
        paired_zero_vs_teacher=paired,
        choices=choices,
        rows=rows,
        whole_recorded_positive_bodies=16,
        derived_courses=8,
        source_scene_clusters=4,
        selected_train_paths=len(selected_train),
        selected_train_sources=sorted({r["source_seed"] for r in selected_train}),
        selected_zero_train_anchors=sum(r["role"] == "zero" for r in selected_train),
        selected_corrective_train_paths=sum(r["role"] == "teacher" for r in selected_train),
        selected_validation_paths=sum(
            r["partition"] == "validation" and r["experimental_actual_path_supervision"]
            for r in rows
        ),
        zero_anchors_preferred_over_unneeded_corrections=True,
        successful_path_union_is_not_one_executable_policy=True,
        validation_excluded_from_gradient_and_normalization=True,
        original_primitive_labels_unchanged=True,
        original_failed_controllers_retained=True,
        source_file_authentication_performed_here=False,
        proposed_research_data_only=True,
        new_physical_steps=0,
        optimizer_updates=0,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        neural_training_authorized=False,
        runtime_execution_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))
