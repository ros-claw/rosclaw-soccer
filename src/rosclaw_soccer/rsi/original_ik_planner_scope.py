"""Fixed original-IK composition study, not a learned policy or fresh exam."""

from typing import Any

from rosclaw_soccer.rsi.first_touch_stage_diagnostics import assess_first_touch_stages
from rosclaw_soccer.rsi.planner_navigation_pulse_scope import (
    GROUPS,
    POSITIVE_PROFILE,
    pulse_scope,
    require_pulse_scope,
)
from rosclaw_soccer.sim.contracts import hash_json

PROFILE = "original-ik-pre-planner-positive-six-v1"


def require_ik_scope(cfg: dict[str, Any]) -> None:
    if (
        cfg.get("exam_profile") != PROFILE
        or type(cfg.get("signed_predecessors")) is not dict
        or set(cfg["signed_predecessors"]) != {"negative", "positive"}
        or any(type(v) is not dict for v in cfg["signed_predecessors"].values())
        or "negative_predecessor" in cfg
        or cfg.get("leg_reference_kind") != "ORIGINAL_BOUNDED_IK"
        or cfg.get("original_teacher_all300_off_parity_required") is not True
        or any(k in cfg for k in ("reference_student_model", "matched_ablation"))
    ):
        raise ValueError(
            "explicit original-IK six-body intent and both whole signed predecessors required"
        )
    # Reuse budgets/forbidden interventions only; do not retag the new scope as NAV-zero.
    shared = dict(cfg, exam_profile=POSITIVE_PROFILE, negative_predecessor={})
    require_pulse_scope(shared)


def ik_scope(parent_model_hash: str) -> dict[str, Any]:
    original = pulse_scope(parent_model_hash, profile=POSITIVE_PROFILE)
    result = dict(
        schema="soccer.rsi.original_ik_planner_six_scope.v1",
        profile=PROFILE,
        source_plan=original["source_plan"],
        source_seeds=original["source_seeds"],
        jobs=[
            dict(
                job=f"original-ik-{role}-g{group}",
                role=role,
                plan="positive19",
                group=group,
                lateral_delta_m_s=delta,
                use_generated_binding=True,
            )
            for group in GROUPS
            for role, delta in (("ik_pulse_off", 0.0), ("ik_pulse", 0.08))
        ],
        leg_reference_kind="ORIGINAL_BOUNDED_IK",
        learned_and_zero_reference_models_forbidden=True,
        original_parent_mean_frozen=True,
        original_teacher_all300_off_parity_required=True,
        pre_action_physical_prefix_frames=100,
        new_native_bodies=6,
        new_independent_cpu_bodies=6,
        combined_body_substeps=36000,
        new_optimizer_updates=0,
        derived_courses=3,
        source_scene_clusters=3,
        validation_or_fresh_sources_used=False,
        trained_policy=False,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(result, scope_hash=hash_json(result))


def assess_ik_scope(scope: dict[str, Any], receipts: list[dict[str, Any]]) -> dict[str, Any]:
    if (
        scope != ik_scope(scope["source_plan"]["behavior_model_hash"])
        or len(receipts) != 6
        or len({r["receipt_hash"] for r in receipts}) != 6
        or any(
            r.get(k) != v
            for job, r in zip(scope["jobs"], receipts, strict=True)
            for k, v in job.items()
        )
        or any(
            r["receipt_hash"] != hash_json({k: v for k, v in r.items() if k != "receipt_hash"})
            for r in receipts
        )
    ):
        raise ValueError("whole ordered six original-IK bodies including all failures required")
    stages = assess_first_touch_stages(
        {str(r["group"]): r["outcome"] for r in receipts if r["role"] == "ik_pulse_off"},
        {str(r["group"]): r["outcome"] for r in receipts if r["role"] == "ik_pulse"},
        declared_pair_ids=tuple(str(g) for g in GROUPS),
    )
    result = dict(
        schema="soccer.rsi.original_ik_planner_six_assessment.v1",
        stages=stages,
        all_failures_retained=True,
        diagnostic_composition_not_learned_gain=True,
        optimizer_updates=0,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))
