"""One bounded contact-orientation study, not a new navigation sweep or policy."""

from typing import Any

from rosclaw_soccer.rsi.first_touch_stage_diagnostics import assess_first_touch_stages
from rosclaw_soccer.rsi.original_ik_planner_scope import ik_scope
from rosclaw_soccer.rsi.planner_navigation_pulse_scope import GROUPS
from rosclaw_soccer.sim.contracts import hash_json

PROFILE = "foot-heading-contact-six-v1"


def require_heading_scope(cfg: dict[str, Any]) -> None:
    if (
        cfg.get("exam_profile") != PROFILE
        or cfg.get("mode") != "SIM_ONLY"
        or cfg.get("leg_reference_kind") != "ORIGINAL_BOUNDED_IK_WITH_HEADING_PROBE"
        or cfg.get("heading_group_order") != list(GROUPS)
        or type(cfg.get("original_ik_predecessor")) is not dict
        or cfg.get("required_receipt_witnesses")
        != ["intent_witness", "encoder_witness", "prefix_witness"]
        or any(
            k in cfg
            for k in (
                "negative_predecessor",
                "signed_predecessors",
                "actual_winner_fit",
                "matched_ablation",
                "reference_student_model",
            )
        )
        or any(
            type(cfg.get(k)) is not int or cfg[k] != v
            for k, v in (
                ("max_attempts", 6),
                ("max_generations", 1),
                ("plateau_generations", 1),
                ("new_physical_budget", 6),
                ("new_independent_cpu_budget", 6),
                ("combined_body_substep_budget", 36000),
                ("new_optimizer_updates", 0),
            )
        )
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "private_fresh_accessed",
                "neural_training_authorized",
                "teacher_auto_admission",
                "use_mesh_surface_swing",
                "suppress_postcontact_swing_release",
                "use_contact_exit_torque_mpc",
                "use_directional_contact_compliance",
            )
        )
    ):
        raise ValueError(
            "fixed six heading OFF/ON bodies and closed original-IK predecessor required"
        )


def heading_scope(parent_model_hash: str) -> dict[str, Any]:
    original = ik_scope(parent_model_hash)
    result = dict(
        schema="soccer.rsi.foot_heading_six_scope.v1",
        profile=PROFILE,
        source_plan=original["source_plan"],
        source_seeds=original["source_seeds"],
        jobs=[
            dict(
                job=f"foot-heading-{role}-g{group}",
                role=role,
                plan="positive19",
                group=group,
                lateral_delta_m_s=0.08,
                heading_enabled=enabled,
                use_generated_binding=True,
            )
            for group in GROUPS
            for role, enabled in (("heading_off", False), ("heading_on", True))
        ],
        same_positive_ik_planner_control_in_both_arms=True,
        learned_and_zero_reference_models_forbidden=True,
        original_parent_mean_frozen=True,
        original_ik_positive_off_all300_array_bits_required=True,
        off_original_array_count=36,
        on_pre_action_physical_prefix_frames=30,
        completed_c2_whole_audit_dense_stage_required=True,
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


def assess_heading_scope(scope: dict[str, Any], receipts: list[dict[str, Any]]) -> dict[str, Any]:
    if (
        hash_json(scope) != hash_json(heading_scope(scope["source_plan"]["behavior_model_hash"]))
        or len(receipts) != 6
        or len({r["receipt_hash"] for r in receipts}) != 6
        or any(type(r.get("heading_enabled")) is not bool for r in receipts)
        or any(
            r.get(k) != v
            for j, r in zip(scope["jobs"], receipts, strict=True)
            for k, v in j.items()
        )
        or any(
            r["receipt_hash"] != hash_json({k: v for k, v in r.items() if k != "receipt_hash"})
            for r in receipts
        )
    ):
        raise ValueError("whole ordered fixed heading cohort including all failures required")
    stages = assess_first_touch_stages(
        {str(r["group"]): r["outcome"] for r in receipts if r["role"] == "heading_off"},
        {str(r["group"]): r["outcome"] for r in receipts if r["role"] == "heading_on"},
        declared_pair_ids=tuple(str(g) for g in GROUPS),
    )
    result = dict(
        schema="soccer.rsi.foot_heading_six_assessment.v1",
        stages=stages,
        all_failed_courses_retained=True,
        geometric_proposal_not_learned_gain=True,
        optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))
