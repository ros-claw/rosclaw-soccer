"""Six predeclared train-only navigation authority bodies, not a policy exam."""

from typing import Any

from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs
from rosclaw_soccer.rsi.positive_lateral_receding_scope import build_scope
from rosclaw_soccer.sim.contracts import hash_json

PROFILE = "pre-planner-navigation-six-v1"
GROUPS = (28, 76, 12)


def pulse_scope(parent_model_hash: str) -> dict[str, Any]:
    positive = build_scope(parent_model_hash)
    by_group = {c["source_group"]: c for c in positive["contexts"]}
    jobs = [
        dict(
            job=f"planner-pulse-{role}-g{group}",
            role=role,
            plan="positive19",
            group=group,
            lateral_delta_m_s=delta,
            use_generated_binding=True,
        )
        for group in GROUPS
        for role, delta in (("pulse_off", 0.0), ("pulse", -0.08))
    ]
    result = dict(
        schema="soccer.rsi.pre_planner_pulse_six_scope.v1",
        jobs=jobs,
        source_plan=positive["source_plan"],
        positive_scope_hash=positive["scope_hash"],
        derived_courses=3,
        source_scene_clusters=3,
        source_seeds=[by_group[g]["source_seed"] for g in GROUPS],
        only_predeclared_training_sources=True,
        validation_or_fresh_sources_used=False,
        new_native_bodies=6,
        new_independent_cpu_bodies=6,
        combined_body_substeps=36000,
        new_optimizer_updates=0,
        same_zero_reference_model_for_both_arms=True,
        original_parent_mean_frozen=True,
        off_all300_arrays_required_before_pulse=True,
        pre_action_physical_prefix_frames=100,
        control_period_s=0.02,
        active_planner_event_frames=[100, 120, 140],
        reset_planner_event_frame=160,
        planner_stride_frames=20,
        reference_lookahead_frames=10,
        no_early_event_replanning=True,
        trained_policy=False,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(result, scope_hash=hash_json(result))


def require_pulse_scope(cfg: dict[str, Any]) -> None:
    if (
        cfg.get("exam_profile") != PROFILE
        or cfg.get("mode") != "SIM_ONLY"
        or "actual_winner_fit" in cfg
        or cfg.get("pulse_group_order") != list(GROUPS)
        or cfg.get("required_receipt_witnesses")
        != ["intent_witness", "encoder_witness", "prefix_witness"]
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
            "fixed three train courses x OFF/pulse, original bounds and zero training required"
        )


def assess_pulse_scope(scope: dict[str, Any], receipts: list[dict[str, Any]]) -> dict[str, Any]:
    expected = pulse_scope(scope["source_plan"]["behavior_model_hash"])
    if (
        scope != expected
        or len(receipts) != 6
        or len({r["receipt_hash"] for r in receipts}) != 6
        or any(
            r.get(k) != v
            for job, r in zip(expected["jobs"], receipts, strict=True)
            for k, v in job.items()
        )
    ):
        raise ValueError("whole ordered six bodies including all failures required")
    paired = assess_pairs(
        {r["group"]: r["outcome"] for r in receipts if r["role"] == "pulse_off"},
        {r["group"]: r["outcome"] for r in receipts if r["role"] == "pulse"},
    )
    result = dict(
        schema="soccer.rsi.pre_planner_pulse_six_assessment.v1",
        paired_results=paired,
        derived_courses=3,
        source_scene_clusters=3,
        all_failed_courses_retained=True,
        diagnostic_authority_probe_not_learned_gain=True,
        optimizer_updates=0,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))
