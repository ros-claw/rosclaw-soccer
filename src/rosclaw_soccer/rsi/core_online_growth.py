"""Pending Core Growth manifest from independently replayed online physics.

Isaac training evidence is E2: it cannot be promotion truth by itself. Even a
consumed-suite pass leaves fresh, CPU and long-chain qualifications missing.
No registry mutation, runtime activation, or physical hardware authority.
"""

from __future__ import annotations

from typing import Any

from rosclaw_soccer.sim.contracts import hash_json


def online_growth_payload(review: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    from rosclaw.growth.contracts import (
        ConsolidationDecision,
        ConsolidationManifest,
        EvidenceLevel,
        EvidenceUsePolicy,
        GateName,
        GateResult,
        GateStatus,
        GrowthMetricSpec,
        MetricDirection,
        SkillGrowthSpec,
    )

    from rosclaw_soccer.rsi.online_motor_actor_critic import validate_model

    validate_model(model)
    if (
        review.get("report_hash")
        != hash_json({k: v for k, v in review.items() if k != "report_hash"})
        or review.get("training_complete") is not True
        or review.get("promotion_authorized") is not False
        or review.get("fresh_holdout_open_authorized") is not False
        or review.get("best_model_hash") != model["model_hash"]
    ):
        raise ValueError("complete independently replayed best model review required")
    matching = [g for g in review["generations"] if g["model_hash"] == model["model_hash"]]
    if not matching:
        raise ValueError("best model is unchanged predecessor, no candidate qualification")
    ledger = matching[-1]
    score = ledger["score"]
    retention = ledger["previous_nine_high_quality_loss"] == 0 and all(
        score[k] == 0
        for k in (
            "predecessor_high_quality_loss",
            "raw_gain_high_quality_loss",
            "old_high_quality_loss",
        )
    )
    safety = (
        score["safe_pelvis_guardrail"]
        and score["old_clean_foot_loss"] == 0
        and score["new_out_of_play"] == 0
    )
    learning = score["high_quality_count"] > 9
    spec = SkillGrowthSpec(
        skill_id="soccer.first_touch",
        adapter_id="soccer.rsi.online",
        body_hashes=(review["body_hash"],),
        capability_ids=("soccer.first_touch",),
        observation_contract_hash=hash_json(
            {
                "names": model["base_model"]["observation_names"],
                "decision_frame": 30,
                "future_input": False,
            }
        ),
        action_contract_hash=hash_json(
            {
                "parameters": 37,
                "joint_cap_rad": 0.16,
                "phase_gap_end_m": [-0.35, 0.25],
                "activation_ceiling": "SIM_ONLY",
            }
        ),
        reward_contract_hash=hash_json(
            {
                "raw": "first_touch_reward",
                "quality_bonus": 10,
                "dirty_contact_cost": 8,
                "out_cost": 20,
                "pelvis_cost": 100,
            }
        ),
        cost_contract_hash=hash_json({"min_pelvis_m": 0.65, "old_clean_loss": 0, "new_out": 0}),
        practice_source_ids=("soccer.isaac.independent.first_touch",),
        collective_source_ids=(),
        allowed_dream_types=("physics_counterfactual",),
        allowed_learner_ids=("contextual.actor_critic.protected_awr",),
        historical_anchor_hashes=(model["base_model"]["model_hash"],),
        boundary_suite_hash=review["report_hash"],
        metrics=(
            GrowthMetricSpec(
                "soccer.high_quality_count",
                MetricDirection.MAXIMIZE,
                primary=True,
                require_ci_lower_bound_positive=False,
            ),
        ),
        promotion_profile_hash=hash_json(
            {
                "fresh_required": True,
                "cpu_mujoco_required": True,
                "continuous_chain_required": True,
                "hardware_authorized": False,
            }
        ),
        rollback_policy_hash=model["base_model"]["model_hash"],
    )
    gates = tuple(
        GateResult(
            name, GateStatus.PASS if passed else GateStatus.FAIL, review["report_hash"], detail
        )
        for name, passed, detail in (
            (GateName.LEARNING, learning, "Consumed-course physical gain; not fresh statistics"),
            (
                GateName.RETENTION,
                retention,
                "All nine previous neural successes plus historical anchors",
            ),
            (GateName.SAFETY, safety, "Measured pilot contact/out/pelvis guardrails only"),
        )
    ) + (
        GateResult(
            GateName.APPLICABILITY,
            GateStatus.MISSING,
            detail="No CPU MuJoCo qualification of this online model",
        ),
        GateResult(
            GateName.DARWIN,
            GateStatus.MISSING,
            detail="Fresh quarantine unopened; continuous chain not qualified",
        ),
    )
    missing = {
        "candidate_hash": model["model_hash"],
        "status": "NOT_RUN",
        "missing": ["fresh", "cpu_mujoco", "continuous_chain"],
    }
    decision = (
        ConsolidationDecision.REJECT
        if any(g.status is GateStatus.FAIL for g in gates)
        else ConsolidationDecision.NEED_MORE_EVIDENCE
    )
    manifest = ConsolidationManifest(
        skill_growth_spec_hash=spec.spec_hash,
        candidate_artifact_hash=model["model_hash"],
        parent_artifact_hash=model["base_model"]["model_hash"],
        rollback_artifact_hash=model["base_model"]["model_hash"],
        learned_changes={"actor": model["model_hash"], "value_baseline": model["model_hash"]},
        new_capability_ids=(),
        retained_capability_ids=("soccer.first_touch.anchors",) if retention else (),
        forgotten_capability_ids=() if retention else ("soccer.first_touch.anchors",),
        gate_results=gates,
        darwin_report_hash=hash_json(missing),
        decision=decision,
    )
    result = dict(
        schema="soccer.rsi.core_online_growth_export.v308",
        skill_growth_spec=spec.to_dict(),
        consolidation_manifest=manifest.to_dict(),
        manifest_hash=manifest.manifest_hash,
        review_hash=review["report_hash"],
        evidence_use_policy=EvidenceUsePolicy(EvidenceLevel.ACCELERATED_SIM).to_dict(),
        darwin_absence_record=missing,
        activation_ceiling="SIM_ONLY",
        hardware_authorized=False,
        registry_write_count=0,
    )
    result["report_hash"] = hash_json(result)
    return result
