"""Soccer evidence adapter to task-neutral ROSClaw Growth contracts.

No controller, Runtime, driver, registry mutation or actuator authority. Isaac
discovery evidence cannot fill missing fresh/CPU qualifications automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.approach_lateral_tracking_evidence import audit_lateral_approach
from rosclaw_soccer.rsi.contact_motor_primitive import load_policy
from rosclaw_soccer.rsi.independent_first_touch_bank import post_contact_displacement
from rosclaw_soccer.rsi.taskspace_swing_evidence import audit_taskspace_swing_trace
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def growth_payload(
    *,
    validation: dict[str, Any],
    policy: dict[str, Any],
    protocol: dict[str, Any],
    body_hash: str,
    parent_hash: str,
) -> dict[str, Any]:
    """Pure contract translation; caller must audit raw evidence first."""
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

    from scripts.rsi_collect_bilateral_motor_validation_v305 import score

    outcome = score(validation["rows"])
    if not outcome["complete"]:
        raise ValueError("full consumed validation required for Core translation")
    retention = outcome["old_high_quality_loss"] == 0 and outcome["raw_gain_case_loss"] == 0
    safety = outcome["safe"] and outcome["clean_foot_loss"] == 0 and outcome["new_out_of_play"] == 0
    learning = (
        outcome["high_quality_count"]["learned"]
        >= max(outcome["high_quality_count"]["gain_08"], outcome["high_quality_count"]["gain_12"])
        + 2
    )
    spec = SkillGrowthSpec(
        skill_id="soccer.first_touch",
        adapter_id="soccer.rsi",
        body_hashes=(body_hash,),
        capability_ids=("soccer.first_touch",),
        observation_contract_hash=hash_json(
            {
                "clock": "isaac.20ms",
                "observations": [
                    "ball_root_gap",
                    "frozen_swing_target",
                    "joint_limits",
                    "previous_residual",
                    "past_contact",
                ],
            }
        ),
        action_contract_hash=hash_json(
            {
                k: policy[k]
                for k in (
                    "schema",
                    "joint_names",
                    "cap_rad",
                    "slew_rad_per_frame",
                    "release_frames",
                    "policy_source_hash",
                )
            }
        ),
        reward_contract_hash=hash_json(
            {
                "primary": "high_quality_count",
                "forward_60_m_min": 1,
                "lateral_ratio_max": 0.3,
                "excursion_max_m": 4,
            }
        ),
        cost_contract_hash=hash_json(
            {"min_pelvis_m": 0.65, "clean_foot_loss_max": 0, "new_out_max": 0}
        ),
        practice_source_ids=("soccer.isaac.independent.first_touch",),
        collective_source_ids=(),
        allowed_dream_types=("physics_counterfactual",),
        allowed_learner_ids=("cem.motor_primitive",),
        historical_anchor_hashes=(parent_hash,),
        boundary_suite_hash=hash_json(protocol),
        metrics=(
            GrowthMetricSpec(
                "soccer.high_quality_count",
                MetricDirection.MAXIMIZE,
                primary=True,
                minimum_relative_improvement=0.05,
                require_ci_lower_bound_positive=False,
            ),
        ),
        promotion_profile_hash=hash_json(
            {
                "fresh_holdout_required": True,
                "cpu_mujoco_required": True,
                "long_chain_required": True,
                "hardware_authorized": False,
            }
        ),
        rollback_policy_hash=parent_hash,
    )
    gates = (
        GateResult(
            GateName.LEARNING,
            GateStatus.PASS if learning else GateStatus.FAIL,
            validation["report_hash"],
            "Fixed consumed-suite point gain only; not statistical unseen-course evidence",
        ),
        GateResult(
            GateName.RETENTION,
            GateStatus.PASS if retention else GateStatus.FAIL,
            validation["report_hash"],
            "Both old high-quality anchors and raw gain successes must be retained",
        ),
        GateResult(
            GateName.SAFETY,
            GateStatus.PASS if safety else GateStatus.FAIL,
            validation["report_hash"],
            "Declared pilot pelvis/contact/out guardrails only",
        ),
        GateResult(
            GateName.APPLICABILITY,
            GateStatus.MISSING,
            detail="No independent CPU MuJoCo replay of this motor model",
        ),
        GateResult(
            GateName.DARWIN,
            GateStatus.MISSING,
            detail="No fresh holdout, retention distribution or continuous team qualification",
        ),
    )
    absence = {
        "status": "NOT_RUN",
        "candidate_hash": policy["policy_hash"],
        "missing": ["fresh_holdout", "cpu_mujoco", "continuous_team"],
    }
    decision = (
        ConsolidationDecision.REJECT
        if any(g.status is GateStatus.FAIL for g in gates)
        else ConsolidationDecision.NEED_MORE_EVIDENCE
    )
    manifest = ConsolidationManifest(
        skill_growth_spec_hash=spec.spec_hash,
        candidate_artifact_hash=policy["policy_hash"],
        parent_artifact_hash=parent_hash,
        rollback_artifact_hash=parent_hash,
        learned_changes={"bilateral_motor": policy["policy_hash"]},
        new_capability_ids=(),
        retained_capability_ids=("soccer.first_touch.anchors",) if retention else (),
        forgotten_capability_ids=() if retention else ("soccer.first_touch.anchors",),
        gate_results=gates,
        darwin_report_hash=hash_json(absence),
        decision=decision,
    )
    return {
        "schema": "soccer.rsi.core_motor_growth_export.v1",
        "activation_ceiling": "SIM_ONLY",
        "skill_growth_spec": spec.to_dict(),
        "skill_growth_spec_hash": spec.spec_hash,
        "consolidation_manifest": manifest.to_dict(),
        "manifest_hash": manifest.manifest_hash,
        "evidence_use_policy": EvidenceUsePolicy(EvidenceLevel.ACCELERATED_SIM).to_dict(),
        "darwin_absence_record": absence,
        "scope": "Consumed deterministic first-touch pilot, not full team",
        "hardware_authorized": False,
        "registry_write_count": 0,
    }


def export_motor_growth(
    *,
    validation_root: Path,
    policy_path: Path,
    protocol_path: Path,
    zero_parent_policy_path: Path,
) -> dict[str, Any]:
    """Audit every raw arm before creating a Core-compatible pending manifest."""
    from scripts.rsi_collect_bilateral_motor_validation_v305 import (
        POLICY_HASH,
        TRAIN_REPORT_HASH,
        score,
    )

    path = validation_root / "validation_summary.json"
    validation = json.loads(path.read_text(encoding="utf-8"))
    policy, _ = load_policy(policy_path)
    parent, parent_knots = load_policy(zero_parent_policy_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        validation.get("schema") != "rsi_bilateral_motor_consumed_validation_v305"
        or validation.get("report_hash")
        != hash_json({k: v for k, v in validation.items() if k != "report_hash"})
        or validation.get("training_report_hash") != TRAIN_REPORT_HASH
        or validation.get("motor_policy_hash") != POLICY_HASH
        or policy["policy_hash"] != POLICY_HASH
        or validation.get("protocol_hash") != hash_json(protocol)
        or parent_knots.any()
        or parent["training_commitment"] != policy["training_commitment"]
        or validation.get("promotion_authorized") is not False
        or validation.get("fresh_holdout_open_authorized") is not False
        or not score(validation["rows"])["complete"]
    ):
        raise ValueError("unsealed complete consumed validation for Core export")
    bodies = set()
    audits = []
    for row in validation["rows"]:
        for arm in ("gain_08", "gain_12", "learned"):
            folder = validation_root / f"seed{row['seed']}-lane{row['lane']}-{arm}-actor"
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            audit = audit_lateral_approach(folder)
            if report["report_hash"] != row["arms"][arm]["report_hash"] or report.get(
                "contact_motor_policy_hash"
            ) != (POLICY_HASH if arm == "learned" else None):
                raise ValueError("raw arm does not bind validation ledger")
            with np.load(folder / "late_swing_action_trace.npz", allow_pickle=False) as swing:
                audit_taskspace_swing_trace(swing, report, frames=300, count=1)
            observed = report["environments"][0]
            with np.load(folder / "trace.npz", allow_pickle=False) as physics:
                displacement = post_contact_displacement(
                    physics["ball_position_m"][:, 0], observed["first_contact_frame"]
                )
            from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality

            measured = {
                "contact_body_indices": observed["contact_body_indices"],
                "first_contact_frame": observed["first_contact_frame"],
                "minimum_pelvis_z_m": observed["minimum_pelvis_z_m"],
                "maximum_lateral_excursion_m": report["single_instance_max_lateral_excursion_m"],
                "clean_foot_only": bool(
                    observed["contact_body_indices"]
                    and set(observed["contact_body_indices"]) <= {0, 1}
                ),
                **displacement,
            }
            measured["high_quality"] = high_quality(measured)
            if (
                any(row["arms"][arm].get(k) != v for k, v in measured.items())
                or row["arms"][arm]["command_audit_hash"] != audit["report_hash"]
            ):
                raise ValueError("validation metrics disagree with raw physical evidence")
            bodies.add(report["asset_hash"])
            audits.append(audit["report_hash"])
    if len(bodies) != 1:
        raise ValueError("motor validation body changed")
    result = growth_payload(
        validation=validation,
        policy=policy,
        protocol=protocol,
        body_hash=bodies.pop(),
        parent_hash=parent["policy_hash"],
    )
    import rosclaw.growth.contracts as core_contracts

    result.update(
        validation_report_hash=validation["report_hash"],
        raw_command_audit_hashes=audits,
        core_contract_source_hash=hash_bytes(Path(core_contracts.__file__).read_bytes()),
    )
    result["report_hash"] = hash_json(result)
    return result
