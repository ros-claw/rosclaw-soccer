"""Matched promotion gate for OpenTrack residual athlete candidates."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.continual.residual_adaptation import (
    ParameterIsolationEvidence,
    ResidualAdaptationContract,
)

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class OpenTrackResidualPromotionDecision:
    verdict: str
    reasons: tuple[str, ...]
    relative_retention_passed: bool
    relative_acquisition_passed: bool
    absolute_physics_passed: bool
    parameter_isolation_passed: bool
    critical_safety_regressions: int
    relative_guardrails: dict[str, bool]
    evidence: ParameterIsolationEvidence
    schema_version: str = "rosclaw_soccer.opentrack_residual_promotion.v1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["evidence"]["evidence_hash"] = self.evidence.evidence_hash
        payload["decision_hash"] = hash_json(payload)
        return payload


def evaluate_opentrack_residual_candidate(
    *,
    contract: ResidualAdaptationContract,
    parent_report: dict[str, Any],
    candidate_report: dict[str, Any],
    isolation_report: dict[str, Any],
) -> OpenTrackResidualPromotionDecision:
    """Require matched exams, immutable base parameters, and real acquisition gains."""

    if parent_report.get("plan_hash") != candidate_report.get("plan_hash"):
        raise ValueError("OpenTrack promotion requires an identical matched exam plan")
    if candidate_report.get("reference_policy_hash") != parent_report.get("policy_hash"):
        raise ValueError("OpenTrack candidate was not compared against the sealed parent")
    if isolation_report.get("frozen_base_unchanged") is not True:
        raise ValueError("OpenTrack residual candidate mutated its frozen base")
    parent_episodes = {
        item["episode_id"]: item for item in parent_report.get("episode_reports", [])
    }
    candidate_episodes = {
        item["episode_id"]: item for item in candidate_report.get("episode_reports", [])
    }
    if not parent_episodes or set(parent_episodes) != set(candidate_episodes):
        raise ValueError("OpenTrack promotion episode sets do not match")
    critical_regressions = sum(
        bool(parent["critical"] and not parent["fell"] and candidate_episodes[key]["fell"])
        for key, parent in parent_episodes.items()
    )
    parent_suites = parent_report["suite_summary"]
    candidate_suites = candidate_report["suite_summary"]
    parent_retention = _suite_physics(parent_episodes.values(), suite_id="retention")
    candidate_retention = _suite_physics(candidate_episodes.values(), suite_id="retention")
    parent_acquisition = _suite_physics(parent_episodes.values(), suite_id="acquisition")
    candidate_acquisition = _suite_physics(candidate_episodes.values(), suite_id="acquisition")
    guardrails = {
        "retention_no_critical_fall_regression": critical_regressions == 0,
        "retention_fall_count_not_worse": candidate_suites["retention"]["fall_count"]
        <= parent_suites["retention"]["fall_count"],
        "retention_recovery_not_worse": candidate_suites["retention"]["recovery_rate"]
        >= parent_suites["retention"]["recovery_rate"],
        "retention_joint_rmse_within_20pct": candidate_retention["joint_rmse_rad"]
        <= parent_retention["joint_rmse_rad"] * 1.20,
        "retention_keypoint_not_worse": candidate_retention["keypoint_mpjpe_m"]
        <= parent_retention["keypoint_mpjpe_m"],
        "retention_jerk_not_worse": candidate_retention["joint_jerk_rms_rad_s3"]
        <= parent_retention["joint_jerk_rms_rad_s3"],
        "retention_pelvis_floor_within_1cm": candidate_retention["minimum_pelvis_height_m"]
        >= parent_retention["minimum_pelvis_height_m"] - 0.01,
        "retention_torque_saturation_not_worse": candidate_retention["torque_saturation_rate"]
        <= parent_retention["torque_saturation_rate"],
        "acquisition_fewer_falls": candidate_suites["acquisition"]["fall_count"]
        < parent_suites["acquisition"]["fall_count"],
        "acquisition_recovery_improved": candidate_suites["acquisition"]["recovery_rate"]
        > parent_suites["acquisition"]["recovery_rate"],
        "acquisition_joint_rmse_improved": candidate_acquisition["joint_rmse_rad"]
        < parent_acquisition["joint_rmse_rad"],
        "acquisition_keypoint_improved": candidate_acquisition["keypoint_mpjpe_m"]
        < parent_acquisition["keypoint_mpjpe_m"],
        "acquisition_jerk_improved": candidate_acquisition["joint_jerk_rms_rad_s3"]
        < parent_acquisition["joint_jerk_rms_rad_s3"],
        "acquisition_pelvis_floor_not_worse": candidate_acquisition["minimum_pelvis_height_m"]
        >= parent_acquisition["minimum_pelvis_height_m"],
        "acquisition_torque_saturation_improved": candidate_acquisition["torque_saturation_rate"]
        < parent_acquisition["torque_saturation_rate"],
    }
    retention_passed = all(
        value for key, value in guardrails.items() if key.startswith("retention_")
    )
    acquisition_passed = all(
        value for key, value in guardrails.items() if key.startswith("acquisition_")
    )
    residual_scale = float(candidate_report.get("residual_scale", 1.0))
    candidate_artifact_hash = hash_json(
        {
            "policy_hash": candidate_report["policy_hash"],
            "reference_policy_hash": candidate_report["reference_policy_hash"],
            "residual_scale": residual_scale,
        }
    )
    matched_exam_hash = hash_json(
        {
            "plan_hash": parent_report["plan_hash"],
            "parent_report_hash": parent_report["report_hash"],
            "candidate_report_hash": candidate_report["report_hash"],
        }
    )
    evidence = ParameterIsolationEvidence(
        adaptation_contract_hash=contract.contract_hash,
        parent_artifact_hash=contract.parent_artifact_hash,
        candidate_artifact_hash=candidate_artifact_hash,
        frozen_base_hash_before=isolation_report["frozen_base_hash_before"],
        frozen_base_hash_after=isolation_report["frozen_base_hash_after"],
        matched_exam_hash=matched_exam_hash,
        examined_frozen_parameter_count=int(isolation_report["examined_frozen_parameter_count"]),
        examined_trainable_parameter_count=int(
            isolation_report["examined_trainable_parameter_count"]
        ),
        candidate_world_steps=int(isolation_report["candidate_world_steps"]),
        maximum_frozen_parameter_drift=float(isolation_report["maximum_frozen_parameter_drift"]),
        residual_output_rms=float(candidate_report["residual_output_rms"]),
        retention_passed=retention_passed,
        acquisition_passed=acquisition_passed,
        critical_safety_regressions=critical_regressions,
    )
    isolation_passed = evidence.passes(contract)
    absolute_passed = bool(candidate_report.get("passed"))
    reasons: list[str] = []
    if not retention_passed:
        reasons.append("relative_retention_gate_failed")
    if not acquisition_passed:
        reasons.append("relative_acquisition_gate_failed")
    if evidence.residual_output_rms > contract.maximum_residual_output_rms:
        reasons.append("residual_output_above_ceiling")
    if evidence.candidate_world_steps > contract.maximum_world_steps:
        reasons.append("training_steps_above_sealed_ceiling")
    if not absolute_passed:
        reasons.extend(str(item) for item in candidate_report.get("reasons", []))
    if isolation_passed and absolute_passed:
        verdict = "PROMOTED"
    elif isolation_passed:
        verdict = "DEVELOPMENT_ADVANCE_NOT_PROMOTED"
    else:
        verdict = "REJECTED"
    return OpenTrackResidualPromotionDecision(
        verdict=verdict,
        reasons=tuple(dict.fromkeys(reasons)),
        relative_retention_passed=retention_passed,
        relative_acquisition_passed=acquisition_passed,
        absolute_physics_passed=absolute_passed,
        parameter_isolation_passed=isolation_passed,
        critical_safety_regressions=critical_regressions,
        relative_guardrails=guardrails,
        evidence=evidence,
    )


def _suite_physics(episodes: Any, *, suite_id: str) -> dict[str, float]:
    items = tuple(item for item in episodes if item["suite_id"] == suite_id)
    if not items:
        raise ValueError(f"OpenTrack promotion is missing the {suite_id} suite")
    joint_count = sum(int(item["joint_error_count"]) for item in items)
    keypoint_count = sum(int(item["keypoint_error_count"]) for item in items)
    jerk_count = sum(int(item["joint_jerk_count"]) for item in items)
    control_steps = sum(int(item["control_steps"]) for item in items)
    if min(joint_count, keypoint_count, jerk_count, control_steps) <= 0:
        raise ValueError("OpenTrack promotion suite contains empty physical traces")
    return {
        "joint_rmse_rad": math.sqrt(
            sum(float(item["joint_squared_error_sum"]) for item in items) / joint_count
        ),
        "keypoint_mpjpe_m": math.sqrt(
            sum(float(item["keypoint_squared_error_sum"]) for item in items) / keypoint_count
        ),
        "joint_jerk_rms_rad_s3": math.sqrt(
            sum(float(item["joint_jerk_squared_sum"]) for item in items) / jerk_count
        ),
        "minimum_pelvis_height_m": min(float(item["minimum_pelvis_height_m"]) for item in items),
        "torque_saturation_rate": sum(int(item["saturated_control_steps"]) for item in items)
        / control_steps,
    }


def write_opentrack_residual_decision(
    decision: OpenTrackResidualPromotionDecision, output_path: Path
) -> dict[str, Any]:
    output = output_path.expanduser().resolve()
    if output.suffix != ".json" or output.exists():
        raise ValueError("OpenTrack promotion decision requires a new JSON output")
    payload = decision.to_dict()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return payload


__all__ = [
    "OpenTrackResidualPromotionDecision",
    "evaluate_opentrack_residual_candidate",
    "write_opentrack_residual_decision",
]
