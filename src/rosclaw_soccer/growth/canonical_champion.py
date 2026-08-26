"""Reconstruct Soccer's canonical champion tracks from sealed evidence.

This adapter is deliberately football-specific; the replay and promotion
semantics live in ROSClaw Core.  It repairs the S12--S31 development lineage by
distinguishing the active global champion from independently qualified
specialists and rejected research branches.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw.continual.champion_registry import (
    CanonicalChampionRegistry,
    ChampionRecordKind,
    ChampionRegistryRecord,
    PromotionAuthority,
)

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


@dataclass(frozen=True)
class SoccerChampionEvidenceLayout:
    """Relative paths for the evidence used by the S33 lineage repair."""

    s12_global_exam: str = (
        "s12-goalkeeper-corrected-teacher-ppo-4gpu-dev-v1/"
        "cpu-mujoco-exam-guarded-v3.json"
    )
    s13_decision: str = (
        "s13-goalkeeper-agility-ppo-4gpu-dev-v2/paired-champion-decision.json"
    )
    s15_decision: str = (
        "s15-bimanual-stability-plasticity-ppo-4gpu-dev-v1/"
        "paired-champion-decision.json"
    )
    s23_decision: str = (
        "s23-motiondecode-second-glove-ppo-4gpu-dev-v1/paired-parent-s15-decision.json"
    )
    s28_decision: str = (
        "s28-strict-arm-only-goalkeeper-ppo-4gpu-dev-v1/"
        "paired-champion-s15-decision.json"
    )
    s31_specialist_exam: str = (
        "s31-conservative-elite-ppo-4gpu-dev-v1/"
        "cpu-final-elite-arm-scale-85-v2-160.json"
    )


DEFAULT_SOCCER_CHAMPION_EVIDENCE_LAYOUT = SoccerChampionEvidenceLayout()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _load_sealed_report(
    root: Path,
    relative_path: str,
    *,
    allow_invalid_wrapper_for_audit: bool = False,
) -> dict[str, Any]:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("evidence path escapes its root") from exc
    payload_bytes = path.read_bytes()
    payload = json.loads(payload_bytes)
    report = dict(_mapping(payload, label=relative_path))
    declared_hash = _string(report.pop("report_hash", None), label="report_hash")
    unsealed_annotations: list[str] = []
    integrity_status = "SEALED"
    if hash_json(report) != declared_hash:
        # Two historical reports were annotated after their report hash was
        # sealed.  These fields are retained for disclosure but are never used
        # as promotion evidence.  No other post-hash mutation is accepted.
        hash_payload = dict(report)
        for key in ("parent_cpu_exam", "qualification_evidence"):
            if key in hash_payload:
                unsealed_annotations.append(key)
                hash_payload.pop(key)
        if hash_json(hash_payload) == declared_hash:
            integrity_status = "SEALED_WITH_UNBOUND_ANNOTATIONS"
        elif allow_invalid_wrapper_for_audit:
            integrity_status = "INVALID_DECLARED_REPORT_HASH_AUDIT_ONLY"
        else:
            raise ValueError(f"sealed evidence report hash mismatch: {relative_path}")
    report["report_hash"] = declared_hash
    report["_unsealed_annotation_keys"] = unsealed_annotations
    report["_wrapper_integrity"] = integrity_status
    report["_evidence_file_hash"] = hash_bytes(payload_bytes)
    if report.get("activation_ceiling") != "SIM_ONLY":
        raise ValueError(f"sealed evidence exceeds SIM_ONLY: {relative_path}")
    if report.get("hardware_command_sent") is not False:
        raise ValueError(f"sealed evidence claims hardware execution: {relative_path}")
    return report


def _paired_fields(report: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    decision = _mapping(report.get("decision"), label="decision")
    candidate = _mapping(report.get("candidate"), label="candidate")
    if hash_json(decision) != report.get("decision_hash"):
        raise ValueError("paired decision hash mismatch")
    for name in ("parent_artifact_hash", "candidate_artifact_hash", "scenario_suite_hash"):
        if decision.get(name) is None:
            raise ValueError(f"paired decision is missing {name}")
    if decision["candidate_artifact_hash"] != candidate.get("artifact_hash"):
        raise ValueError("paired decision candidate does not match snapshot")
    if decision["parent_artifact_hash"] != candidate.get("parent_artifact_hash"):
        raise ValueError("paired decision parent does not match snapshot")
    if decision["scenario_suite_hash"] != candidate.get("scenario_suite_hash"):
        raise ValueError("paired decision suite does not match snapshot")
    return decision, candidate


def _baseline_suite_hash(report: Mapping[str, Any]) -> str:
    return str(
        hash_json(
            {
                "exam_config_hash": report.get("exam_config_hash"),
                "physics_scene_hash": report.get("physics_scene_hash"),
                "seeds": report.get("seeds"),
                "paired_rollouts": report.get("paired_rollouts"),
            }
        )
    )


def reconstruct_s33_champion_registry(
    evidence_root: Path,
    *,
    layout: SoccerChampionEvidenceLayout = DEFAULT_SOCCER_CHAMPION_EVIDENCE_LAYOUT,
) -> dict[str, Any]:
    """Build the canonical S33 registry and audit every later parent claim."""

    root = evidence_root.expanduser().resolve()
    s12 = _load_sealed_report(root, layout.s12_global_exam)
    if s12.get("passed") is not True or s12.get("promotion_status") != "PROMOTED_SIM_ONLY":
        raise ValueError("S12 global baseline did not pass its strict CPU MuJoCo exam")
    global_baseline = ChampionRegistryRecord(
        agent_id="agent.goalkeeper",
        track_id="global",
        artifact_hash=_string(s12.get("checkpoint_hash"), label="S12 checkpoint_hash"),
        evidence_hash=_string(s12.get("report_hash"), label="S12 report_hash"),
        scenario_suite_hash=_baseline_suite_hash(s12),
        authority=PromotionAuthority.BASELINE_STRICT_EXAM,
        kind=ChampionRecordKind.GLOBAL_BASELINE,
        evidence_valid=True,
        promotion_passed=True,
        generation=0,
    )
    registry = CanonicalChampionRegistry().append(global_baseline)
    audits: list[dict[str, Any]] = []

    for stage, relative_path in (
        ("S13", layout.s13_decision),
        ("S15", layout.s15_decision),
        ("S23", layout.s23_decision),
        ("S28", layout.s28_decision),
    ):
        report = _load_sealed_report(
            root,
            relative_path,
            allow_invalid_wrapper_for_audit=stage in {"S15", "S28"},
        )
        decision, candidate = _paired_fields(report)
        parent_hash = _string(
            decision.get("parent_artifact_hash"), label=f"{stage} parent_artifact_hash"
        )
        claim = registry.audit_parent_claim(
            track_id="global",
            claimed_parent_artifact_hash=parent_hash,
        )
        status = _string(decision.get("status"), label=f"{stage} decision status")
        replace = status == "REPLACE_CHAMPION"
        audit_entry: dict[str, Any] = {
            "stage": stage,
            "evidence_path": relative_path,
            "evidence_report_hash": report["report_hash"],
            "evidence_file_hash": report["_evidence_file_hash"],
            "wrapper_integrity": report["_wrapper_integrity"],
            "decision_hash": report["decision_hash"],
            "decision_status": status,
            "parent_claim": claim.to_dict(),
            "admitted_to_registry": False,
        }
        if claim.valid:
            active = registry.active_head("global")
            if active is None:
                raise RuntimeError("global track unexpectedly has no active head")
            record = ChampionRegistryRecord(
                agent_id="agent.goalkeeper",
                track_id="global",
                artifact_hash=_string(
                    candidate.get("artifact_hash"), label=f"{stage} candidate artifact"
                ),
                evidence_hash=_string(report.get("report_hash"), label=f"{stage} report_hash"),
                scenario_suite_hash=_string(
                    candidate.get("scenario_suite_hash"), label=f"{stage} scenario suite"
                ),
                authority=PromotionAuthority.PAIRED_DOMINANCE,
                kind=(
                    ChampionRecordKind.TRACK_REPLACEMENT
                    if replace
                    else ChampionRecordKind.CANDIDATE_ARCHIVED
                ),
                evidence_valid=bool(candidate.get("qualified")),
                promotion_passed=replace,
                generation=active.generation + 1,
                parent_record_hash=active.record_hash,
                parent_artifact_hash=active.artifact_hash,
            )
            registry = registry.append(record)
            audit_entry["admitted_to_registry"] = True
            audit_entry["record_hash"] = record.record_hash
        else:
            audit_entry["exclusion_reason"] = "parent_not_active_track_head"
        audits.append(audit_entry)

    s31 = _load_sealed_report(root, layout.s31_specialist_exam)
    if s31.get("passed") is not True or s31.get("parent_checkpoint_hash") is not None:
        raise ValueError("S31 specialist must pass a standalone sealed exam")
    specialist = ChampionRegistryRecord(
        agent_id="agent.goalkeeper",
        track_id="goalkeeper.arm.elite",
        artifact_hash=_string(s31.get("checkpoint_hash"), label="S31 checkpoint_hash"),
        evidence_hash=_string(s31.get("report_hash"), label="S31 report_hash"),
        scenario_suite_hash=_baseline_suite_hash(s31),
        authority=PromotionAuthority.SEALED_SPECIALIST_EXAM,
        kind=ChampionRecordKind.SPECIALIST_BASELINE,
        evidence_valid=True,
        promotion_passed=True,
        generation=0,
    )
    registry = registry.append(specialist)
    registry_report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.s33_canonical_champion_repair.v1",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "registry": registry.to_dict(),
        "registry_hash": registry.registry_hash,
        "lineage_audits": audits,
        "interpretation": {
            "global_champion": "S12",
            "specialist_baseline": "S31_ARM_ONLY",
            "excluded_global_descendants": ["S15", "S23", "S28"],
            "rule": "ONLY_ACTIVE_TRACK_HEAD_CAN_AUTHORIZE_A_PAIRED_CHILD",
        },
    }
    registry_report["report_hash"] = hash_json(registry_report)
    return registry_report


def write_s33_champion_registry(
    evidence_root: Path,
    output_path: Path,
    *,
    layout: SoccerChampionEvidenceLayout = DEFAULT_SOCCER_CHAMPION_EVIDENCE_LAYOUT,
) -> dict[str, Any]:
    """Atomically write the reconstructed registry outside the source tree."""

    report = reconstruct_s33_champion_registry(evidence_root, layout=layout)
    target = output_path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return report


__all__ = [
    "SoccerChampionEvidenceLayout",
    "reconstruct_s33_champion_registry",
    "write_s33_champion_registry",
]
