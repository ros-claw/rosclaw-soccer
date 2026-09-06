"""Independent evidence validation for S123 temporal multi-agent growth."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.temporal_route_actor import load_temporal_route_actor
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_REJECTED_HISTORY = (
    "s123-temporal-route-growth-v4/retention-v1/aborted-exam.json",
    "s123-temporal-route-growth-v4/retention-v2/retention-exam.json",
    "s123-temporal-route-growth-v6/retention-v3/retention-exam.json",
    "s123-temporal-route-growth-v6/retention-v4/retention-exam.json",
    "s123-temporal-route-growth-v10/retention-v5/retention-exam.json",
    "s123-temporal-route-growth-v10/retention-v6/retention-exam.json",
    "s123-temporal-route-growth-v10/retention-v7/retention-exam.json",
)


def write_temporal_route_stage_summary(
    evidence_dir: Path,
    *,
    source_checkout: Path,
    source_commit: str,
    retention_name: str = "retention-v8",
) -> dict[str, Any]:
    """Bind the passing exam, acquisition, failures, and exact implementation."""

    root = evidence_dir.expanduser().resolve()
    evidence_root = root.parent
    checkout = source_checkout.expanduser().resolve()
    destination = root / "stage-summary.json"
    if destination.exists() or not checkout.is_dir():
        raise ValueError("temporal route stage destination or checkout is invalid")
    source = _load(root / "source-snapshot.json")
    acquisition = _load(root / "population-report.json")
    development_path = (
        evidence_root
        / "s123-temporal-route-growth-v4/failure-development-v1/development-report.json"
    )
    development = _load(development_path)
    manifest_path = root / retention_name / "sealed-retention.json"
    retention_path = root / retention_name / "retention-exam.json"
    manifest = _load(manifest_path)
    retention = _load(retention_path)
    actor_path = root / "temporal-route-champion.json"
    actor = load_temporal_route_actor(actor_path)
    if not (
        acquisition.get("status") == "PASS_TEMPORAL_ROUTE_ACQUISITION"
        and retention.get("status") == "PASS_TEMPORAL_MULTI_AGENT_GROWTH"
        and isinstance(retention.get("gates"), dict)
        and all(value is True for value in retention["gates"].values())
        and manifest.get("training_access_allowed") is False
        and actor.actor_hash
        == acquisition.get("champion_hash")
        == retention.get("temporal_actor_hash")
    ):
        raise ValueError("temporal route inputs do not form a passing stage")
    rejected_history = []
    for relative in _REJECTED_HISTORY:
        path = evidence_root / relative
        payload = _load(path)
        if not str(payload.get("status", "")).startswith("REJECTED_"):
            raise ValueError("temporal route failure history contains a non-rejection")
        rejected_history.append(
            {
                "path": relative,
                "status": payload["status"],
                "report_hash": payload["report_hash"],
                "file_hash": hash_bytes(path.read_bytes()),
            }
        )
    stage: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.temporal_route_growth_stage.v1",
        "status": "PASS_TEMPORAL_MULTI_AGENT_GROWTH_STAGE",
        "source_commit": source_commit,
        "source_snapshot_hash": source["snapshot_hash"],
        "source_snapshot_summary_hash": source["summary_hash"],
        "acquisition_report_hash": acquisition["report_hash"],
        "development_report_path": str(development_path.relative_to(evidence_root)),
        "development_report_hash": development["report_hash"],
        "development_report_file_hash": hash_bytes(development_path.read_bytes()),
        "actor_hash": actor.actor_hash,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "retention_name": retention_name,
        "sealed_retention_manifest_hash": manifest["manifest_hash"],
        "retention_report_hash": retention["report_hash"],
        "retention_report_file_hash": hash_bytes(retention_path.read_bytes()),
        "rejected_history": rejected_history,
        "implementation_hashes": _implementation_hashes(checkout),
        "metrics": retention["metrics"],
        "evidence_boundary": {
            "physics_authority": "CPU_MUJOCO",
            "four_gpu_training_only": True,
            "whole_body_g1_count": 3,
            "shared_solver_and_ball": True,
            "temporal_memory_reset_per_episode": True,
            "predictive_reception_reflex_safety_projected": True,
            "route_actor_pose_joint_torque_or_ball_authority": False,
            "retention_used_for_training": False,
            "pixels_used_for_scoring": False,
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
        },
    }
    stage["stage_hash"] = hash_json(stage)
    destination.write_text(
        json.dumps(stage, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return stage


def validate_temporal_route_growth_stage(
    evidence_dir: Path,
    *,
    source_checkout: Path,
) -> dict[str, Any]:
    """Recompute all S123 commitments without executing its training code."""

    root = evidence_dir.expanduser().resolve()
    evidence_root = root.parent
    checkout = source_checkout.expanduser().resolve()
    errors: list[str] = []
    try:
        stage = _load(root / "stage-summary.json")
        source = _load(root / "source-snapshot.json")
        acquisition = _load(root / "population-report.json")
        retention_name = str(stage.get("retention_name"))
        manifest = _load(root / retention_name / "sealed-retention.json")
        retention = _load(root / retention_name / "retention-exam.json")
        actor_path = root / "temporal-route-champion.json"
        actor = load_temporal_route_actor(actor_path)
        development_path = _safe_relative(evidence_root, str(stage.get("development_report_path")))
        development = _load(development_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "status": "REJECTED_TEMPORAL_MULTI_AGENT_STAGE",
            "errors": [f"load_error:{type(exc).__name__}:{exc}"],
        }

    for payload, key, label in (
        (stage, "stage_hash", "stage"),
        (source, "summary_hash", "source_summary"),
        (acquisition, "report_hash", "acquisition"),
        (development, "report_hash", "development"),
        (manifest, "manifest_hash", "manifest"),
        (retention, "report_hash", "retention"),
    ):
        _check_commitment(payload, key, label, errors)
    retention_path = root / retention_name / "retention-exam.json"
    if not (
        stage.get("status") == "PASS_TEMPORAL_MULTI_AGENT_GROWTH_STAGE"
        and stage.get("source_snapshot_hash") == source.get("snapshot_hash")
        and stage.get("source_snapshot_summary_hash") == source.get("summary_hash")
        and stage.get("acquisition_report_hash") == acquisition.get("report_hash")
        and stage.get("development_report_hash") == development.get("report_hash")
        and stage.get("development_report_file_hash") == hash_bytes(development_path.read_bytes())
        and stage.get("sealed_retention_manifest_hash") == manifest.get("manifest_hash")
        and stage.get("retention_report_hash") == retention.get("report_hash")
        and stage.get("retention_report_file_hash") == hash_bytes(retention_path.read_bytes())
    ):
        errors.append("stage_report_binding_mismatch")
    if not (
        actor.actor_hash
        == stage.get("actor_hash")
        == acquisition.get("champion_hash")
        == retention.get("temporal_actor_hash")
        and hash_bytes(actor_path.read_bytes())
        == stage.get("actor_file_hash")
        == acquisition.get("champion_file_hash")
        == retention.get("temporal_actor_file_hash")
    ):
        errors.append("actor_binding_mismatch")
    if stage.get("implementation_hashes") != _implementation_hashes(checkout):
        errors.append("implementation_hash_mismatch")
    _validate_failure_history(stage, evidence_root, errors)
    _validate_retention_rows(root / retention_name, manifest, retention, errors)
    gates = retention.get("gates")
    metrics = retention.get("metrics")
    if not (
        acquisition.get("status") == "PASS_TEMPORAL_ROUTE_ACQUISITION"
        and isinstance(acquisition.get("gates"), dict)
        and all(value is True for value in acquisition["gates"].values())
        and retention.get("status") == "PASS_TEMPORAL_MULTI_AGENT_GROWTH"
        and isinstance(gates, dict)
        and gates
        and all(value is True for value in gates.values())
        and isinstance(metrics, dict)
        and metrics.get("case_count") == 8
        and metrics.get("temporal_qualified_rate") == 1.0
        and metrics.get("temporal_safe_rate") == 1.0
        and metrics.get("exact_replay_rate") == 1.0
        and metrics.get("selected_action_counts") == {"pass": 4, "shoot": 4}
        and metrics.get("maximum_temporal_robot_contact_steps") == 0
        and float(metrics.get("maximum_temporal_reception_reflex_torque_nm", 1e9)) <= 20.0 + 1.0e-8
        and float(metrics.get("maximum_temporal_velocity_braking_correction_mps", 1e9))
        <= 0.18 + 1.0e-8
    ):
        errors.append("growth_gate_invalid")
    boundary = stage.get("evidence_boundary")
    if not isinstance(boundary, dict) or not (
        boundary.get("physics_authority") == "CPU_MUJOCO"
        and boundary.get("four_gpu_training_only") is True
        and boundary.get("whole_body_g1_count") == 3
        and boundary.get("shared_solver_and_ball") is True
        and boundary.get("temporal_memory_reset_per_episode") is True
        and boundary.get("predictive_reception_reflex_safety_projected") is True
        and boundary.get("route_actor_pose_joint_torque_or_ball_authority") is False
        and boundary.get("retention_used_for_training") is False
        and boundary.get("pixels_used_for_scoring") is False
        and boundary.get("activation_ceiling") == "SIM_ONLY"
        and boundary.get("hardware_command_sent") is False
    ):
        errors.append("authority_boundary_invalid")
    try:
        ancestor = subprocess.run(
            ("git", "merge-base", "--is-ancestor", str(stage.get("source_commit")), "HEAD"),
            cwd=checkout,
            check=False,
        )
        if ancestor.returncode != 0:
            errors.append("source_commit_not_in_history")
    except OSError:
        errors.append("source_commit_unverifiable")
    return {
        "status": (
            "VALIDATED_TEMPORAL_MULTI_AGENT_STAGE"
            if not errors
            else "REJECTED_TEMPORAL_MULTI_AGENT_STAGE"
        ),
        "errors": errors,
        "stage_hash": stage.get("stage_hash"),
        "actor_hash": actor.actor_hash,
        "retention_report_hash": retention.get("report_hash"),
    }


def _validate_failure_history(
    stage: dict[str, Any], evidence_root: Path, errors: list[str]
) -> None:
    rows = stage.get("rejected_history")
    if not isinstance(rows, list) or len(rows) != len(_REJECTED_HISTORY):
        errors.append("failure_history_invalid")
        return
    for index, (row, expected_relative) in enumerate(zip(rows, _REJECTED_HISTORY, strict=True)):
        if not isinstance(row, dict) or row.get("path") != expected_relative:
            errors.append(f"failure_{index}_path_invalid")
            continue
        try:
            path = _safe_relative(evidence_root, expected_relative)
            payload = _load(path)
            _check_commitment(payload, "report_hash", f"failure_{index}", errors)
            if not (
                str(payload.get("status", "")).startswith("REJECTED_")
                and row.get("status") == payload.get("status")
                and row.get("report_hash") == payload.get("report_hash")
                and row.get("file_hash") == hash_bytes(path.read_bytes())
            ):
                errors.append(f"failure_{index}_binding_invalid")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            errors.append(f"failure_{index}_unreadable")


def _validate_retention_rows(
    retention_dir: Path,
    manifest: dict[str, Any],
    retention: dict[str, Any],
    errors: list[str],
) -> None:
    rows = retention.get("rows")
    case_hashes = manifest.get("case_hashes")
    if not isinstance(rows, list) or len(rows) != 8 or not isinstance(case_hashes, list):
        errors.append("retention_rows_invalid")
        return
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("case_hash") != case_hashes[index]:
            errors.append(f"row_{index}_case_binding_invalid")
            continue
        digests: dict[str, str] = {}
        for label in ("temporal_artifact", "replay_artifact", "parent_artifact"):
            artifact = row.get(label)
            if not isinstance(artifact, dict):
                errors.append(f"row_{index}_{label}_invalid")
                continue
            path = retention_dir / f"case-{index:03d}" / str(artifact.get("file"))
            try:
                with np.load(path, allow_pickle=False) as archive:
                    trajectory = {name: np.asarray(archive[name]) for name in archive.files}
                digest = trajectory_digest(trajectory)
                digests[label] = digest
                if not (
                    hash_bytes(path.read_bytes()) == artifact.get("file_hash")
                    and digest == artifact.get("trajectory_digest")
                ):
                    errors.append(f"row_{index}_{label}_hash_mismatch")
            except (OSError, ValueError):
                errors.append(f"row_{index}_{label}_unreadable")
        temporal = row.get("temporal")
        if not (
            row.get("temporal_qualified") is True
            and row.get("temporal_safe") is True
            and row.get("exact_replay") is True
            and isinstance(temporal, dict)
            and temporal.get("qualified") is True
            and temporal.get("movement_quality_passed") is True
            and isinstance(temporal.get("base"), dict)
            and temporal["base"].get("task_succeeded") is True
            and temporal["base"].get("safe") is True
            and digests.get("temporal_artifact") == digests.get("replay_artifact")
        ):
            errors.append(f"row_{index}_qualification_invalid")


def _implementation_hashes(checkout: Path) -> dict[str, str]:
    package = checkout / "src/rosclaw_soccer"
    files = {
        "temporal_actor": package / "growth/temporal_route_actor.py",
        "temporal_growth": package / "training/temporal_route_growth.py",
        "temporal_validation": package / "training/temporal_route_validation.py",
        "reactive_growth": package / "training/reactive_route_growth.py",
        "shared_world": package / "skills/team/shared_world.py",
        "full_body_bridge": package / "training/full_body_tactical_2v1.py",
        "first_touch_interception": package / "growth/first_touch_interception.py",
    }
    return {name: hash_bytes(path.read_bytes()) for name, path in files.items()}


def _check_commitment(payload: dict[str, Any], key: str, label: str, errors: list[str]) -> None:
    expected = payload.get(key)
    body = dict(payload)
    body.pop(key, None)
    if expected != hash_json(body):
        errors.append(f"{label}_hash_mismatch")


def _safe_relative(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError("evidence path escapes its root")
    return path


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


__all__ = [
    "validate_temporal_route_growth_stage",
    "write_temporal_route_stage_summary",
]
