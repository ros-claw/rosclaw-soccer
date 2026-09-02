"""Independent integrity validator for the S121 active off-ball stage."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.active_off_ball_growth import ActiveRouteCandidate


def validate_active_off_ball_growth_stage(
    evidence_dir: Path,
    *,
    source_checkout: Path,
) -> dict[str, Any]:
    """Recompute durable commitments and reject edited or incomplete evidence."""

    root = evidence_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    errors: list[str] = []
    try:
        manifest = _load_object(root / "sealed-retention.json")
        acquisition = _load_object(root / "acquisition-ledger.json")
        retention = _load_object(root / "retention/retention-exam.json")
        stage = _load_object(root / "stage-summary.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "REJECTED_ACTIVE_OFF_BALL_STAGE",
            "errors": [f"load_error:{type(exc).__name__}:{exc}"],
        }

    _check_commitment(manifest, "manifest_hash", "manifest", errors)
    _check_commitment(acquisition, "ledger_hash", "acquisition", errors)
    _check_commitment(retention, "report_hash", "retention", errors)
    _check_commitment(stage, "stage_hash", "stage", errors)
    if (
        manifest.get("training_access_allowed") is not False
        or manifest.get("activation_ceiling") != "SIM_ONLY"
        or acquisition.get("retention_manifest_visible_to_training") is not False
        or stage.get("sealed_retention_manifest_hash") != manifest.get("manifest_hash")
    ):
        errors.append("sealed_boundary_invalid")

    try:
        candidate_payload = retention.get("selected_candidate")
        if not isinstance(candidate_payload, dict):
            raise ValueError("selected candidate is absent")
        candidate = ActiveRouteCandidate(**candidate_payload)
        candidate_hash = candidate.candidate_hash
        if not (
            candidate_hash
            == retention.get("selected_candidate_hash")
            == stage.get("selected_candidate_hash")
            == acquisition.get("selected_candidate_hash")
        ):
            errors.append("selected_candidate_hash_mismatch")
    except (TypeError, ValueError):
        errors.append("selected_candidate_invalid")

    implementation = stage.get("implementation_hashes")
    package = Path(__file__).parents[1]
    expected_implementation = {
        "shared_world": hash_bytes(package.joinpath("skills/team/shared_world.py").read_bytes()),
        "full_body_bridge": hash_bytes(
            Path(__file__).with_name("full_body_tactical_2v1.py").read_bytes()
        ),
        "active_growth": hash_bytes(
            Path(__file__).with_name("active_off_ball_growth.py").read_bytes()
        ),
    }
    if implementation != expected_implementation:
        errors.append("implementation_hash_mismatch")
    if (
        stage.get("retention_report_hash") != retention.get("report_hash")
        or stage.get("actor_hash") != retention.get("actor_hash")
        or stage.get("actor_hash") != acquisition.get("actor_hash")
    ):
        errors.append("stage_binding_mismatch")

    rows = retention.get("rows")
    if not isinstance(rows, list) or len(rows) != 8:
        errors.append("retention_rows_invalid")
    else:
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"row_{index}_invalid")
                continue
            artifacts: dict[str, dict[str, str]] = {}
            for key in ("primary_artifact", "replay_artifact"):
                artifact = row.get(key)
                if not isinstance(artifact, dict):
                    errors.append(f"row_{index}_{key}_invalid")
                    continue
                path = root / "retention" / f"case-{index:03d}" / str(artifact.get("file"))
                try:
                    with np.load(path, allow_pickle=False) as archive:
                        trajectory = {name: np.asarray(archive[name]) for name in archive.files}
                    if hash_bytes(path.read_bytes()) != artifact.get(
                        "file_hash"
                    ) or trajectory_digest(trajectory) != artifact.get("trajectory_digest"):
                        errors.append(f"row_{index}_{key}_hash_mismatch")
                    artifacts[key] = artifact
                except (OSError, ValueError):
                    errors.append(f"row_{index}_{key}_unreadable")
            primary = artifacts.get("primary_artifact", {})
            replay = artifacts.get("replay_artifact", {})
            result = row.get("result")
            if (
                primary.get("trajectory_digest") != replay.get("trajectory_digest")
                or row.get("exact_replay") is not True
                or row.get("qualified") is not True
                or row.get("safe") is not True
                or not isinstance(result, dict)
                or result.get("movement_quality_passed") is not True
                or not isinstance(result.get("base"), dict)
                or result["base"].get("task_succeeded") is not True
                or result["base"].get("safe") is not True
            ):
                errors.append(f"row_{index}_qualification_invalid")

    metrics = retention.get("metrics")
    gates = retention.get("gates")
    if (
        retention.get("status") != "PASS_ACTIVE_OFF_BALL_GROWTH"
        or not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
        or not isinstance(metrics, dict)
        or metrics.get("case_count") != 8
        or metrics.get("qualified_rate") != 1.0
        or metrics.get("task_success_rate") != 1.0
        or metrics.get("safe_rate") != 1.0
        or metrics.get("movement_quality_rate") != 1.0
        or metrics.get("exact_replay_rate") != 1.0
    ):
        errors.append("retention_gate_invalid")
    boundary = retention.get("evidence_boundary")
    if not isinstance(boundary, dict) or not (
        boundary.get("activation_ceiling") == "SIM_ONLY"
        and boundary.get("physics_authority") == "CPU_MUJOCO"
        and boundary.get("whole_body_g1_count") == 3
        and boundary.get("shared_solver_and_ball") is True
        and boundary.get("movement_executed_by_frozen_neural_locomotion") is True
        and boundary.get("pose_or_joint_scripted") is False
        and boundary.get("pixels_used_for_scoring") is False
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
            "VALIDATED_ACTIVE_OFF_BALL_STAGE" if not errors else "REJECTED_ACTIVE_OFF_BALL_STAGE"
        ),
        "errors": errors,
        "stage_hash": stage.get("stage_hash"),
        "retention_report_hash": retention.get("report_hash"),
        "selected_candidate_hash": retention.get("selected_candidate_hash"),
    }


def _check_commitment(
    payload: dict[str, Any],
    key: str,
    label: str,
    errors: list[str],
) -> None:
    claimed = payload.get(key)
    body = dict(payload)
    body.pop(key, None)
    if claimed != hash_json(body):
        errors.append(f"{label}_hash_mismatch")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


__all__ = ["validate_active_off_ball_growth_stage"]
