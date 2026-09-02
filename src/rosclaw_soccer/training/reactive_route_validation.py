"""Independent integrity validation for the S122 reactive-route stage."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.reactive_route_actor import load_reactive_route_actor
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def validate_reactive_route_growth_stage(
    evidence_dir: Path,
    *,
    source_checkout: Path,
) -> dict[str, Any]:
    root = evidence_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    errors: list[str] = []
    try:
        manifest = _load(root / "sealed-retention-v2.json")
        acquisition = _load(root / "acquisition-ledger.json")
        cross_validation = _load(root / "cross-validation.json")
        sensitivity = _load(root / "counterfactual-sensitivity.json")
        development = _load(root / "champion-v2-development-exam.json")
        rejected = _load(root / "retention/retention-exam.json")
        retention = _load(root / "retention-v2/retention-exam.json")
        stage = _load(root / "stage-summary.json")
        actor = load_reactive_route_actor(root / "reactive-route-champion-v2.json")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "status": "REJECTED_REACTIVE_MULTI_AGENT_STAGE",
            "errors": [f"load_error:{type(exc).__name__}:{exc}"],
        }

    for payload, key, label in (
        (manifest, "manifest_hash", "manifest"),
        (acquisition, "ledger_hash", "acquisition"),
        (cross_validation, "report_hash", "cross_validation"),
        (sensitivity, "report_hash", "sensitivity"),
        (development, "report_hash", "development"),
        (rejected, "report_hash", "rejected_retention"),
        (retention, "report_hash", "retention"),
        (stage, "stage_hash", "stage"),
    ):
        _check_commitment(payload, key, label, errors)
    if not (
        manifest.get("training_access_allowed") is False
        and manifest.get("activation_ceiling") == "SIM_ONLY"
        and acquisition.get("retention_v2_manifest_visible_to_training") is False
        and acquisition.get("rejected_retention_v1_trajectory_used_for_training") is False
        and rejected.get("status") == "REJECTED_REACTIVE_MULTI_AGENT_GROWTH"
        and stage.get("sealed_retention_manifest_hash") == manifest.get("manifest_hash")
    ):
        errors.append("sealed_growth_boundary_invalid")
    actor_path = root / "reactive-route-champion-v2.json"
    if not (
        actor.actor_hash
        == retention.get("actor_hash")
        == stage.get("actor_hash")
        == sensitivity.get("actor_hash")
        and hash_bytes(actor_path.read_bytes())
        == retention.get("actor_file_hash")
        == stage.get("actor_file_hash")
    ):
        errors.append("actor_binding_mismatch")
    if not (
        stage.get("acquisition_ledger_hash") == acquisition.get("ledger_hash")
        and stage.get("cross_validation_report_hash") == cross_validation.get("report_hash")
        and stage.get("counterfactual_sensitivity_report_hash") == sensitivity.get("report_hash")
        and stage.get("development_champion_report_hash") == development.get("report_hash")
        and stage.get("rejected_retention_v1_report_hash") == rejected.get("report_hash")
        and stage.get("retention_report_hash") == retention.get("report_hash")
    ):
        errors.append("stage_report_binding_mismatch")

    package = Path(__file__).parents[1]
    expected_implementation = {
        "shared_world": hash_bytes(package.joinpath("skills/team/shared_world.py").read_bytes()),
        "full_body_bridge": hash_bytes(
            Path(__file__).with_name("full_body_tactical_2v1.py").read_bytes()
        ),
        "route_actor": hash_bytes(package.joinpath("growth/reactive_route_actor.py").read_bytes()),
        "reactive_growth": hash_bytes(
            Path(__file__).with_name("reactive_route_growth.py").read_bytes()
        ),
    }
    if stage.get("implementation_hashes") != expected_implementation:
        errors.append("implementation_hash_mismatch")

    rows = retention.get("rows")
    case_hashes = manifest.get("case_hashes")
    if not isinstance(rows, list) or len(rows) != 8 or not isinstance(case_hashes, list):
        errors.append("retention_rows_invalid")
    else:
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or row.get("case_hash") != case_hashes[index]:
                errors.append(f"row_{index}_case_binding_invalid")
                continue
            digests: list[str] = []
            for label in ("primary_artifact", "replay_artifact"):
                artifact = row.get(label)
                if not isinstance(artifact, dict):
                    errors.append(f"row_{index}_{label}_invalid")
                    continue
                path = root / "retention-v2" / f"case-{index:03d}" / str(artifact.get("file"))
                try:
                    with np.load(path, allow_pickle=False) as archive:
                        trajectory = {key: np.asarray(archive[key]) for key in archive.files}
                    digest = trajectory_digest(trajectory)
                    digests.append(digest)
                    if hash_bytes(path.read_bytes()) != artifact.get(
                        "file_hash"
                    ) or digest != artifact.get("trajectory_digest"):
                        errors.append(f"row_{index}_{label}_hash_mismatch")
                except (OSError, ValueError):
                    errors.append(f"row_{index}_{label}_unreadable")
            result = row.get("result")
            if not (
                len(digests) == 2
                and digests[0] == digests[1]
                and row.get("exact_replay") is True
                and row.get("qualified") is True
                and row.get("safe") is True
                and row.get("movement_quality_passed") is True
                and row.get("reactive_actor_accepted") is True
                and isinstance(result, dict)
                and result.get("qualified") is True
                and isinstance(result.get("base"), dict)
                and result["base"].get("task_succeeded") is True
                and result["base"].get("safe") is True
            ):
                errors.append(f"row_{index}_qualification_invalid")

    metrics = retention.get("metrics")
    gates = retention.get("gates")
    if not (
        retention.get("status") == "PASS_REACTIVE_MULTI_AGENT_GROWTH"
        and isinstance(gates, dict)
        and gates
        and all(value is True for value in gates.values())
        and isinstance(metrics, dict)
        and metrics.get("case_count") == 8
        and all(
            metrics.get(key) == 1.0
            for key in (
                "qualified_rate",
                "task_success_rate",
                "safe_rate",
                "movement_quality_rate",
                "reactive_actor_acceptance_rate",
                "exact_replay_rate",
            )
        )
        and metrics.get("selected_action_counts") == {"pass": 4, "shoot": 4}
        and development.get("qualified_rate") == 1.0
        and cross_validation.get("passed") is True
        and sensitivity.get("passed") is True
    ):
        errors.append("growth_gate_invalid")
    boundary = retention.get("evidence_boundary")
    if not isinstance(boundary, dict) or not (
        boundary.get("activation_ceiling") == "SIM_ONLY"
        and boundary.get("physics_authority") == "CPU_MUJOCO"
        and boundary.get("whole_body_g1_count") == 3
        and boundary.get("shared_solver_and_ball") is True
        and boundary.get("route_actor_observation_closed_loop") is True
        and boundary.get("movement_executed_by_frozen_neural_locomotion") is True
        and boundary.get("pose_joint_torque_or_ball_scripted_by_route_actor") is False
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
            "VALIDATED_REACTIVE_MULTI_AGENT_STAGE"
            if not errors
            else "REJECTED_REACTIVE_MULTI_AGENT_STAGE"
        ),
        "errors": errors,
        "stage_hash": stage.get("stage_hash"),
        "actor_hash": actor.actor_hash,
        "retention_report_hash": retention.get("report_hash"),
    }


def _check_commitment(payload: dict[str, Any], key: str, label: str, errors: list[str]) -> None:
    claimed = payload.get(key)
    body = dict(payload)
    body.pop(key, None)
    if claimed != hash_json(body):
        errors.append(f"{label}_hash_mismatch")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


__all__ = ["validate_reactive_route_growth_stage"]
