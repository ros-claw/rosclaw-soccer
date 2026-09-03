"""Distil measured runtime RECEIVE trajectories into an immutable actor."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.runtime_receive_actor import (
    RUNTIME_RECEIVE_FEATURE_RESOLUTION,
    G1RuntimeReceiveActor,
    RuntimeReceiveAction,
    RuntimeReceiveMemory,
    runtime_receive_features,
    save_runtime_receive_actor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def extract_runtime_receive_features(trajectory_path: Path) -> tuple[float, ...]:
    """Read the first stable incoming-ball observation from a bound trajectory."""

    with np.load(trajectory_path.expanduser().resolve(), allow_pickle=False) as trajectory:
        count = np.asarray(
            trajectory["shooter_causal_strike_option_incoming_observation_count"],
            dtype=np.int64,
        )
        indices = np.flatnonzero(count >= 5)
        if indices.size == 0:
            raise ValueError("runtime RECEIVE trajectory lacks a stable incoming observation")
        index = int(indices[0])
        eta = float(trajectory["shooter_causal_strike_option_ball_arrival_eta_sec"][index])
        if eta < 0.0:
            raise ValueError("runtime RECEIVE trajectory has no measured arrival ETA")
        return runtime_receive_features(
            ball_local_position_m=trajectory["shooter_ball_local_position"][index],
            ball_local_velocity_mps=trajectory["shooter_ball_local_velocity"][index],
            ball_arrival_eta_sec=eta,
            pelvis_local_position_m=trajectory["shooter_pelvis_local_position"][index],
            joint_velocity_rad_s=trajectory["shooter_joint_velocity"][index],
            policy_frame=int(trajectory["shooter_policy_frame"][index]),
        )


def train_runtime_receive_actor(
    *,
    discovery_report_path: Path,
    output_dir: Path,
    additional_report_paths: tuple[Path, ...] = (),
) -> tuple[G1RuntimeReceiveActor, dict[str, Any]]:
    primary_source = discovery_report_path.expanduser().resolve()
    discovery = _bound_discovery(primary_source)
    if discovery.get("status") != "PASS_RUNTIME_RECEIVE_DISCOVERY":
        raise ValueError("runtime RECEIVE training requires passing discovery")
    sources = [(primary_source, discovery)]
    for path in additional_report_paths:
        source = path.expanduser().resolve()
        sources.append((source, _bound_repair(source)))
    request = _read_object(primary_source.parent / "request.json")
    if (
        request.get("body_hash") != discovery.get("body_hash")
        or request.get("kick_prior_hash") != discovery.get("kick_prior_hash")
        or request.get("roster_hash") != discovery.get("roster_hash")
        or request.get("finisher_self_model_hash") != discovery.get("finisher_self_model_hash")
    ):
        raise ValueError("runtime RECEIVE discovery identity changed")
    rows: list[dict[str, Any]] = []
    source_roots: list[Path] = []
    for source, source_report in sources:
        if (
            source_report.get("body_hash") != discovery.get("body_hash")
            or source_report.get("kick_prior_hash") != discovery.get("kick_prior_hash")
            or source_report.get("roster_hash") != discovery.get("roster_hash")
            or source_report.get("finisher_self_model_hash")
            != discovery.get("finisher_self_model_hash")
        ):
            raise ValueError("runtime RECEIVE training sources disagree on identity")
        source_rows = source_report.get("rows")
        if not isinstance(source_rows, list) or len(source_rows) < 8:
            raise ValueError("runtime RECEIVE source has too few trajectories")
        if not all(row.get("runtime_intervention_observed") is True for row in source_rows):
            raise ValueError("runtime RECEIVE source contains pre-action leakage")
        rows.extend(cast(list[dict[str, Any]], source_rows))
        source_roots.extend([source.parent] * len(source_rows))
    memories: list[RuntimeReceiveMemory] = []
    all_features: list[tuple[float, ...]] = []
    for row, source_root in zip(rows, source_roots, strict=True):
        artifact = row["trajectory"]
        trajectory_path = source_root / artifact["file"]
        if hash_bytes(trajectory_path.read_bytes()) != artifact["file_hash"]:
            raise ValueError("runtime RECEIVE trajectory binding changed")
        features = extract_runtime_receive_features(trajectory_path)
        action = RuntimeReceiveAction(**row["action"])
        quality = cast(dict[str, Any], row["quality"])
        result = cast(dict[str, Any], row["result"])
        quality_score = float(
            4.0 * bool(row["strict_team_chain"])
            + bool(quality["safe"])
            + bool(quality["clear_outcome"])
            + bool(quality["intended_foot_contact"])
            + 3.0 * min(float(result["shot_peak_ball_speed_mps"]) / 10.0, 1.0)
        )
        memories.append(
            RuntimeReceiveMemory(
                context_hash=row["context_hash"],
                trajectory_hash=artifact["trajectory_digest"],
                features=features,
                action=action,
                quality_score=quality_score,
            )
        )
        all_features.append(features)
    successful = tuple(
        memory for memory, row in zip(memories, rows, strict=True) if row["strict_team_chain"]
    )
    failed = tuple(
        memory for memory, row in zip(memories, rows, strict=True) if not row["strict_team_chain"]
    )
    if len(successful) < 2 or len(failed) < 2:
        raise ValueError("runtime RECEIVE training needs both success and failure memory")
    matrix = np.asarray(all_features, dtype=np.float64)
    center = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.maximum(scale, np.asarray(RUNTIME_RECEIVE_FEATURE_RESOLUTION, dtype=np.float64))
    snapshot = {
        "source_report_hashes": [item["report_hash"] for _, item in sources],
        "trajectory_hashes": [memory.trajectory_hash for memory in memories],
        "features": [list(memory.features) for memory in memories],
        "actions": [asdict(memory.action) for memory in memories],
        "success_labels": [bool(row["strict_team_chain"]) for row in rows],
    }
    actor = G1RuntimeReceiveActor(
        body_hash=discovery["body_hash"],
        kick_prior_hash=discovery["kick_prior_hash"],
        roster_hash=discovery["roster_hash"],
        finisher_self_model_hash=discovery["finisher_self_model_hash"],
        source_evidence_hashes=tuple(item["report_hash"] for _, item in sources),
        training_snapshot_hash=hash_json(snapshot),
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
        successful_memories=successful,
        failed_memories=failed,
    )
    output = _new_external_output(output_dir)
    actor_path = output / "runtime-receive-actor.json"
    save_runtime_receive_actor(actor, actor_path)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_receive_training.v1",
        "status": "PASS_RUNTIME_RECEIVE_TRAINING",
        "promotion_eligible": False,
        "request_hashes": [item["request_hash"] for _, item in sources],
        "source_report_hashes": [item["report_hash"] for _, item in sources],
        "training_snapshot_hash": actor.training_snapshot_hash,
        "actor_hash": actor.actor_hash,
        "actor_file": actor_path.name,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "trajectory_count": len(memories),
        "successful_memory_count": len(successful),
        "failed_memory_count": len(failed),
        "role_authority": {
            "agent_id": actor.agent_id,
            "primary_role": actor.primary_role,
            "tactical_intent": actor.tactical_intent,
            "owned_skill": actor.owned_skill,
            "roster_hash": actor.roster_hash,
            "self_model_hash": actor.finisher_self_model_hash,
        },
        "decision_clock": "FIFTH_CONSECUTIVE_MEASURED_INCOMING_BALL_OBSERVATION",
        "control_clock": "EVERY_20MS_CAUSAL_OPTION_FRAME_AFTER_LATCH",
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "pixels_used_for_training": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "training-report.json", report)
    return actor, report


def _bound_discovery(path: Path) -> dict[str, Any]:
    report = _read_object(path)
    claimed = report.pop("report_hash", None)
    if (
        claimed != hash_json(report)
        or report.get("schema_version") != "rosclaw_soccer.runtime_receive_discovery.v2"
        or report.get("gates", {}).get("runtime_intervention_observed_all") is not True
        or report.get("intervention_contract", {}).get("observation_precedes_action") is not True
        or not _is_sha256(report.get("implementation_hash"))
        or report.get("physics_authority") != "CPU_MUJOCO"
        or report.get("activation_ceiling") != "SIM_ONLY"
        or report.get("hardware_command_sent") is not False
    ):
        raise ValueError("runtime RECEIVE discovery authority is invalid")
    report["report_hash"] = claimed
    request = path.parent / "request.json"
    if not request.is_file() or hash_bytes(request.read_bytes()) != report["request_hash"]:
        raise ValueError("runtime RECEIVE discovery request changed")
    return report


def _bound_repair(path: Path) -> dict[str, Any]:
    report = _read_object(path)
    claimed = report.pop("report_hash", None)
    if (
        claimed != hash_json(report)
        or report.get("schema_version") != "rosclaw_soccer.runtime_receive_repair.v2"
        or report.get("gates", {}).get("runtime_intervention_observed_all") is not True
        or report.get("intervention_contract", {}).get("observation_precedes_action") is not True
        or report.get("status")
        not in {
            "PASS_RUNTIME_RECEIVE_REPAIR_DATA",
            "REJECTED_RUNTIME_RECEIVE_REPAIR_DATA",
        }
        or not _is_sha256(report.get("implementation_hash"))
        or report.get("physics_authority") != "CPU_MUJOCO"
        or report.get("activation_ceiling") != "SIM_ONLY"
        or report.get("hardware_command_sent") is not False
    ):
        raise ValueError("runtime RECEIVE repair authority is invalid")
    report["report_hash"] = claimed
    request = path.parent / "request.json"
    if not request.is_file() or hash_bytes(request.read_bytes()) != report["request_hash"]:
        raise ValueError("runtime RECEIVE repair request changed")
    return report


def _is_sha256(value: object) -> bool:
    """Recognize the implementation identity stored with immutable old evidence."""

    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == len("sha256:") + 64
        and all(character in "0123456789abcdef" for character in value.removeprefix("sha256:"))
    )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("runtime RECEIVE training output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["extract_runtime_receive_features", "train_runtime_receive_actor"]
