"""Distil causal multi-target teacher traces into a perceptive target actor."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.runtime_contact_target_actor import (
    G1RuntimeContactTargetActor,
    RuntimeContactTargetAction,
    RuntimeContactTargetMemory,
    save_runtime_contact_target_actor,
)
from rosclaw_soccer.growth.runtime_receive_actor import (
    RUNTIME_RECEIVE_FEATURE_RESOLUTION,
    RuntimeReceiveAction,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.neural_contact_canary import _bound_json
from rosclaw_soccer.training.runtime_receive_growth import extract_runtime_receive_features


def train_runtime_contact_target_actor(
    *,
    teacher_report_path: Path,
    neural_training_report_path: Path,
    neural_actor_path: Path,
    role_source_report_path: Path,
    output_dir: Path,
    additional_report_paths: tuple[Path, ...] = (),
) -> tuple[G1RuntimeContactTargetActor, dict[str, Any]]:
    """Learn target choice from pre-intervention features and strict outcomes."""

    teacher_path = teacher_report_path.expanduser().resolve()
    teacher = _bound_json(teacher_path)
    training_path = neural_training_report_path.expanduser().resolve()
    neural_training = _bound_json(training_path)
    neural_path = neural_actor_path.expanduser().resolve()
    neural_actor = load_g1_neural_contact_actor(neural_path)
    role_path = role_source_report_path.expanduser().resolve()
    role_source = _bound_json(role_path)
    rows = teacher.get("rows")
    if (
        teacher.get("schema_version") != "rosclaw_soccer.runtime_receive_contact_teacher.v1"
        or teacher.get("status") != "PASS_RUNTIME_RECEIVE_CONTACT_TEACHER"
        or teacher.get("teacher_role") != "SIM_ONLY_LOW_LEVEL_CONTACT_DATA_GENERATOR"
        or not isinstance(rows, list)
        or len(rows) != 16
        or neural_training.get("status") != "PASS_NEURAL_CONTACT_DISTILLATION"
        or neural_training.get("source_teacher_report_hash") != teacher.get("report_hash")
        or neural_training.get("actor_hash") != neural_actor.actor_hash
        or neural_training.get("actor_file_hash") != hash_bytes(neural_path.read_bytes())
        or neural_training.get("metrics", {}).get("specialist_target_velocity_xyz_mps") is not None
        or role_source.get("report_hash") != teacher.get("rejected_direction_report_hash")
        or role_source.get("roster_hash") is None
        or role_source.get("finisher_self_model_hash") is None
        or teacher.get("body_hash") != neural_actor.body_hash
        or teacher.get("body_hash") != role_source.get("body_hash")
        or teacher.get("kick_prior_hash") != role_source.get("kick_prior_hash")
    ):
        raise ValueError("runtime contact target training lineage changed")

    sources: list[tuple[Path, dict[str, Any]]] = [(teacher_path, teacher)]
    for path in additional_report_paths:
        source_path = path.expanduser().resolve()
        source = _bound_json(source_path)
        if (
            source.get("schema_version") != "rosclaw_soccer.runtime_contact_target_repair.v1"
            or source.get("status")
            not in {
                "PASS_RUNTIME_CONTACT_TARGET_REPAIR_DATA",
                "REJECTED_RUNTIME_CONTACT_TARGET_REPAIR_DATA",
            }
            or source.get("promotion_eligible") is not False
            or source.get("neural_actor_hash") != neural_actor.actor_hash
            or source.get("body_hash") != teacher.get("body_hash")
            or source.get("kick_prior_hash") != teacher.get("kick_prior_hash")
            or source.get("roster_hash") != role_source.get("roster_hash")
            or source.get("finisher_self_model_hash") != role_source.get("finisher_self_model_hash")
            or source.get("causal_contract", {}).get(
                "target_effect_starts_after_pre_action_observation"
            )
            is not True
            or source.get("causal_contract", {}).get("same_pre_action_features_across_targets")
            is not True
        ):
            raise ValueError("runtime contact target repair authority is invalid")
        sources.append((source_path, source))

    memories: list[RuntimeContactTargetMemory] = []
    labels: list[bool] = []
    feature_by_context: dict[str, tuple[float, ...]] = {}
    required_receive_action: RuntimeReceiveAction | None = None
    source_rows: list[tuple[Path, dict[str, Any]]] = []
    for source_path, source in sources:
        request_path = source_path.parent / "request.json"
        source_values = source.get("rows")
        if (
            not request_path.is_file()
            or hash_bytes(request_path.read_bytes()) != source.get("request_hash")
            or not isinstance(source_values, list)
        ):
            raise ValueError("runtime contact target source request changed")
        source_rows.extend(
            (source_path.parent, row) for row in cast(list[dict[str, Any]], source_values)
        )
    for source_root, row in source_rows:
        artifact = cast(dict[str, Any], row["trajectory"])
        trajectory_path = source_root / str(artifact["file"])
        if hash_bytes(trajectory_path.read_bytes()) != artifact["file_hash"]:
            raise ValueError("runtime contact target source trajectory changed")
        features = extract_runtime_receive_features(trajectory_path)
        context_hash = str(row["context_hash"])
        previous = feature_by_context.setdefault(context_hash, features)
        if not np.allclose(previous, features, rtol=0.0, atol=1.0e-12):
            raise ValueError("teacher target contaminated its pre-action observation")
        action_payload = cast(dict[str, Any], row["action"])
        if "maximum_arrival_advance_frames" in action_payload:
            receive_action = RuntimeReceiveAction(
                maximum_arrival_advance_frames=int(
                    action_payload["maximum_arrival_advance_frames"]
                ),
                arrival_alignment_tolerance_sec=0.02,
                stance_offset_x_m=float(action_payload["stance_offset_x_m"]),
                stance_offset_y_m=float(action_payload["stance_offset_y_m"]),
                contact_policy_frame=int(action_payload["contact_policy_frame"]),
                foot_yaw_offset_rad=float(action_payload["foot_yaw_offset_rad"]),
                foot_pitch_offset_rad=float(action_payload["foot_pitch_offset_rad"]),
            )
            if required_receive_action is None:
                required_receive_action = receive_action
            elif receive_action != required_receive_action:
                raise ValueError("target curriculum changed the upper-layer RECEIVE law")
        target = cast(
            tuple[float, float, float],
            tuple(float(value) for value in action_payload["target_foot_velocity_xyz_mps"]),
        )
        if not neural_actor.target_supported(target):
            raise ValueError("target curriculum exceeds neural contact support")
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            contacts = np.asarray(trajectory["shooter_ball_contact_foot"], dtype=np.int64)
            observed = contacts[contacts != 0]
            intended = bool(observed.size and int(observed[0]) == 1)
        quality = cast(dict[str, Any], row["quality"])
        result = cast(dict[str, Any], row["result"])
        strict = bool(quality.get("strict_chain_passed", quality["chain_passed"] and intended))
        clear = bool(result["goal_crossed"] or result["goalkeeper_save_observed"])
        quality_score = float(
            4.0 * strict
            + bool(quality["safe"])
            + clear
            + intended
            + 3.0 * min(float(result["shot_peak_ball_speed_mps"]) / 10.0, 1.0)
            + bool(result["goal_crossed"])
        )
        memories.append(
            RuntimeContactTargetMemory(
                context_hash=context_hash,
                trajectory_hash=str(artifact["trajectory_digest"]),
                features=features,
                action=RuntimeContactTargetAction(target),
                quality_score=quality_score,
            )
        )
        labels.append(strict)
    if required_receive_action is None or len(feature_by_context) < 2:
        raise ValueError("runtime contact target training needs causal context diversity")
    successful = tuple(memory for memory, passed in zip(memories, labels, strict=True) if passed)
    failed = tuple(memory for memory, passed in zip(memories, labels, strict=True) if not passed)
    if len(successful) < 4 or len(failed) < 2:
        raise ValueError("runtime contact target training lacks success/failure contrast")
    matrix = np.asarray(tuple(feature_by_context.values()), dtype=np.float64)
    center = np.mean(matrix, axis=0)
    scale = np.maximum(
        np.std(matrix, axis=0),
        np.asarray(RUNTIME_RECEIVE_FEATURE_RESOLUTION, dtype=np.float64),
    )
    snapshot = {
        "teacher_report_hash": teacher["report_hash"],
        "additional_report_hashes": [source["report_hash"] for _, source in sources[1:]],
        "neural_training_report_hash": neural_training["report_hash"],
        "neural_contact_actor_hash": neural_actor.actor_hash,
        "role_source_report_hash": role_source["report_hash"],
        "trajectory_hashes": [memory.trajectory_hash for memory in memories],
        "features": [list(memory.features) for memory in memories],
        "target_actions": [asdict(memory.action) for memory in memories],
        "strict_success_labels": labels,
        "observation_precedes_target_intervention": True,
    }
    actor = G1RuntimeContactTargetActor(
        body_hash=str(teacher["body_hash"]),
        kick_prior_hash=str(teacher["kick_prior_hash"]),
        roster_hash=str(role_source["roster_hash"]),
        finisher_self_model_hash=str(role_source["finisher_self_model_hash"]),
        neural_contact_actor_hash=neural_actor.actor_hash,
        source_evidence_hashes=tuple(
            str(value)
            for value in (
                teacher["report_hash"],
                neural_training["report_hash"],
                role_source["report_hash"],
                *(source["report_hash"] for _, source in sources[1:]),
            )
        ),
        training_snapshot_hash=str(hash_json(snapshot)),
        required_receive_action=required_receive_action,
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
        successful_memories=successful,
        failed_memories=failed,
    )
    output = _new_external_output(output_dir)
    actor_path = output / "runtime-contact-target-actor.json"
    save_runtime_contact_target_actor(actor, actor_path)
    selected_targets = {
        context_hash: actor.decide(features).action
        for context_hash, features in feature_by_context.items()
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_contact_target_training.v1",
        "status": "PASS_RUNTIME_CONTACT_TARGET_TRAINING",
        "promotion_eligible": False,
        "teacher_report_hash": teacher["report_hash"],
        "additional_report_hashes": [source["report_hash"] for _, source in sources[1:]],
        "neural_training_report_hash": neural_training["report_hash"],
        "role_source_report_hash": role_source["report_hash"],
        "training_snapshot_hash": actor.training_snapshot_hash,
        "actor_hash": actor.actor_hash,
        "actor_file": actor_path.name,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "metrics": {
            "trajectory_count": len(memories),
            "context_cluster_count": len(feature_by_context),
            "strict_successful_memory_count": len(successful),
            "failed_memory_count": len(failed),
            "selected_targets_by_context": {
                context_hash: (
                    None if action is None else list(action.target_foot_velocity_xyz_mps)
                )
                for context_hash, action in selected_targets.items()
            },
        },
        "causal_contract": {
            "observation_precedes_target_intervention": True,
            "same_pre_action_features_across_targets": True,
            "decision_clock": "FIFTH_CONSECUTIVE_MEASURED_INCOMING_BALL_OBSERVATION",
            "action_authority": "TASK_SPACE_TARGET_VELOCITY_ONLY",
            "joint_torque_actor_hash": neural_actor.actor_hash,
        },
        "activation_ceiling": "SIM_ONLY",
        "physics_authority": "CPU_MUJOCO",
        "pixels_used_for_training": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "training-report.json", report)
    return actor, report


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("runtime contact target output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["train_runtime_contact_target_actor"]
