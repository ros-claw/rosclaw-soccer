"""Consolidate complete failures into a joint high-level finisher plan actor."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.runtime_contact_target_actor import (
    RuntimeContactTargetAction,
    load_runtime_contact_target_actor,
)
from rosclaw_soccer.growth.runtime_finish_plan_actor import (
    G1RuntimeFinishPlanActor,
    RuntimeFinishPlanAction,
    RuntimeFinishPlanMemory,
    load_runtime_finish_plan_actor,
    prepared_finish_plan_features,
    save_runtime_finish_plan_actor,
)
from rosclaw_soccer.growth.runtime_receive_actor import RuntimeReceiveAction
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.causal_transition_growth import _load_lead_policy
from rosclaw_soccer.training.neural_contact_canary import _bound_json

_SOURCE_SCHEMAS = {
    "rosclaw_soccer.runtime_contact_target_repair.v1",
    "rosclaw_soccer.runtime_receive_strike_repair.v1",
    "rosclaw_soccer.runtime_ready_stance_repair.v1",
    "rosclaw_soccer.team_compatible_pass_discovery.v1",
    "rosclaw_soccer.prepared_finish_plan_repair.v1",
    "rosclaw_soccer.prepared_finish_precision_repair.v1",
    "rosclaw_soccer.continuous_finish_plan_repair.v1",
}


def train_runtime_finish_plan_actor(
    *,
    base_target_actor_path: Path,
    neural_actor_path: Path,
    handoff_actor_path: Path,
    source_s95_dir: Path,
    source_report_paths: tuple[Path, ...],
    output_dir: Path,
    parent_finish_plan_actor_path: Path | None = None,
) -> tuple[G1RuntimeFinishPlanActor, dict[str, Any]]:
    """Train only from bound physical outcomes; rejected reports are valid data."""

    base_path = base_target_actor_path.expanduser().resolve()
    base = load_runtime_contact_target_actor(base_path)
    neural_path = neural_actor_path.expanduser().resolve()
    neural = load_g1_neural_contact_actor(neural_path)
    handoff_path = handoff_actor_path.expanduser().resolve()
    handoff = load_contact_handoff_actor(handoff_path)
    if (
        base.neural_contact_actor_hash != neural.actor_hash
        or base.body_hash != neural.body_hash
        or handoff.body_hash != base.body_hash
    ):
        raise ValueError("runtime finish plan neural lineage changed")
    if len(source_report_paths) < 4:
        raise ValueError("runtime finish plan needs heterogeneous failure sources")
    parent = (
        None
        if parent_finish_plan_actor_path is None
        else load_runtime_finish_plan_actor(parent_finish_plan_actor_path.expanduser().resolve())
    )
    if parent is not None and (
        parent.body_hash != base.body_hash
        or parent.kick_prior_hash != base.kick_prior_hash
        or parent.roster_hash != base.roster_hash
        or parent.finisher_self_model_hash != base.finisher_self_model_hash
        or parent.neural_contact_actor_hash != neural.actor_hash
        or parent.contact_handoff_actor_hash != handoff.actor_hash
    ):
        raise ValueError("runtime finish plan parent lineage changed")
    lead, lead_source = _load_lead_policy(source_s95_dir)

    sources: list[tuple[Path, dict[str, Any]]] = []
    for path in source_report_paths:
        report_path = path.expanduser().resolve()
        report = _bound_json(report_path)
        request_path = report_path.parent / "request.json"
        rows = report.get("rows")
        precision_repair = bool(
            report.get("schema_version") == "rosclaw_soccer.prepared_finish_precision_repair.v1"
        )
        if (
            report.get("schema_version") not in _SOURCE_SCHEMAS
            or report.get("promotion_eligible") is not False
            or report.get("body_hash") != base.body_hash
            or report.get("kick_prior_hash") != base.kick_prior_hash
            or report.get("neural_actor_hash", report.get("frozen_neural_actor_hash"))
            != neural.actor_hash
            or report.get("roster_hash") != base.roster_hash
            or report.get("finisher_self_model_hash") != base.finisher_self_model_hash
            or report.get("teacher_enabled") is not False
            or report.get("scripted_contact_torque_enabled") is not False
            or report.get("physics_authority") != "CPU_MUJOCO"
            or report.get("hardware_command_sent") is not False
            or (
                precision_repair
                and (
                    parent is None
                    or (
                        report.get("finish_plan_actor_hash") != parent.actor_hash
                        and report.get("report_hash") not in parent.source_evidence_hashes
                    )
                )
            )
            or not request_path.is_file()
            or hash_bytes(request_path.read_bytes()) != report.get("request_hash")
            or not isinstance(rows, list)
        ):
            raise ValueError("runtime finish plan source authority is invalid")
        sources.append((report_path, report))

    memories: list[RuntimeFinishPlanMemory] = []
    labels: list[bool] = []
    precise_labels: list[bool] = []
    source_counts: dict[str, int] = {}
    for report_path, report in sources:
        schema = str(report["schema_version"])
        for row in cast(list[dict[str, Any]], report["rows"]):
            action = _row_action(schema=schema, row=row, base_receive=base.required_receive_action)
            if action is None:
                continue
            artifact = cast(dict[str, Any], row["trajectory"])
            trajectory_path = report_path.parent / str(artifact["file"])
            if hash_bytes(trajectory_path.read_bytes()) != artifact["file_hash"]:
                raise ValueError("runtime finish plan trajectory binding changed")
            context = cast(dict[str, Any], row["context"])
            playmaker = cast(dict[str, Any], row["playmaker_action"])
            base_yaw = lead.passer_world_yaw(target_lateral_m=float(context["receiver_lane_m"]))
            yaw = math.atan2(
                math.sin(base_yaw + float(playmaker["body_yaw_correction_rad"])),
                math.cos(base_yaw + float(playmaker["body_yaw_correction_rad"])),
            )
            features = prepared_finish_plan_features(
                receiver_lane_m=float(context["receiver_lane_m"]),
                reception_target_x_m=float(context["reception_target_x_m"]),
                passer_ball_local_xy_m=context["passer_ball_local_xy_m"],
                ball_ground_friction=float(context["ball_ground_friction"]),
                passer_yaw_rad=yaw,
                passer_stance_offset_xy_m=(
                    float(context["passer_ball_local_xy_m"][0])
                    - 1.205
                    + float(playmaker["stance_correction_x_m"]),
                    float(context["passer_ball_local_xy_m"][1])
                    + 0.160
                    + float(playmaker["stance_correction_y_m"]),
                ),
                passer_swing_speed_scale=float(playmaker["swing_speed_scale"]),
            )
            quality = cast(dict[str, Any], row["quality"])
            result = cast(dict[str, Any], row["result"])
            strict = bool(quality["strict_chain_passed"])
            target_error = result.get("target_error_m")
            precise_goal = bool(
                strict
                and result["goal_crossed"]
                and isinstance(target_error, int | float)
                and math.isfinite(float(target_error))
                and float(target_error) <= 0.10
            )
            score = float(
                4.0 * strict
                + bool(quality["safe"])
                + bool(quality["clear_outcome"])
                + bool(quality["intended_foot_contact"])
                + 3.0 * min(float(result["shot_peak_ball_speed_mps"]) / 10.0, 1.0)
                + bool(result["goal_crossed"])
                + precise_goal
            )
            memories.append(
                RuntimeFinishPlanMemory(
                    context_hash=str(row["context_hash"]),
                    trajectory_hash=str(artifact["trajectory_digest"]),
                    features=features,
                    action=action,
                    quality_score=score,
                )
            )
            labels.append(strict)
            precise_labels.append(precise_goal)
            source_counts[schema] = source_counts.get(schema, 0) + 1
    if len({memory.trajectory_hash for memory in memories}) != len(memories):
        raise ValueError("runtime finish plan sources contain duplicate trajectories")
    parent_memory_count = 0
    if parent is not None:
        parent_by_trajectory = {
            memory.trajectory_hash: (memory, passed)
            for passed, collection in (
                (True, parent.successful_memories),
                (False, parent.failed_memories),
            )
            for memory in collection
        }
        if len(parent_by_trajectory) != len(parent.successful_memories) + len(
            parent.failed_memories
        ):
            raise ValueError("runtime finish plan parent trajectories are duplicated")
        for index, reconstructed in enumerate(memories):
            prior = parent_by_trajectory.get(reconstructed.trajectory_hash)
            if prior is None:
                continue
            prior_memory, prior_label = prior
            if (
                reconstructed.context_hash != prior_memory.context_hash
                or reconstructed.features != prior_memory.features
                or reconstructed.action != prior_memory.action
                or labels[index] != prior_label
            ):
                raise ValueError("runtime finish plan parent memory binding changed")
            # Stability: preserve the immutable prior memory, including its old
            # quality score. Revised scoring applies only to new trajectories.
            memories[index] = prior_memory
            parent_memory_count += 1
        if parent_memory_count != len(parent_by_trajectory):
            raise ValueError("runtime finish plan parent memory set is incomplete")
    successful = tuple(memory for memory, passed in zip(memories, labels, strict=True) if passed)
    failed = tuple(memory for memory, passed in zip(memories, labels, strict=True) if not passed)
    matrix = np.asarray([memory.features for memory in memories], dtype=np.float64)
    if parent is None:
        center = np.mean(matrix, axis=0)
        scale = np.maximum(
            np.std(matrix, axis=0),
            np.asarray(
                (0.02, 0.02, 0.001, 0.001, 0.0005, 0.01, 0.005, 0.005, 0.01),
                dtype=np.float64,
            ),
        )
    else:
        center = np.asarray(parent.feature_center, dtype=np.float64)
        scale = np.asarray(parent.feature_scale, dtype=np.float64)
    snapshot = {
        "base_target_actor_hash": base.actor_hash,
        "neural_actor_hash": neural.actor_hash,
        "contact_handoff_actor_hash": handoff.actor_hash,
        "contact_handoff_offset_frames": handoff.selected_offset_frames,
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "source_report_hashes": [report["report_hash"] for _, report in sources],
        "parent_finish_plan_actor_hash": None if parent is None else parent.actor_hash,
        "trajectory_hashes": [memory.trajectory_hash for memory in memories],
        "features": [list(memory.features) for memory in memories],
        "actions": [asdict(memory.action) for memory in memories],
        "strict_success_labels": labels,
        "joint_plan_uses_only_pre_rollout_shared_team_intent": True,
        "feature_normalization_frozen_from_parent": parent is not None,
    }
    actor = G1RuntimeFinishPlanActor(
        body_hash=base.body_hash,
        kick_prior_hash=base.kick_prior_hash,
        roster_hash=base.roster_hash,
        finisher_self_model_hash=base.finisher_self_model_hash,
        neural_contact_actor_hash=neural.actor_hash,
        contact_handoff_actor_hash=handoff.actor_hash,
        contact_handoff_offset_frames=handoff.selected_offset_frames,
        source_evidence_hashes=tuple(str(report["report_hash"]) for _, report in sources),
        training_snapshot_hash=str(hash_json(snapshot)),
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
        successful_memories=successful,
        failed_memories=failed,
    )
    output = _new_external_output(output_dir)
    actor_path = output / "runtime-finish-plan-actor.json"
    save_runtime_finish_plan_actor(actor, actor_path)
    training_report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_finish_plan_training.v1",
        "status": "PASS_RUNTIME_FINISH_PLAN_TRAINING",
        "promotion_eligible": False,
        "actor_hash": actor.actor_hash,
        "actor_file": actor_path.name,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "base_target_actor_hash": base.actor_hash,
        "neural_actor_hash": neural.actor_hash,
        "contact_handoff_actor_hash": handoff.actor_hash,
        "parent_finish_plan_actor_hash": None if parent is None else parent.actor_hash,
        "source_report_hashes": list(actor.source_evidence_hashes),
        "training_snapshot_hash": actor.training_snapshot_hash,
        "metrics": {
            "trajectory_count": len(memories),
            "strict_successful_memory_count": len(successful),
            "failed_memory_count": len(failed),
            "unique_context_count": len({memory.context_hash for memory in memories}),
            "unique_plan_count": len({memory.action.action_hash for memory in memories}),
            "precise_goal_memory_count": sum(precise_labels),
            "exact_parent_memory_count": parent_memory_count,
            "new_memory_count": len(memories) - parent_memory_count,
            "source_row_counts": source_counts,
        },
        "causal_contract": {
            "joint_plan_uses_only_pre_rollout_shared_team_intent": True,
            "action_authority": "RECEIVE_GEOMETRY_PHASE_AND_TASK_SPACE_TARGET_ONLY",
            "joint_torque_actor_hash": neural.actor_hash,
            "same_actor_selects_coupled_contact_variables": True,
            "parent_memory_retained_exactly": parent is not None,
            "feature_normalization_frozen_from_parent": parent is not None,
        },
        "activation_ceiling": "SIM_ONLY",
        "physics_authority": "CPU_MUJOCO",
        "pixels_used_for_training": False,
        "hardware_command_sent": False,
    }
    training_report["report_hash"] = hash_json(training_report)
    _write_json(output / "training-report.json", training_report)
    return actor, training_report


def _row_action(
    *, schema: str, row: dict[str, Any], base_receive: RuntimeReceiveAction
) -> RuntimeFinishPlanAction | None:
    if schema == "rosclaw_soccer.runtime_contact_target_repair.v1":
        target_payload = row["action"]
        receive = base_receive
    elif schema == "rosclaw_soccer.runtime_receive_strike_repair.v1":
        target_payload = row["target_action"]
        receive = RuntimeReceiveAction(**row["receive_action"])
    elif schema == "rosclaw_soccer.runtime_ready_stance_repair.v1":
        target_payload = {
            "target_foot_velocity_xyz_mps": [9.0, 5.0, -1.0],
            "activation_ceiling": "SIM_ONLY",
            "direct_joint_torque_output": False,
        }
        receive = replace(
            base_receive,
            contact_policy_frame=248,
            stance_offset_y_m=float(row["stance_offset_y_m"]),
        )
    elif schema == "rosclaw_soccer.team_compatible_pass_discovery.v1":
        if row.get("target_accepted") is not True:
            return None
        target_value = row.get("selected_target_velocity_xyz_mps")
        if not isinstance(target_value, list):
            return None
        target_payload = {
            "target_foot_velocity_xyz_mps": target_value,
            "activation_ceiling": "SIM_ONLY",
            "direct_joint_torque_output": False,
        }
        receive = base_receive
    elif schema in {
        "rosclaw_soccer.prepared_finish_plan_repair.v1",
        "rosclaw_soccer.prepared_finish_precision_repair.v1",
        "rosclaw_soccer.continuous_finish_plan_repair.v1",
    }:
        action = cast(dict[str, Any], row["action"])
        receive = RuntimeReceiveAction(**cast(dict[str, Any], action["receive"]))
        target_payload = cast(dict[str, Any], action["target"])
    else:
        raise ValueError("unsupported runtime finish plan source")
    target = RuntimeContactTargetAction(
        target_foot_velocity_xyz_mps=tuple(target_payload["target_foot_velocity_xyz_mps"]),
        activation_ceiling=target_payload.get("activation_ceiling", "SIM_ONLY"),
        direct_joint_torque_output=target_payload.get("direct_joint_torque_output", False),
    )
    return RuntimeFinishPlanAction(receive=receive, target=target)


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("runtime finish plan output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["train_runtime_finish_plan_actor"]
