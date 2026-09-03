"""Train coherent target-contact plans from teacher-free whole-world replay."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.target_contact_plan_actor import (
    G1TargetContactPlanActor,
    TargetContactPlanAction,
    TargetContactPlanMemory,
    planned_contact_mode_features,
    save_target_contact_plan_actor,
)
from rosclaw_soccer.growth.target_velocity_contact_actor import (
    load_g1_target_velocity_contact_actor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def train_target_contact_plan_actor(
    *,
    replay_report_path: Path,
    contact_actor_path: Path,
    contact_training_report_path: Path,
    output_dir: Path,
) -> tuple[G1TargetContactPlanActor, dict[str, Any]]:
    replay_path = replay_report_path.expanduser().resolve()
    replay = _bound_json(replay_path, "report_hash")
    if (
        replay.get("schema_version") != "rosclaw.growth.target_velocity_contact_replay.v1"
        or replay.get("status") != "PASS_TARGET_VELOCITY_CONTACT_REPLAY"
        or replay.get("promotion_eligible") is not False
        or replay.get("teacher_enabled") is not False
    ):
        raise ValueError("target contact planning needs passing teacher-free replay")
    contact = load_g1_target_velocity_contact_actor(contact_actor_path)
    training = _bound_json(contact_training_report_path, "report_hash")
    if (
        replay.get("actor_hash") != contact.actor_hash
        or training.get("actor_hash") != contact.actor_hash
        or training.get("body_hash") != contact.body_hash
        or training.get("status") != "PASS_TARGET_VELOCITY_CONTACT_DISTILLATION"
    ):
        raise ValueError("target contact actor lineage changed")
    memories: list[tuple[TargetContactPlanMemory, bool]] = []
    source_files = {
        str(replay_path): hash_bytes(replay_path.read_bytes()),
        str(contact_actor_path.resolve()): hash_bytes(contact_actor_path.resolve().read_bytes()),
        str(contact_training_report_path.resolve()): hash_bytes(
            contact_training_report_path.resolve().read_bytes()
        ),
    }
    for row in replay["rows"]:
        artifact = row["trajectory"]
        trajectory_path = replay_path.parent / str(artifact["file"])
        if hash_bytes(trajectory_path.read_bytes()) != artifact["file_hash"]:
            raise ValueError("target contact replay trajectory changed")
        source_files[str(trajectory_path)] = artifact["file_hash"]
        probe = row["probe"]
        context = probe["context"]
        action = TargetContactPlanAction(
            maximum_arrival_advance_frames=int(probe["maximum_arrival_advance_frames"]),
            stance_offset_x_m=float(probe["stance_offset_x_m"]),
            stance_offset_y_m=float(probe["stance_offset_y_m"]),
            contact_policy_frame=int(probe["contact_policy_frame"]),
            foot_yaw_offset_rad=float(probe["foot_yaw_offset_rad"]),
            foot_pitch_offset_rad=float(probe["foot_pitch_offset_rad"]),
            target_foot_velocity_xyz_mps=cast(
                tuple[float, float, float],
                tuple(float(item) for item in probe["target_foot_velocity_xyz_mps"]),
            ),
        )
        features = planned_contact_mode_features(
            receiver_lane_m=float(context["receiver_lane_m"]),
            reception_target_x_m=float(context["reception_target_x_m"]),
            passer_ball_local_xy_m=context["passer_ball_local_xy_m"],
            ball_ground_friction=float(context["ball_ground_friction"]),
        )
        memories.append(
            (
                TargetContactPlanMemory(
                    context_hash=str(hash_json(context)),
                    trajectory_hash=str(artifact["file_hash"]),
                    features=features,
                    action=action,
                    quality_score=float(row["result"]["shot_peak_ball_speed_mps"]),
                ),
                bool(row["quality"]["chain_passed"]),
            )
        )
    successful = tuple(memory for memory, passed in memories if passed)
    failed = tuple(memory for memory, passed in memories if not passed)
    if len(successful) < 4 or len(failed) < 4:
        raise ValueError("target contact plan training needs four successes and failures")
    vectors = np.asarray([memory.features for memory, _ in memories], dtype=np.float64)
    center = np.mean(vectors, axis=0)
    scale = np.maximum(np.ptp(vectors, axis=0), 1.0e-4)
    snapshot = {
        "source_replay_hash": replay["report_hash"],
        "target_contact_actor_hash": contact.actor_hash,
        "memories": [
            {"memory": asdict(memory), "chain_passed": passed} for memory, passed in memories
        ],
    }
    actor = G1TargetContactPlanActor(
        body_hash=contact.body_hash,
        kick_prior_hash=str(training["kick_prior_hash"]),
        target_contact_actor_hash=contact.actor_hash,
        source_replay_hash=str(replay["report_hash"]),
        training_snapshot_hash=str(hash_json(snapshot)),
        feature_center=tuple(float(item) for item in center),
        feature_scale=tuple(float(item) for item in scale),
        successful_memories=successful,
        failed_memories=failed,
    )
    output = _new_external_output(output_dir)
    actor_path = output / "target-contact-plan-actor.json"
    save_target_contact_plan_actor(actor, actor_path)
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.target_contact_plan_training.v1",
        "status": "PASS_TARGET_CONTACT_PLAN_TRAINING",
        "promotion_eligible": False,
        "actor_hash": actor.actor_hash,
        "actor_file": actor_path.name,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "target_contact_actor_hash": contact.actor_hash,
        "source_replay_hash": replay["report_hash"],
        "training_snapshot_hash": actor.training_snapshot_hash,
        "successful_memory_count": len(successful),
        "failed_memory_count": len(failed),
        "source_files": source_files,
        "late_stance_rewrite_allowed": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "training-report.json", report)
    return actor, report


def _bound_json(path: Path, hash_key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    claimed = payload.pop(hash_key, None)
    if claimed != hash_json(payload):
        raise ValueError(f"{path.name} integrity changed")
    payload[hash_key] = claimed
    return payload


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("target contact plan output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["train_target_contact_plan_actor"]
