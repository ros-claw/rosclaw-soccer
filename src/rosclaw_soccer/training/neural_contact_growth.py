"""Distill a complete neural contact muscle memory from bound SIM traces."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.neural_contact_actor import (
    fit_g1_neural_contact_actor,
    neural_contact_features,
    save_g1_neural_contact_actor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def train_neural_contact_actor(
    *,
    source_teacher_report_path: Path,
    output_dir: Path,
    specialist_target_velocity_xyz_mps: tuple[float, float, float] | None = (
        9.0,
        -3.0,
        -1.0,
    ),
) -> dict[str, Any]:
    source_path = source_teacher_report_path.expanduser().resolve()
    source = _bound_teacher_report(source_path)
    feature_rows: list[np.ndarray] = []
    torque_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    source_files: list[dict[str, str]] = []
    safe_count = 0
    success_count = 0
    failure_count = 0
    specialist_target = (
        None
        if specialist_target_velocity_xyz_mps is None
        else np.asarray(specialist_target_velocity_xyz_mps, dtype=np.float64)
    )
    if specialist_target is not None and (
        specialist_target.shape != (3,) or not np.all(np.isfinite(specialist_target))
    ):
        raise ValueError("neural contact specialist target must be three finite values")
    for row in source["rows"]:
        artifact = row["trajectory"]
        trajectory_path = source_path.parent / artifact["file"]
        if hash_bytes(trajectory_path.read_bytes()) != artifact["file_hash"]:
            raise ValueError("neural contact source trajectory changed")
        source_files.append({"file": artifact["file"], "file_hash": artifact["file_hash"]})
        if not row["quality"]["safe"]:
            continue
        action = row["action"]
        target = np.asarray(action["target_foot_velocity_xyz_mps"], dtype=np.float64)
        if specialist_target is not None and not np.array_equal(target, specialist_target):
            continue
        safe_count += 1
        chain_passed = bool(row["quality"]["chain_passed"])
        success_count += int(chain_passed)
        failure_count += int(not chain_passed)
        contact_frame = int(action["contact_policy_frame"])
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            policy_frame = np.asarray(trajectory["shooter_policy_frame"], dtype=np.int64)
            selected = np.flatnonzero(
                (policy_frame - contact_frame >= -5) & (policy_frame - contact_frame <= 8)
            )
            total_torque = np.asarray(
                trajectory["shooter_ballistic_contact_torque"]
                + trajectory["shooter_loft_teacher_torque"],
                dtype=np.float64,
            )
            for index in selected:
                feature_rows.append(
                    neural_contact_features(
                        phase_offset_frames=float(policy_frame[index] - contact_frame),
                        target_velocity_xyz_mps=target,
                        ball_local_position_m=trajectory["shooter_ball_local_position"][index],
                        ball_local_velocity_mps=trajectory["shooter_ball_local_velocity"][index],
                        joint_position_rad=trajectory["shooter_joint_position"][index],
                        joint_velocity_rad_s=trajectory["shooter_joint_velocity"][index],
                    )
                )
                torque_rows.append(total_torque[index].copy())
                target_rows.append(target.copy())
    if safe_count < 2 or success_count < 1 or failure_count < 1:
        raise ValueError("neural contact specialist lacks safe success/failure support")
    actor = fit_g1_neural_contact_actor(
        features=np.asarray(feature_rows, dtype=np.float64),
        target_torque_nm=np.asarray(torque_rows, dtype=np.float64),
        target_velocity_xyz_mps=np.asarray(target_rows, dtype=np.float64),
        body_hash=str(source["body_hash"]),
        source_evidence_hashes=(str(source["report_hash"]),),
        training_trajectory_count=safe_count,
        failed_trajectory_count=failure_count,
    )
    output = _new_external_output(output_dir)
    actor_path = output / "neural-contact-actor.json"
    save_g1_neural_contact_actor(actor, actor_path)
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.neural_contact_training.v1",
        "status": "PASS_NEURAL_CONTACT_DISTILLATION",
        "promotion_eligible": False,
        "source_teacher_status": source["status"],
        "source_teacher_report_hash": source["report_hash"],
        "source_files": source_files,
        "actor_hash": actor.actor_hash,
        "actor_file": actor_path.name,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "body_hash": actor.body_hash,
        "metrics": {
            "safe_training_trajectory_count": safe_count,
            "successful_training_trajectory_count": success_count,
            "failed_training_trajectory_count": failure_count,
            "training_sample_count": actor.training_sample_count,
            "training_rmse_nm": actor.training_rmse_nm,
            "hidden_width": len(actor.hidden_one_bias),
            "specialist_target_velocity_xyz_mps": (
                None if specialist_target is None else specialist_target.tolist()
            ),
        },
        "output_authority": "BOUNDED_29_DOF_CONTACT_RESIDUAL_TORQUE",
        "replaces_scripted_contact_torque": True,
        "teacher_required_at_runtime": False,
        "runtime_canary_passed": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "training-report.json", report)
    return report


def _bound_teacher_report(path: Path) -> dict[str, Any]:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    claimed = payload.pop("report_hash", None)
    adaptive = bool(
        payload.get("schema_version") == "rosclaw.growth.adaptive_target_teacher_discovery.v1"
        and payload.get("status") == "REJECTED_ADAPTIVE_TARGET_TEACHER_DISCOVERY"
        and len(payload.get("rows", ())) >= 24
    )
    runtime_receive = bool(
        payload.get("schema_version") == "rosclaw_soccer.runtime_receive_contact_teacher.v1"
        and payload.get("status") == "PASS_RUNTIME_RECEIVE_CONTACT_TEACHER"
        and payload.get("teacher_role") == "SIM_ONLY_LOW_LEVEL_CONTACT_DATA_GENERATOR"
        and len(payload.get("rows", ())) >= 16
    )
    if (
        claimed != hash_json(payload)
        or not (adaptive or runtime_receive)
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("neural contact training requires intact teacher responses")
    payload["report_hash"] = claimed
    return payload


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("neural contact output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["train_neural_contact_actor"]
