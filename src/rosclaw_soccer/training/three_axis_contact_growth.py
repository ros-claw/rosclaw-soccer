"""Distil the SIM-only contact teacher into a frozen proprioceptive actor."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.three_axis_contact_actor import (
    G1ThreeAxisContactActor,
    fit_g1_three_axis_contact_actor,
    save_g1_three_axis_contact_actor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_DISCOVERY_SCHEMA = "rosclaw.growth.three_axis_contact_discovery.v1"


def train_three_axis_contact_actor(
    *,
    discovery_report_path: Path,
    output_dir: Path,
    ridge_regularization: float = 1.0e-4,
) -> tuple[G1ThreeAxisContactActor, dict[str, Any]]:
    """Fit only from safe trajectories while binding every observed failure."""

    source, report = _validate_discovery(discovery_report_path)
    output = _new_external_output(output_dir)
    safe_samples: list[tuple[np.ndarray, np.ndarray]] = []
    evidence_hashes: list[str] = []
    failed_count = 0
    active_trajectory_count = 0
    for row in report["rows"]:
        artifact = row["trajectory"]
        trajectory_path = source.parent / str(artifact["file"])
        evidence_hashes.append(str(artifact["file_hash"]))
        chain_passed = bool(row["quality"]["chain_passed"])
        if not chain_passed:
            failed_count += 1
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            required = {
                "shooter_loft_teacher_active",
                "shooter_loft_teacher_force_xyz_n",
                "shooter_loft_teacher_foot_velocity_xyz_mps",
            }
            if not required.issubset(trajectory.files):
                raise ValueError("three-axis discovery trajectory lacks exact teacher channels")
            active = np.asarray(trajectory["shooter_loft_teacher_active"], dtype=np.bool_)
            force = np.asarray(trajectory["shooter_loft_teacher_force_xyz_n"], dtype=np.float64)
            velocity = np.asarray(
                trajectory["shooter_loft_teacher_foot_velocity_xyz_mps"], dtype=np.float64
            )
        if (
            active.ndim != 1
            or force.shape != (len(active), 3)
            or velocity.shape != force.shape
            or not np.all(np.isfinite(force))
            or not np.all(np.isfinite(velocity))
        ):
            raise ValueError("three-axis discovery teacher channels are invalid")
        if np.any(active):
            active_trajectory_count += 1
        if bool(row["quality"]["safe"]) and np.any(active):
            safe_samples.append((velocity[active], force[active]))
    if (
        len(report["rows"]) < 4
        or active_trajectory_count < 4
        or len(safe_samples) < 3
        or failed_count < 1
        or failed_count >= len(report["rows"])
    ):
        raise ValueError("three-axis discovery lacks safe activity and bound failures")
    velocity_samples = np.concatenate([sample[0] for sample in safe_samples], axis=0)
    force_samples = np.concatenate([sample[1] for sample in safe_samples], axis=0)
    teacher = report["teacher_config"]
    actor = fit_g1_three_axis_contact_actor(
        foot_velocity_xyz_mps=velocity_samples,
        teacher_force_xyz_n=force_samples,
        body_hash=str(report["body_hash"]),
        implementation_hash=hash_bytes(
            (Path(__file__).parents[1] / "growth" / "three_axis_contact_actor.py").read_bytes()
        ),
        source_evidence_hashes=tuple(evidence_hashes),
        training_trajectory_count=len(report["rows"]),
        failed_trajectory_count=failed_count,
        maximum_foot_ball_distance_m=float(teacher["maximum_foot_ball_distance_m"]),
        start_policy_frame=int(teacher["start_policy_frame"]),
        end_policy_frame=int(teacher["end_policy_frame"]),
        foot_strike_point_offset_m=cast(
            tuple[float, float, float],
            tuple(float(item) for item in teacher["foot_strike_point_offset_m"]),
        ),
        ridge_regularization=ridge_regularization,
    )
    actor_path = output / "three-axis-contact-actor.json"
    save_g1_three_axis_contact_actor(actor, actor_path)
    training_report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.three_axis_contact_training.v1",
        "status": "PASS_THREE_AXIS_CONTACT_DISTILLATION",
        "source_discovery_hash": report["report_hash"],
        "source_evidence_hashes": evidence_hashes,
        "safe_training_trajectory_count": len(safe_samples),
        "failed_trajectory_count": failed_count,
        "training_sample_count": actor.training_sample_count,
        "distillation_rmse_n": actor.distillation_rmse_n,
        "actor_hash": actor.actor_hash,
        "actor_file": actor_path.name,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "teacher_enabled_at_runtime": False,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "hardware_command_sent": False,
    }
    training_report["report_hash"] = hash_json(training_report)
    _write_json(output / "training-report.json", training_report)
    return actor, training_report


def _validate_discovery(path: Path) -> tuple[Path, dict[str, Any]]:
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("three-axis discovery report must be an object")
    claimed = payload.pop("report_hash", None)
    try:
        if (
            claimed != hash_json(payload)
            or payload.get("schema_version") != _DISCOVERY_SCHEMA
            or payload.get("status")
            not in {"PASS_THREE_AXIS_CONTACT_DISCOVERY", "REJECTED_THREE_AXIS_CONTACT_DISCOVERY"}
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or not isinstance(payload.get("rows"), list)
        ):
            raise ValueError("three-axis discovery report authority is invalid")
        for row in payload["rows"]:
            artifact = row.get("trajectory", {})
            trajectory_path = source.parent / str(artifact.get("file", ""))
            if not trajectory_path.is_file() or hash_bytes(
                trajectory_path.read_bytes()
            ) != artifact.get("file_hash"):
                raise ValueError("three-axis discovery trajectory binding changed")
    finally:
        if claimed is not None:
            payload["report_hash"] = claimed
    return source, payload


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("three-axis training output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["train_three_axis_contact_actor"]
