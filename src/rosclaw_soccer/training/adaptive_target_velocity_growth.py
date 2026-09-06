"""Distil a wider target-velocity actor from safe failure-driven teacher data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.target_velocity_contact_actor import (
    fit_g1_target_velocity_contact_actor,
    save_g1_target_velocity_contact_actor,
)
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def train_adaptive_target_velocity_actor(
    *, discovery_report_path: Path, output_dir: Path
) -> dict[str, Any]:
    source_path = discovery_report_path.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    claimed = source.pop("report_hash", None)
    if (
        claimed != hash_json(source)
        or source.get("schema_version") != "rosclaw.growth.adaptive_target_teacher_discovery.v1"
        or source.get("status") != "REJECTED_ADAPTIVE_TARGET_TEACHER_DISCOVERY"
        or source.get("promotion_eligible") is not False
        or source.get("teacher_role") != "SIM_ONLY_DATA_GENERATOR"
    ):
        raise ValueError("adaptive target growth requires intact failure-driven teacher data")
    source["report_hash"] = claimed
    targets: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    forces: list[np.ndarray] = []
    source_files = {str(source_path): hash_bytes(source_path.read_bytes())}
    safe_count = 0
    failed_count = 0
    successful_count = 0
    bound_artifact_count = 0
    for row in source["rows"]:
        artifact = row["trajectory"]
        trajectory_path = source_path.parent / artifact["file"]
        if hash_bytes(trajectory_path.read_bytes()) != artifact["file_hash"]:
            raise ValueError("adaptive teacher trajectory changed")
        with np.load(trajectory_path, allow_pickle=False) as archive:
            trajectory = {name: np.asarray(archive[name]) for name in archive.files}
        if trajectory_digest(trajectory) != artifact["trajectory_digest"]:
            raise ValueError("adaptive teacher trajectory digest changed")
        source_files[str(trajectory_path)] = artifact["file_hash"]
        bound_artifact_count += 1
        if not bool(row["quality"]["safe"]):
            continue
        active = np.asarray(trajectory["shooter_loft_teacher_active"], dtype=np.bool_)
        force = np.asarray(trajectory["shooter_loft_teacher_force_xyz_n"], dtype=np.float64)
        velocity = np.asarray(
            trajectory["shooter_loft_teacher_foot_velocity_xyz_mps"], dtype=np.float64
        )
        if (
            not np.any(active)
            or force.shape != (len(active), 3)
            or velocity.shape != force.shape
            or not np.all(np.isfinite(force))
            or not np.all(np.isfinite(velocity))
        ):
            raise ValueError("adaptive teacher safe trace is incomplete")
        target = np.asarray(row["action"]["target_foot_velocity_xyz_mps"], dtype=np.float64)
        targets.append(np.repeat(target[None, :], np.count_nonzero(active), axis=0))
        velocities.append(velocity[active])
        forces.append(force[active])
        safe_count += 1
        passed = bool(row["quality"]["chain_passed"])
        successful_count += int(passed)
        failed_count += int(not passed)
    if bound_artifact_count < 24 or safe_count < 12 or successful_count < 1 or failed_count < 8:
        raise ValueError("adaptive teacher data lacks safe success/failure support")
    target_array = np.concatenate(targets, axis=0)
    velocity_array = np.concatenate(velocities, axis=0)
    force_array = np.concatenate(forces, axis=0)
    actor = fit_g1_target_velocity_contact_actor(
        target_velocity_xyz_mps=target_array,
        foot_velocity_xyz_mps=velocity_array,
        teacher_force_xyz_n=force_array,
        body_hash=str(source["body_hash"]),
        implementation_hash=hash_bytes(
            (Path(__file__).parents[1] / "growth" / "target_velocity_contact_actor.py").read_bytes()
        ),
        source_evidence_hashes=(str(claimed),),
        training_trajectory_count=safe_count,
        failed_trajectory_count=failed_count,
        maximum_foot_ball_distance_m=0.50,
        start_policy_frame=230,
        end_policy_frame=335,
    )
    output = _new_external_output(output_dir)
    actor_path = output / "adaptive-target-velocity-contact-actor.json"
    save_g1_target_velocity_contact_actor(actor, actor_path)
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.adaptive_target_velocity_training.v1",
        "status": "PASS_ADAPTIVE_TARGET_VELOCITY_DISTILLATION",
        "promotion_eligible": False,
        "source_discovery_status": source["status"],
        "source_discovery_report_hash": claimed,
        "source_files": source_files,
        "actor_hash": actor.actor_hash,
        "actor_file": actor_path.name,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "body_hash": actor.body_hash,
        "kick_prior_hash": source["kick_prior_hash"],
        "metrics": {
            "bound_artifact_count": bound_artifact_count,
            "safe_training_trajectory_count": safe_count,
            "successful_training_trajectory_count": successful_count,
            "failed_training_trajectory_count": failed_count,
            "training_sample_count": actor.training_sample_count,
            "distillation_rmse_n": actor.distillation_rmse_n,
            "target_velocity_min_xyz_mps": list(actor.minimum_target_velocity_xyz_mps),
            "target_velocity_max_xyz_mps": list(actor.maximum_target_velocity_xyz_mps),
            "force_min_xyz_n": list(actor.minimum_force_xyz_n),
            "force_max_xyz_n": list(actor.maximum_force_xyz_n),
        },
        "teacher_required_at_runtime": False,
        "runtime_replay_passed": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "training-report.json", report)
    return report


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("adaptive target actor output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["train_adaptive_target_velocity_actor"]
