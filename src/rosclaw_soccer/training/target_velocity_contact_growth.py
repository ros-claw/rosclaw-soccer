"""Distil target-conditioned contact muscle memory from bound teacher traces."""

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


def train_target_velocity_contact_actor(
    *,
    discovery_report_paths: tuple[Path, ...],
    output_dir: Path,
) -> dict[str, Any]:
    if not discovery_report_paths:
        raise ValueError("target-velocity growth needs discovery evidence")
    output = _new_external_output(output_dir)
    target_samples: list[np.ndarray] = []
    velocity_samples: list[np.ndarray] = []
    force_samples: list[np.ndarray] = []
    source_hashes: list[str] = []
    body_hashes: set[str] = set()
    kick_prior_hashes: set[str] = set()
    safe_trajectories = 0
    failed_trajectories = 0
    artifact_count = 0
    source_files: dict[str, str] = {}
    windows: set[tuple[int, int]] = set()
    proximities: set[float] = set()
    offsets: set[tuple[float, float, float]] = set()
    for raw_path in discovery_report_paths:
        path = raw_path.expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        claimed = payload.pop("report_hash", None)
        if (
            claimed != hash_json(payload)
            or payload.get("schema_version")
            != "rosclaw.growth.target_velocity_contact_discovery.v1"
            or payload.get("promotion_eligible") is not False
            or payload.get("teacher_role") != "SIM_ONLY_DATA_GENERATOR"
        ):
            raise ValueError("target-velocity discovery report is invalid")
        payload["report_hash"] = claimed
        source_hashes.append(str(claimed))
        body_hashes.add(str(payload["body_hash"]))
        kick_prior_hashes.add(str(payload["kick_prior_hash"]))
        source_files[str(path)] = hash_bytes(path.read_bytes())
        for row in payload["rows"]:
            artifact = row["trajectory"]
            trajectory_path = path.parent / str(artifact["file"])
            if hash_bytes(trajectory_path.read_bytes()) != artifact["file_hash"]:
                raise ValueError("target-velocity discovery trajectory changed")
            with np.load(trajectory_path, allow_pickle=False) as archive:
                trajectory = {name: np.asarray(archive[name]) for name in archive.files}
            if trajectory_digest(trajectory) != artifact["trajectory_digest"]:
                raise ValueError("target-velocity trajectory digest changed")
            artifact_count += 1
            source_files[str(trajectory_path)] = artifact["file_hash"]
            if not bool(row["quality"]["safe"]):
                continue
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
                raise ValueError("target-velocity teacher trace is incomplete")
            if not np.any(active):
                # A safe miss outside the teacher's foot-ball envelope remains
                # bound failure evidence, but contributes no supervised label.
                continue
            safe_trajectories += 1
            failed_trajectories += int(not bool(row["quality"]["chain_passed"]))
            probe = row["probe"]
            target = np.asarray(probe["target_foot_velocity_xyz_mps"], dtype=np.float64)
            target_samples.append(np.repeat(target[None, :], np.count_nonzero(active), axis=0))
            velocity_samples.append(velocity[active])
            force_samples.append(force[active])
            config = row.get("teacher_config_hash")
            if not isinstance(config, str) or not config.startswith("sha256:"):
                raise ValueError("target-velocity teacher config is unbound")
            # Discovery v1 has one explicit, immutable teacher envelope.
            windows.add((230, 335))
            proximities.add(0.50)
            offsets.add((0.13, 0.0, -0.025))
    if (
        len(body_hashes) != 1
        or len(kick_prior_hashes) != 1
        or len(windows) != 1
        or len(proximities) != 1
        or len(offsets) != 1
        or safe_trajectories < 6
        or not 1 <= failed_trajectories < safe_trajectories
        or artifact_count < 8
    ):
        raise ValueError("target-velocity training evidence lacks safe failure diversity")
    target = np.concatenate(target_samples, axis=0)
    velocity = np.concatenate(velocity_samples, axis=0)
    force = np.concatenate(force_samples, axis=0)
    implementation_hash = hash_bytes(
        (Path(__file__).parents[1] / "growth" / "target_velocity_contact_actor.py").read_bytes()
    )
    actor = fit_g1_target_velocity_contact_actor(
        target_velocity_xyz_mps=target,
        foot_velocity_xyz_mps=velocity,
        teacher_force_xyz_n=force,
        body_hash=next(iter(body_hashes)),
        implementation_hash=implementation_hash,
        source_evidence_hashes=tuple(source_hashes),
        training_trajectory_count=safe_trajectories,
        failed_trajectory_count=failed_trajectories,
        maximum_foot_ball_distance_m=next(iter(proximities)),
        start_policy_frame=next(iter(windows))[0],
        end_policy_frame=next(iter(windows))[1],
        foot_strike_point_offset_m=next(iter(offsets)),
    )
    actor_path = output / "target-velocity-contact-actor.json"
    save_g1_target_velocity_contact_actor(actor, actor_path)
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.target_velocity_contact_training.v1",
        "status": "PASS_TARGET_VELOCITY_CONTACT_DISTILLATION",
        "promotion_eligible": False,
        "actor_hash": actor.actor_hash,
        "actor_file": actor_path.name,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "source_evidence_hashes": source_hashes,
        "source_files": source_files,
        "body_hash": actor.body_hash,
        "kick_prior_hash": next(iter(kick_prior_hashes)),
        "metrics": {
            "training_sample_count": actor.training_sample_count,
            "safe_trajectory_count": safe_trajectories,
            "failed_trajectory_count": failed_trajectories,
            "distillation_rmse_n": actor.distillation_rmse_n,
            "target_velocity_min_xyz_mps": list(actor.minimum_target_velocity_xyz_mps),
            "target_velocity_max_xyz_mps": list(actor.maximum_target_velocity_xyz_mps),
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
        raise ValueError("target-velocity actor output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["train_target_velocity_contact_actor"]
