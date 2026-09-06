"""Train measured-arrival body/contact routing from teacher discovery."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.runtime_contact_mode_actor import (
    G1RuntimeContactModeActor,
    RuntimeContactModeAction,
    RuntimeContactModeMemory,
    save_runtime_contact_mode_actor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.runtime_causal_strike_growth import (
    extract_runtime_causal_strike_features,
)


def train_runtime_contact_mode_actor(
    *, discovery_report_paths: tuple[Path, ...], output_dir: Path
) -> tuple[G1RuntimeContactModeActor, dict[str, Any]]:
    """Retain successful body modes and same-action failure exclusions."""

    if not discovery_report_paths:
        raise ValueError("runtime contact training needs discovery evidence")
    sources = tuple(_validate_discovery(path) for path in discovery_report_paths)
    body_hashes = {source["report"]["body_hash"] for source in sources}
    kick_hashes = {source["report"]["kick_prior_hash"] for source in sources}
    if len(body_hashes) != 1 or len(kick_hashes) != 1:
        raise ValueError("runtime contact discovery asset identities disagree")
    records: list[dict[str, Any]] = []
    for source in sources:
        for row in source["report"]["rows"]:
            probe = row["probe"]
            context = probe["context"]
            trajectory_path = source["root"] / row["trajectory"]["file"]
            features, observation_frame = extract_runtime_causal_strike_features(
                trajectory_path=trajectory_path,
                receiver_lane_m=float(context["receiver_lane_m"]),
            )
            records.append(
                {
                    "context_hash": str(row["probe_hash"]),
                    "trajectory_hash": str(row["trajectory"]["file_hash"]),
                    "features": features,
                    "observation_frame": observation_frame,
                    "action": RuntimeContactModeAction(
                        maximum_arrival_advance_frames=int(probe["maximum_arrival_advance_frames"]),
                        stance_offset_x_m=float(probe["stance_offset_x_m"]),
                        stance_offset_y_m=float(probe["stance_offset_y_m"]),
                        contact_policy_frame=int(probe["contact_policy_frame"]),
                    ),
                    "chain_passed": bool(row["quality"]["chain_passed"]),
                    "safe": bool(row["quality"]["safe"]),
                }
            )
    vectors = np.asarray([row["features"] for row in records], dtype=np.float64)
    center = np.mean(vectors, axis=0)
    scale = np.maximum(np.ptp(vectors, axis=0), 1.0e-4)

    def memory(record: dict[str, Any]) -> RuntimeContactModeMemory:
        return RuntimeContactModeMemory(
            context_hash=record["context_hash"],
            trajectory_hash=record["trajectory_hash"],
            features=tuple(float(item) for item in record["features"]),
            action=record["action"],
        )

    successful = tuple(memory(row) for row in records if row["chain_passed"])
    failed = tuple(memory(row) for row in records if not row["chain_passed"])
    if len(successful) < 4 or len(failed) < 4:
        raise ValueError("runtime contact training needs four successes and failures")
    snapshot = {
        "source_discovery_hashes": [source["report"]["report_hash"] for source in sources],
        "records": [
            {
                **record,
                "features": list(record["features"]),
                "action": asdict(record["action"]),
            }
            for record in records
        ],
    }
    actor = G1RuntimeContactModeActor(
        body_hash=next(iter(body_hashes)),
        kick_prior_hash=next(iter(kick_hashes)),
        source_discovery_hashes=tuple(str(source["report"]["report_hash"]) for source in sources),
        training_snapshot_hash=str(hash_json(snapshot)),
        feature_center=tuple(float(item) for item in center),
        feature_scale=tuple(float(item) for item in scale),
        successful_memories=successful,
        failed_memories=failed,
    )
    output = _new_external_output(output_dir)
    actor_path = output / "runtime-contact-mode-actor.json"
    save_runtime_contact_mode_actor(actor, actor_path)
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.runtime_contact_mode_training.v1",
        "status": "PASS_RUNTIME_CONTACT_MODE_TRAINING",
        "source_discovery_hashes": list(actor.source_discovery_hashes),
        "training_snapshot_hash": actor.training_snapshot_hash,
        "trajectory_count": len(records),
        "successful_memory_count": len(successful),
        "failed_memory_count": len(failed),
        "unsafe_memory_count": sum(not row["safe"] for row in records),
        "actor_hash": actor.actor_hash,
        "actor_file": actor_path.name,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "decision_clock": "FIFTH_CONSECUTIVE_MEASURED_INCOMING_BALL_OBSERVATION",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "training-report.json", report)
    return actor, report


def _validate_discovery(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    report = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("runtime contact discovery report must be an object")
    claimed = report.pop("report_hash", None)
    try:
        if (
            claimed != hash_json(report)
            or report.get("schema_version") != "rosclaw.growth.three_axis_contact_discovery.v1"
            or report.get("activation_ceiling") != "SIM_ONLY"
            or not isinstance(report.get("rows"), list)
        ):
            raise ValueError("runtime contact discovery authority is invalid")
        for row in report["rows"]:
            trajectory = source.parent / row["trajectory"]["file"]
            if (
                not trajectory.is_file()
                or hash_bytes(trajectory.read_bytes()) != row["trajectory"]["file_hash"]
            ):
                raise ValueError("runtime contact discovery trajectory binding changed")
    finally:
        if claimed is not None:
            report["report_hash"] = claimed
    return {"report": report, "root": source.parent}


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("runtime contact output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["train_runtime_contact_mode_actor"]
