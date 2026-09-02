"""Train runtime strike routing from immutable measured-arrival trajectories."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.causal_strike_router import CausalStrikeRouteAction
from rosclaw_soccer.growth.runtime_causal_strike_router import (
    G1RuntimeCausalStrikeRouter,
    RuntimeCausalStrikeMemory,
    runtime_causal_strike_features,
    save_runtime_causal_strike_router,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def extract_runtime_causal_strike_features(
    *, trajectory_path: Path, receiver_lane_m: float
) -> tuple[tuple[float, ...], int]:
    """Read the first stable incoming observation from a complete trajectory."""

    with np.load(trajectory_path.expanduser().resolve(), allow_pickle=False) as trajectory:
        counts = np.asarray(
            trajectory["shooter_causal_strike_option_incoming_observation_count"],
            dtype=np.int64,
        )
        indices = np.flatnonzero(counts >= 5)
        if indices.size == 0:
            raise ValueError("trajectory never formed a stable incoming-ball observation")
        frame = int(indices[0])
        eta = float(trajectory["shooter_causal_strike_option_ball_arrival_eta_sec"][frame])
        if "shooter_ball_local_position" in trajectory:
            ball_position = np.asarray(
                trajectory["shooter_ball_local_position"][frame], dtype=np.float64
            )
            ball_velocity = np.asarray(
                trajectory["shooter_ball_local_velocity"][frame], dtype=np.float64
            )
            pelvis_position = np.asarray(
                trajectory["shooter_pelvis_local_position"][frame], dtype=np.float64
            )
        else:
            # S126 predates the explicit local-state trace. Its curriculum has
            # shooter yaw=0 and origin=(0, receiver_lane, 0), so this is the
            # exact corresponding frame transform rather than an estimate.
            ball_position = np.asarray(trajectory["ball_pose"][frame, :3], dtype=np.float64)
            pelvis_position = np.asarray(
                trajectory["shooter_pelvis_pose"][frame, :3], dtype=np.float64
            )
            ball_position[1] -= receiver_lane_m
            pelvis_position[1] -= receiver_lane_m
            ball_velocity = np.asarray(trajectory["ball_velocity"][frame, :3], dtype=np.float64)
        joint_velocity = np.asarray(trajectory["shooter_joint_velocity"][frame], dtype=np.float64)
    features = runtime_causal_strike_features(
        ball_local_position_m=ball_position.tolist(),
        ball_local_velocity_mps=ball_velocity.tolist(),
        ball_arrival_eta_sec=eta,
        pelvis_local_position_m=pelvis_position.tolist(),
        joint_velocity_rad_s=joint_velocity.tolist(),
    )
    return features, frame


def train_runtime_causal_strike_router(
    *, discovery_report_paths: tuple[Path, ...], output_dir: Path
) -> tuple[G1RuntimeCausalStrikeRouter, dict[str, Any]]:
    """Retain all success/failure modes from one or more bound discoveries."""

    if not discovery_report_paths:
        raise ValueError("runtime strike training needs at least one discovery report")
    output = _new_external_output(output_dir)
    sources = tuple(_validate_discovery(path) for path in discovery_report_paths)
    body_hashes = {source["report"]["body_hash"] for source in sources}
    kick_hashes = {source["report"]["kick_prior_hash"] for source in sources}
    if len(body_hashes) != 1 or len(kick_hashes) != 1:
        raise ValueError("runtime strike discovery asset identities disagree")

    records: list[dict[str, Any]] = []
    for source in sources:
        source_report = source["report"]
        request = source["request"]
        contexts = {
            str(context_hash): context
            for context_hash, context in zip(
                request["context_hashes"], request["contexts"], strict=True
            )
        }
        for row in source_report["rows"]:
            context = contexts[row["context_hash"]]
            trajectory_path = (
                source["root"]
                / (f"case-{int(row['case_index']):03d}-action-{int(row['action_index']):02d}")
                / row["trajectory"]["file"]
            )
            features, observation_frame = extract_runtime_causal_strike_features(
                trajectory_path=trajectory_path,
                receiver_lane_m=float(context["receiver_lane_m"]),
            )
            records.append(
                {
                    "source_report_hash": source_report["report_hash"],
                    "context_hash": row["context_hash"],
                    "trajectory_hash": row["trajectory"]["file_hash"],
                    "features": features,
                    "observation_frame": observation_frame,
                    "action": CausalStrikeRouteAction(**row["action"]),
                    "chain_passed": bool(row["quality"]["chain_passed"]),
                    "safe": bool(row["quality"]["safe"]),
                }
            )
    vectors = np.asarray([record["features"] for record in records], dtype=np.float64)
    center = np.mean(vectors, axis=0)
    scale = np.maximum(np.ptp(vectors, axis=0), 1.0e-4)

    def memory(record: dict[str, Any]) -> RuntimeCausalStrikeMemory:
        return RuntimeCausalStrikeMemory(
            context_hash=record["context_hash"],
            trajectory_hash=record["trajectory_hash"],
            features=tuple(float(value) for value in record["features"]),
            action=record["action"],
        )

    successful = tuple(memory(record) for record in records if record["chain_passed"])
    failed = tuple(memory(record) for record in records if not record["chain_passed"])
    if len(successful) < 4 or len(failed) < 4:
        raise ValueError("runtime strike training needs at least four successes and failures")
    snapshot = {
        "source_discovery_hashes": [source["report"]["report_hash"] for source in sources],
        "records": [
            {
                "context_hash": record["context_hash"],
                "trajectory_hash": record["trajectory_hash"],
                "features": list(record["features"]),
                "observation_frame": record["observation_frame"],
                "action": asdict(record["action"]),
                "chain_passed": record["chain_passed"],
                "safe": record["safe"],
            }
            for record in records
        ],
    }
    actor = G1RuntimeCausalStrikeRouter(
        body_hash=next(iter(body_hashes)),
        kick_prior_hash=next(iter(kick_hashes)),
        source_discovery_hashes=tuple(source["report"]["report_hash"] for source in sources),
        training_snapshot_hash=hash_json(snapshot),
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
        successful_memories=successful,
        failed_memories=failed,
    )
    actor_path = output / "runtime-causal-strike-router.json"
    save_runtime_causal_strike_router(actor, actor_path)
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.runtime_causal_strike_training.v1",
        "status": "PASS_RUNTIME_CAUSAL_STRIKE_TRAINING",
        "source_discovery_hashes": list(actor.source_discovery_hashes),
        "training_snapshot_hash": actor.training_snapshot_hash,
        "trajectory_count": len(records),
        "successful_memory_count": len(successful),
        "failed_memory_count": len(failed),
        "unsafe_memory_count": sum(not record["safe"] for record in records),
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
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime strike discovery source must be an object")
    claimed = payload.pop("report_hash", None)
    try:
        if (
            claimed != hash_json(payload)
            or payload.get("schema_version") != "rosclaw.growth.causal_strike_route_discovery.v1"
            or payload.get("status") not in {"PASS_ROUTE_DISCOVERY", "REJECTED_ROUTE_DISCOVERY"}
            or not isinstance(payload.get("rows"), list)
            or not payload["rows"]
        ):
            raise ValueError("runtime strike discovery source authority is invalid")
        request_path = source.parent / "request.json"
        if (
            not request_path.is_file()
            or hash_bytes(request_path.read_bytes()) != payload["request_hash"]
        ):
            raise ValueError("runtime strike discovery request binding changed")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            trajectory_path = (
                source.parent
                / (f"case-{int(row['case_index']):03d}-action-{int(row['action_index']):02d}")
                / row["trajectory"]["file"]
            )
            if (
                not trajectory_path.is_file()
                or hash_bytes(trajectory_path.read_bytes()) != row["trajectory"]["file_hash"]
            ):
                raise ValueError("runtime strike source trajectory binding changed")
    finally:
        if claimed is not None:
            payload["report_hash"] = claimed
    return {"report": payload, "request": request, "root": source.parent}


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("runtime strike training output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "extract_runtime_causal_strike_features",
    "train_runtime_causal_strike_router",
]
