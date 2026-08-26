"""Content-bound CPU recovery routes for route-specialized MJX learning.

The CPU bridge exam searches reference motion, entry phase and time dilation.
Those choices are privileged development data, but they are still the only
physically demonstrated starting point for a recovery teacher.  This module
turns the selected CPU trials into an immutable route manifest before MJX is
allowed to train.  It deliberately grants no deployment or promotion
authority.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.recovery_snapshot import load_recovery_snapshot_corpus
from rosclaw_soccer.training.recovery_teacher_bridge import (
    RecoveryBridgeTrial,
    RecoveryEntryMatch,
)


@dataclass(frozen=True)
class RecoveryMJXRouteManifestConfig:
    """Fail-closed acceptance rules for CPU-demonstrated training routes."""

    minimum_final_stable_sec: float = 2.0
    control_dt_sec: float = 0.02
    episode_margin_sec: float = 4.0
    require_failed_snapshots: bool = True
    required_trial_backend: str = "mujoco_cpu"
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_mjx_route_manifest_config.v2"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.minimum_final_stable_sec)
            or not math.isfinite(self.control_dt_sec)
            or not math.isfinite(self.episode_margin_sec)
            or not 0.5 <= self.minimum_final_stable_sec <= 10.0
            or not math.isclose(self.control_dt_sec, 0.02, abs_tol=1e-9)
            or not 1.0 <= self.episode_margin_sec <= 10.0
            or not self.require_failed_snapshots
            or self.required_trial_backend != "mujoco_cpu"
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery MJX route manifest config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _verified_development_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery bridge development report must be an object")
    declared_hash = payload.pop("report_hash", None)
    if (
        payload.get("schema_version") != "rosclaw_soccer.opentrack_recovery_bridge_exam.v1"
        or declared_hash != hash_json(payload)
        or payload.get("physical_truth") is not True
        or payload.get("physics_backend") != "opentrack_mujoco_cpu"
        or payload.get("promotion_eligible") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("recovery bridge development report integrity failed")
    transfer = payload.get("post_skill_transfer")
    if not isinstance(transfer, dict):
        raise ValueError("recovery bridge post-skill transfer is absent")
    schedule = transfer.get("development_schedule")
    if not isinstance(schedule, dict):
        raise ValueError("recovery bridge development schedule is absent")
    schedule = dict(schedule)
    schedule_hash = schedule.pop("schedule_hash", None)
    if (
        schedule_hash != hash_json(schedule)
        or schedule.get("activation_ceiling") != "SIM_ONLY"
        or schedule.get("hardware_command_sent") is not False
    ):
        raise ValueError("recovery bridge development schedule integrity failed")
    schedule["schedule_hash"] = schedule_hash
    transfer = dict(transfer)
    transfer["development_schedule"] = schedule
    payload["post_skill_transfer"] = transfer
    payload["report_hash"] = declared_hash
    return payload


def _trial_from_dict(payload: dict[str, Any]) -> RecoveryBridgeTrial:
    raw = dict(payload)
    recorded_hash = raw.pop("trial_hash", None)
    raw_match = raw.pop("match", None)
    if not isinstance(raw_match, dict):
        raise ValueError("recovery route trial match is invalid")
    trial = RecoveryBridgeTrial(match=RecoveryEntryMatch(**raw_match), **raw)
    if recorded_hash != trial.trial_hash:
        raise ValueError("recovery route trial hash mismatch")
    return trial


def validate_recovery_mjx_route_manifest(path: Path) -> dict[str, Any]:
    """Load a route manifest and reject integrity or authority drift."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery MJX route manifest must be an object")
    declared_hash = payload.pop("report_hash", None)
    routes = payload.get("routes")
    groups = payload.get("route_groups")
    if (
        payload.get("schema_version") != "rosclaw_soccer.recovery_mjx_route_manifest.v2"
        or declared_hash != hash_json(payload)
        or not isinstance(routes, list)
        or not routes
        or not isinstance(groups, list)
        or not groups
        or payload.get("route_count") != len(routes)
        or payload.get("route_group_count") != len(groups)
        or payload.get("cpu_development_pass_rate") != 1.0
        or payload.get("training_backend") != "MUJOCO_MJX"
        or payload.get("physics_truth_backend") != "CPU_MUJOCO"
        or payload.get("promotion_eligible") is not False
        or payload.get("promotion_authority") != "NONE"
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("recovery MJX route manifest is invalid")
    route_hashes: set[str] = set()
    snapshot_indices: set[int] = set()
    snapshot_hashes: set[str] = set()
    for raw in routes:
        if not isinstance(raw, dict):
            raise ValueError("recovery MJX route row is invalid")
        route = dict(raw)
        route_hash = route.pop("route_hash", None)
        snapshot_index = route.get("snapshot_index")
        snapshot_hash = route.get("snapshot_hash")
        if (
            route_hash != hash_json(route)
            or route_hash in route_hashes
            or not isinstance(snapshot_index, int)
            or snapshot_index in snapshot_indices
            or not isinstance(snapshot_hash, str)
            or snapshot_hash in snapshot_hashes
            or route.get("cpu_trial_succeeded") is not True
            or route.get("time_dilation") not in (1, 2, 3, 4)
        ):
            raise ValueError("recovery MJX route row integrity failed")
        route_hashes.add(str(route_hash))
        snapshot_indices.add(snapshot_index)
        snapshot_hashes.add(snapshot_hash)
    if snapshot_indices != set(range(len(routes))):
        raise ValueError("recovery MJX route coverage is not ordered and exact")
    covered_group_routes: set[str] = set()
    for raw in groups:
        if not isinstance(raw, dict):
            raise ValueError("recovery MJX route group is invalid")
        group = dict(raw)
        group_hash = group.pop("route_group_hash", None)
        members = group.get("route_hashes")
        minimum_episode_length = group.get("minimum_episode_length")
        if (
            group_hash != hash_json(group)
            or not isinstance(members, list)
            or not members
            or any(not isinstance(value, str) for value in members)
            or covered_group_routes.intersection(members)
            or not isinstance(minimum_episode_length, int)
            or not 600 <= minimum_episode_length <= 3_000
        ):
            raise ValueError("recovery MJX route group integrity failed")
        covered_group_routes.update(members)
    if covered_group_routes != route_hashes:
        raise ValueError("recovery MJX route groups do not cover routes exactly")
    payload["report_hash"] = declared_hash
    return payload


def resolve_recovery_mjx_route_group(
    *,
    route_manifest_path: Path,
    route_group_index: int,
    snapshot_manifest_path: Path,
    teacher_policy_path: Path,
    teacher_config_path: Path,
    motion_archive_path: Path,
) -> dict[str, Any]:
    """Resolve one training job only when every local asset matches its route."""

    manifest = validate_recovery_mjx_route_manifest(route_manifest_path)
    if not isinstance(route_group_index, int) or not 0 <= route_group_index < len(
        manifest["route_groups"]
    ):
        raise ValueError("recovery MJX route group index is invalid")
    group = manifest["route_groups"][route_group_index]
    paths = tuple(
        path.expanduser().resolve()
        for path in (
            snapshot_manifest_path,
            teacher_policy_path,
            teacher_config_path,
            motion_archive_path,
        )
    )
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("recovery MJX route group assets are incomplete")
    snapshot_path, policy_path, config_path, motion_path = paths
    expected = (
        (hash_bytes(snapshot_path.read_bytes()), manifest["snapshot_manifest_hash"]),
        (hash_bytes(policy_path.read_bytes()), manifest["teacher_policy_hash"]),
        (hash_bytes(config_path.read_bytes()), manifest["teacher_config_hash"]),
        (hash_bytes(motion_path.read_bytes()), group["motion_source_hash"]),
    )
    if any(actual != declared for actual, declared in expected):
        raise ValueError("recovery MJX route group asset binding mismatch")
    return {
        "schema_version": "rosclaw_soccer.recovery_mjx_route_training_job.v1",
        "route_manifest_hash": manifest["report_hash"],
        "route_group_hash": group["route_group_hash"],
        "route_group_index": group["group_index"],
        "motion_id": group["motion_id"],
        "entry_frame": group["entry_frame"],
        "successor_end_frame": group["successor_end_frame"],
        "time_dilation": group["time_dilation"],
        "minimum_episode_length": group["minimum_episode_length"],
        "snapshot_indices": list(group["snapshot_indices"]),
        "snapshot_hashes": list(group["snapshot_hashes"]),
        "training_role": group["training_role"],
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
    }


def build_recovery_mjx_route_manifest(
    *,
    snapshot_manifest_path: Path,
    development_report_path: Path,
    output_path: Path,
    config: RecoveryMJXRouteManifestConfig | None = None,
) -> dict[str, Any]:
    """Freeze one CPU-successful route for every exact recovery snapshot."""

    active = config or RecoveryMJXRouteManifestConfig()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    development_path = development_report_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if not snapshot_path.is_file() or not development_path.is_file():
        raise FileNotFoundError("recovery MJX route inputs are incomplete")
    if target.exists():
        raise ValueError("recovery MJX route manifest refuses to overwrite evidence")
    snapshots = load_recovery_snapshot_corpus(snapshot_path)
    if not snapshots or len({item.snapshot_hash for item in snapshots}) != len(snapshots):
        raise ValueError("recovery MJX routes require a unique snapshot corpus")
    if active.require_failed_snapshots and any(not item.failed for item in snapshots):
        raise ValueError("recovery MJX routes require physically failed snapshots")
    development = _verified_development_report(development_path)
    snapshot_file_hash = hash_bytes(snapshot_path.read_bytes())
    if development.get("snapshot_manifest_hash") != snapshot_file_hash:
        raise ValueError("recovery MJX route snapshot binding differs from development")
    schedule = development["post_skill_transfer"]["development_schedule"]
    raw_selected = schedule.get("selected_trials")
    if not isinstance(raw_selected, list) or len(raw_selected) != len(snapshots):
        raise ValueError("recovery MJX route schedule does not cover the corpus")
    selected = tuple(_trial_from_dict(dict(item)) for item in raw_selected)
    selected_by_snapshot = {item.snapshot_hash: item for item in selected}
    if len(selected_by_snapshot) != len(selected):
        raise ValueError("recovery MJX route schedule contains duplicate snapshots")
    if set(selected_by_snapshot) != {item.snapshot_hash for item in snapshots}:
        raise ValueError("recovery MJX route schedule snapshot coverage differs")

    routes: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for snapshot_index, snapshot in enumerate(snapshots):
        trial = selected_by_snapshot[snapshot.snapshot_hash]
        if (
            not trial.succeeded
            or not trial.finite_state
            or not trial.ready_handoff_triggered
            or trial.final_stable_sec < active.minimum_final_stable_sec
            or trial.physics_backend != active.required_trial_backend
            or trial.teacher_policy_hash != development.get("teacher_policy_hash")
        ):
            raise ValueError("recovery MJX route lacks a qualifying CPU success")
        group_contract = {
            "motion_id": trial.match.motion_id,
            "motion_source_hash": trial.match.source_hash,
            "entry_frame": trial.match.entry_frame,
            "successor_end_frame": trial.match.successor_end_frame,
            "time_dilation": trial.time_dilation,
            "teacher_policy_hash": trial.teacher_policy_hash,
        }
        group_key = str(hash_json(group_contract))
        route: dict[str, Any] = {
            "snapshot_index": snapshot_index,
            "snapshot_hash": snapshot.snapshot_hash,
            "posture_cluster": snapshot.posture_cluster,
            "stage": snapshot.stage,
            "source_body_hash": snapshot.body_hash,
            "source_physics_scene_hash": snapshot.physics_scene_hash,
            "source_policy_hash": snapshot.source_policy_hash,
            "source_config_hash": snapshot.source_config_hash,
            **group_contract,
            "match_hash": trial.match.match_hash,
            "search_config_hash": trial.match.search_config_hash,
            "cpu_trial_hash": trial.trial_hash,
            "cpu_trial_succeeded": True,
            "cpu_final_stable_sec": trial.final_stable_sec,
            "cpu_executed_sec": trial.executed_sec,
            "cpu_peak_root_angular_speed_rad_s": (trial.peak_root_angular_speed_rad_s),
            "cpu_physics_backend": trial.physics_backend,
            "route_group_key": group_key,
        }
        route["route_hash"] = hash_json(route)
        routes.append(route)
        grouped.setdefault(group_key, []).append(route)

    route_groups: list[dict[str, Any]] = []
    for group_index, group_key in enumerate(sorted(grouped)):
        members = sorted(grouped[group_key], key=lambda item: item["snapshot_index"])
        first = members[0]
        maximum_cpu_executed_sec = max(float(item["cpu_executed_sec"]) for item in members)
        minimum_episode_length = max(
            600,
            math.ceil(
                (maximum_cpu_executed_sec + active.episode_margin_sec) / active.control_dt_sec
            ),
        )
        if minimum_episode_length > 3_000:
            raise ValueError("recovery MJX route exceeds the bounded episode contract")
        group: dict[str, Any] = {
            "group_index": group_index,
            "route_group_key": group_key,
            "motion_id": first["motion_id"],
            "motion_source_hash": first["motion_source_hash"],
            "entry_frame": first["entry_frame"],
            "successor_end_frame": first["successor_end_frame"],
            "time_dilation": first["time_dilation"],
            "maximum_cpu_executed_sec": maximum_cpu_executed_sec,
            "minimum_episode_length": minimum_episode_length,
            "snapshot_indices": [item["snapshot_index"] for item in members],
            "snapshot_hashes": [item["snapshot_hash"] for item in members],
            "route_hashes": [item["route_hash"] for item in members],
            "training_role": "ROUTE_SPECIALIZED_PRIVILEGED_TEACHER_RESIDUAL",
        }
        group["route_group_hash"] = hash_json(group)
        route_groups.append(group)

    snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_mjx_route_manifest.v2",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "source_development_report_hash": development["report_hash"],
        "source_development_schedule_hash": schedule["schedule_hash"],
        "snapshot_manifest_hash": snapshot_file_hash,
        "snapshot_corpus_hash": snapshot_payload.get("corpus_hash"),
        "reference_library_hash": development.get("reference_library_hash"),
        "teacher_policy_hash": development.get("teacher_policy_hash"),
        "teacher_config_hash": development.get("teacher_config_hash"),
        "opentrack_commit": development.get("opentrack_commit"),
        "route_count": len(routes),
        "route_group_count": len(route_groups),
        "cpu_development_pass_rate": 1.0,
        "routes": routes,
        "route_groups": route_groups,
        "training_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO",
        "route_selection_uses_development_outcomes": True,
        "requires_stratified_fixed_seed_evaluation": True,
        "requires_reference_free_distillation": True,
        "requires_independent_cpu_mujoco_exam": True,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": "CPU_PROVEN_PRIVILEGED_ROUTES_FOR_MJX_TRAINING_ONLY",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target, report)
    return validate_recovery_mjx_route_manifest(target)


__all__ = [
    "RecoveryMJXRouteManifestConfig",
    "build_recovery_mjx_route_manifest",
    "resolve_recovery_mjx_route_group",
    "validate_recovery_mjx_route_manifest",
]
