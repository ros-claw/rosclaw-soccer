"""Failure-memory reachability audit for ``athlete-foundation-v1``.

This module freezes the S49--S75 gate-sweep era as evidence and builds the
first deliberately different experiment: a content-bound 256-state bank plus
paired 1x/4x adapter-authority counterfactuals.  It is diagnostic-only.  The
legacy failure archives do not contain contact topology, so the generated
bank says so explicitly instead of pretending that qpos alone proves contact.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.recovery_mjx import (
    validate_recovery_mjx_failure_state_exam_report,
    validate_recovery_mjx_failure_state_manifest,
)

_ARCHIVE_KEYS = (
    "qpos",
    "qvel",
    "control_step",
    "environment_index",
    "handoff_frozen",
    "trajectory_step",
    "trajectory_initial_step",
    "root_body_backward_speed_mps",
    "root_body_lateral_speed_mps",
    "pelvis_yaw_speed_rad_s",
    "last_motor_targets",
    "last_teacher_action",
    "last_residual",
    "proprioception_history",
    "phase_repeat",
)
_CONTEXT_FEATURES = (
    "qpos",
    "qvel",
    "trajectory_step",
    "trajectory_initial_step",
    "handoff_frozen",
    "last_motor_targets",
    "last_teacher_action",
    "last_residual",
    "proprioception_history",
    "phase_repeat",
)
_SHARED_BINDINGS = (
    "source_actor_checkpoint_hash",
    "source_actor_config_hash",
    "source_route_manifest_hash",
    "source_route_group_hash",
    "teacher_checkpoint_hash",
    "motion_archive_hash",
    "snapshot_manifest_hash",
)


class ReachabilityMainline(StrEnum):
    """Preliminary learner-routing result from the paired authority probe."""

    EXPAND_RESIDUAL_AUTHORITY = "EXPAND_RESIDUAL_AUTHORITY"
    TRAIN_PARENT_FREE_EXPERT_ORACLE = "TRAIN_PARENT_FREE_EXPERT_ORACLE"
    RESIDUAL_REMAINS_PLAUSIBLE = "RESIDUAL_REMAINS_PLAUSIBLE"
    FIX_ENVIRONMENT_REWARD_OR_ACTUATOR = "FIX_ENVIRONMENT_REWARD_OR_ACTUATOR"
    DISTILL_HISTORY_AWARE_STUDENT = "DISTILL_HISTORY_AWARE_STUDENT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class RecoveryReachabilityBankConfig:
    state_count: int = 256
    random_seed: int = 7601
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_reachability_bank_config.v1"

    def __post_init__(self) -> None:
        if (
            not 32 <= self.state_count <= 2_048
            or self.state_count % 4
            or not 0 <= self.random_seed < 2**31
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery reachability bank config is invalid")


@dataclass(frozen=True)
class RecoveryReachabilityDecisionConfig:
    oracle_success_threshold: float = 0.70
    residual_low_success_threshold: float = 0.20
    expanded_residual_success_threshold: float = 0.50
    expanded_residual_minimum_gain: float = 0.20
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False

    def __post_init__(self) -> None:
        values = (
            self.oracle_success_threshold,
            self.residual_low_success_threshold,
            self.expanded_residual_success_threshold,
            self.expanded_residual_minimum_gain,
        )
        if (
            any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values)
            or self.residual_low_success_threshold >= self.oracle_success_threshold
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery reachability decision config is invalid")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _posture_proxy(qpos: NDArray[np.floating[Any]]) -> str:
    quaternion = np.asarray(qpos[3:7], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1.0e-6:
        return "INVALID_QUATERNION"
    w, x, y, z = quaternion / norm
    upright = 1.0 - 2.0 * (x * x + y * y)
    forward_up = 2.0 * (x * z - w * y)
    lateral_up = 2.0 * (y * z + w * x)
    if float(qpos[2]) >= 0.68 and upright >= 0.85:
        return "UPRIGHT"
    if abs(lateral_up) >= 0.55:
        return "RIGHT_SIDE" if lateral_up > 0.0 else "LEFT_SIDE"
    if abs(forward_up) >= 0.55:
        return "SUPINE" if forward_up > 0.0 else "PRONE"
    if float(qpos[2]) >= 0.42 or upright > 0.25:
        return "KNEELING_OR_SUPPORTED_PROXY"
    return "AMBIGUOUS_FALLEN"


def _event_window_proxy(control_step: int) -> str:
    if control_step <= 200:
        return "EARLY_DRIFT"
    if control_step <= 300:
        return "IMPACT_OR_FIRST_SUPPORT"
    if control_step <= 400:
        return "SUPPORT_TRANSFER"
    if control_step <= 500:
        return "COM_LIFT"
    return "CAPTURE_OR_READY_TRANSITION"


def _momentum_proxy(qvel: NDArray[np.floating[Any]]) -> str:
    angular_speed = float(np.linalg.norm(np.asarray(qvel[3:6], dtype=np.float64)))
    if angular_speed < 1.0:
        return "LOW"
    if angular_speed < 2.0:
        return "MEDIUM"
    if angular_speed < 4.0:
        return "HIGH"
    return "EXTREME"


def _state_identity(arrays: Mapping[str, NDArray[Any]], index: int) -> str:
    payload = b"".join(
        np.ascontiguousarray(arrays[name][index]).tobytes()
        for name in (
            "qpos",
            "qvel",
            "control_step",
            "trajectory_step",
            "trajectory_initial_step",
            "handoff_frozen",
            "last_motor_targets",
            "last_teacher_action",
            "last_residual",
            "proprioception_history",
            "phase_repeat",
        )
    )
    return str(hash_bytes(payload))


def build_recovery_reachability_bank(
    *,
    source_manifest_paths: Sequence[Path],
    output_dir: Path,
    source_checkout_path: Path,
    config: RecoveryReachabilityBankConfig | None = None,
) -> dict[str, Any]:
    """Build a deterministic, source-balanced bank from exact failure memory."""

    active = config or RecoveryReachabilityBankConfig()
    if len(source_manifest_paths) < 2:
        raise ValueError("reachability bank requires at least two failure-memory sources")
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("reachability bank output must be new and external")

    paths = tuple(path.expanduser().resolve() for path in source_manifest_paths)
    if len(set(paths)) != len(paths):
        raise ValueError("reachability bank source manifests must be unique")
    manifests = tuple(validate_recovery_mjx_failure_state_manifest(path) for path in paths)
    if any(
        manifest.get("schema_version") != "rosclaw_soccer.recovery_mjx_failure_state_manifest.v2"
        for manifest in manifests
    ):
        raise ValueError("reachability bank requires exact policy-context sources")
    first = manifests[0]
    if any(
        any(manifest.get(name) != first.get(name) for name in _SHARED_BINDINGS)
        or hash_json(manifest.get("compiled_model_contract"))
        != hash_json(first.get("compiled_model_contract"))
        for manifest in manifests[1:]
    ):
        raise ValueError("reachability bank sources have incompatible physical lineage")

    source_arrays: list[dict[str, NDArray[Any]]] = []
    candidates: list[dict[str, Any]] = []
    identities: set[str] = set()
    for source_index, (path, source_manifest) in enumerate(zip(paths, manifests, strict=True)):
        archive_path = path.parent / str(source_manifest["state_archive"])
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in _ARCHIVE_KEYS}
        source_arrays.append(arrays)
        for row_index in range(int(source_manifest["collected_state_count"])):
            identity = _state_identity(arrays, row_index)
            if identity in identities:
                continue
            identities.add(identity)
            qpos = arrays["qpos"][row_index]
            qvel = arrays["qvel"][row_index]
            control_step = int(arrays["control_step"][row_index])
            posture = _posture_proxy(qpos)
            event_window = _event_window_proxy(control_step)
            momentum = _momentum_proxy(qvel)
            candidates.append(
                {
                    "source_index": source_index,
                    "source_manifest_hash": source_manifest["report_hash"],
                    "source_manifest_file_hash": hash_bytes(path.read_bytes()),
                    "source_row_index": row_index,
                    "source_environment_index": int(arrays["environment_index"][row_index]),
                    "control_step": control_step,
                    "posture_proxy": posture,
                    "event_window_proxy": event_window,
                    "angular_momentum_proxy": momentum,
                    "contact_topology": "UNRECORDED_IN_SOURCE_FAILURE_BANK_V2",
                    "state_identity": identity,
                    "stratum": (
                        source_index,
                        event_window,
                        posture,
                        momentum,
                    ),
                }
            )
    if len(candidates) < active.state_count:
        raise ValueError("reachability bank does not contain enough unique exact states")

    rng = np.random.default_rng(active.random_seed)
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        buckets[row["stratum"]].append(row)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)
    selected: list[dict[str, Any]] = []
    ordered_keys = sorted(buckets, key=lambda value: tuple(str(item) for item in value))
    while len(selected) < active.state_count:
        progressed = False
        for key in ordered_keys:
            if buckets[key] and len(selected) < active.state_count:
                selected.append(buckets[key].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) != active.state_count:
        raise RuntimeError("reachability bank stratified selection is incomplete")

    output_arrays: dict[str, NDArray[Any]] = {}
    for name in _ARCHIVE_KEYS:
        array_rows: list[NDArray[Any]] = [
            np.asarray(source_arrays[int(row["source_index"])][name][int(row["source_row_index"])])
            for row in selected
        ]
        output_arrays[name] = np.stack(array_rows) if array_rows[0].ndim else np.asarray(array_rows)
    output_arrays["environment_index"] = np.arange(active.state_count, dtype=np.int32)

    destination.mkdir(parents=True)
    archive_path = destination / "failure-window-states.npz"
    _atomic_npz(archive_path, output_arrays)
    selection_rows = []
    for output_index, row in enumerate(selected):
        serializable = {name: value for name, value in row.items() if name != "stratum"}
        serializable["output_index"] = output_index
        selection_rows.append(serializable)
    source_hashes = [manifest["report_hash"] for manifest in manifests]
    source_file_hashes = [hash_bytes(path.read_bytes()) for path in paths]

    def source_values(name: str) -> list[Any]:
        return [manifest[name] for manifest in manifests]

    history_shape = list(output_arrays["proprioception_history"].shape)
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_mjx_failure_state_manifest.v2",
        "config": {
            **asdict(active),
            "num_environments": active.state_count,
            "source_manifest_count": len(manifests),
        },
        "source_failure_window_plan_hash": hash_json(
            source_values("source_failure_window_plan_hash")
        ),
        "source_failure_window_plan_file_hash": hash_json(
            source_values("source_failure_window_plan_file_hash")
        ),
        "source_training_report_hash": hash_json(source_values("source_training_report_hash")),
        "source_actor_checkpoint_hash": first["source_actor_checkpoint_hash"],
        "source_actor_config_hash": first["source_actor_config_hash"],
        "source_route_manifest_hash": first["source_route_manifest_hash"],
        "source_route_group_hash": first["source_route_group_hash"],
        "teacher_checkpoint_hash": first["teacher_checkpoint_hash"],
        "motion_archive_hash": first["motion_archive_hash"],
        "snapshot_manifest_hash": first["snapshot_manifest_hash"],
        "compiled_model_contract": first["compiled_model_contract"],
        "rollout_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO_REQUIRED_FOR_PROMOTION",
        "deterministic_actor": True,
        "full_route_reset": True,
        "requested_collection_steps": sorted(
            {int(value) for value in output_arrays["control_step"].tolist()}
        ),
        "collected_state_count": active.state_count,
        "state_archive": archive_path.name,
        "state_archive_hash": hash_bytes(archive_path.read_bytes()),
        "qpos_shape": list(output_arrays["qpos"].shape),
        "qvel_shape": list(output_arrays["qvel"].shape),
        "proprioception_history_shape": history_shape,
        "context_features_collected": list(_CONTEXT_FEATURES),
        "source_manifest_hashes": source_hashes,
        "source_manifest_file_hashes": source_file_hashes,
        "selection_protocol": "ROUND_ROBIN_RARE_STRATA_FIRST_WITH_SEEDED_INTRA_STRATUM_SHUFFLE",
        "stratification_dimensions": [
            "SOURCE_MANIFEST",
            "EVENT_WINDOW_PROXY",
            "POSTURE_PROXY",
            "ANGULAR_SPEED_PROXY_FOR_ANGULAR_MOMENTUM",
        ],
        "contact_topology_coverage": "MISSING_REQUIRES_CPU_MUJOCO_FORWARD_ENRICHMENT",
        "stratification_complete": False,
        "selection_rows": selection_rows,
        "curriculum_use_only": True,
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    manifest["report_hash"] = hash_json(manifest)
    manifest_path = destination / "failure-state-manifest.json"
    _atomic_json(manifest_path, manifest)
    return cast(
        dict[str, Any], validate_recovery_mjx_failure_state_manifest(manifest_path)
    )


def decide_recovery_mainline(
    *,
    bounded_residual_success_rate: float,
    expanded_residual_success_rate: float,
    privileged_oracle_success_rate: float | None = None,
    teacher_success_rate: float | None = None,
    student_success_rate: float | None = None,
    config: RecoveryReachabilityDecisionConfig | None = None,
) -> ReachabilityMainline:
    """Apply the breakthrough document's hard learner-routing rules."""

    active = config or RecoveryReachabilityDecisionConfig()
    values = (
        bounded_residual_success_rate,
        expanded_residual_success_rate,
        privileged_oracle_success_rate,
        teacher_success_rate,
        student_success_rate,
    )
    if any(
        value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0)
        for value in values
    ):
        raise ValueError("reachability success rates must be finite fractions")
    if (
        teacher_success_rate is not None
        and student_success_rate is not None
        and teacher_success_rate >= active.oracle_success_threshold
        and student_success_rate <= active.residual_low_success_threshold
    ):
        return ReachabilityMainline.DISTILL_HISTORY_AWARE_STUDENT
    if privileged_oracle_success_rate is not None:
        if privileged_oracle_success_rate < active.oracle_success_threshold:
            return ReachabilityMainline.FIX_ENVIRONMENT_REWARD_OR_ACTUATOR
        if bounded_residual_success_rate <= active.residual_low_success_threshold:
            return ReachabilityMainline.TRAIN_PARENT_FREE_EXPERT_ORACLE
    expanded_gain = expanded_residual_success_rate - bounded_residual_success_rate
    if (
        expanded_residual_success_rate >= active.expanded_residual_success_threshold
        and expanded_gain >= active.expanded_residual_minimum_gain
    ):
        return ReachabilityMainline.EXPAND_RESIDUAL_AUTHORITY
    if (
        bounded_residual_success_rate <= active.residual_low_success_threshold
        and expanded_residual_success_rate <= active.residual_low_success_threshold
    ):
        return ReachabilityMainline.TRAIN_PARENT_FREE_EXPERT_ORACLE
    if bounded_residual_success_rate >= active.expanded_residual_success_threshold:
        return ReachabilityMainline.RESIDUAL_REMAINS_PLAUSIBLE
    return ReachabilityMainline.INCONCLUSIVE


def write_preliminary_reachability_decision(
    *,
    bounded_exam_path: Path,
    expanded_exam_path: Path,
    output_path: Path,
    config: RecoveryReachabilityDecisionConfig | None = None,
) -> dict[str, Any]:
    """Bind paired 1x/4x exams and emit a preliminary architecture decision."""

    active = config or RecoveryReachabilityDecisionConfig()
    bounded_path = bounded_exam_path.expanduser().resolve()
    expanded_path = expanded_exam_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if target.exists():
        raise ValueError("reachability decision refuses to overwrite evidence")
    bounded = validate_recovery_mjx_failure_state_exam_report(bounded_path)
    expanded = validate_recovery_mjx_failure_state_exam_report(expanded_path)
    bounded_config = bounded["config"]
    expanded_config = expanded["config"]
    shared_fields = (
        "failure_state_manifest_hash",
        "parent_checkpoint_hash",
        "candidate_checkpoint_hash",
        "route_manifest_hash",
        "route_group_hash",
        "teacher_checkpoint_hash",
    )
    parent_metrics_bounded = bounded["parent_metrics"]
    parent_metrics_expanded = expanded["parent_metrics"]
    parent_metric_deltas = {
        name: float(parent_metrics_expanded[name]) - float(parent_metrics_bounded[name])
        for name in parent_metrics_bounded
        if name != "episode_count"
    }
    episode_count = int(parent_metrics_bounded["episode_count"])
    parent_reproducibility_passed = bool(
        parent_metrics_expanded["episode_count"] == episode_count
        and abs(parent_metric_deltas["success_rate"]) <= 1.0 / episode_count
        and abs(parent_metric_deltas["stable_fraction"]) <= 0.01
        and abs(parent_metric_deltas["ready_fraction"]) <= 0.01
        and abs(parent_metric_deltas["root_body_backward_speed_mps"]) <= 0.01
        and abs(parent_metric_deltas["root_body_lateral_speed_mps"]) <= 0.01
        and abs(parent_metric_deltas["pelvis_yaw_speed_rad_s"]) <= 0.01
        and abs(parent_metric_deltas["non_success_termination_rate"]) <= 0.01
    )
    if (
        any(bounded[name] != expanded[name] for name in shared_fields)
        or not parent_reproducibility_passed
        or bounded_config.get("candidate_adapter_gain", 1.0) != 1.0
        or expanded_config.get("candidate_adapter_gain") != 4.0
        or any(
            bounded_config.get(name) != expanded_config.get(name)
            for name in ("num_environments", "horizon_steps", "random_seed")
        )
    ):
        raise ValueError("reachability 1x/4x exams are not a paired counterfactual")
    bounded_success = float(bounded["candidate_metrics"]["success_rate"])
    expanded_success = float(expanded["candidate_metrics"]["success_rate"])
    decision = decide_recovery_mainline(
        bounded_residual_success_rate=bounded_success,
        expanded_residual_success_rate=expanded_success,
        config=active,
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_reachability_preliminary_decision.v1",
        "config": asdict(active),
        "bounded_exam_file_hash": hash_bytes(bounded_path.read_bytes()),
        "bounded_exam_report_hash": bounded["report_hash"],
        "expanded_exam_file_hash": hash_bytes(expanded_path.read_bytes()),
        "expanded_exam_report_hash": expanded["report_hash"],
        "failure_state_manifest_hash": bounded["failure_state_manifest_hash"],
        "parent_checkpoint_hash": bounded["parent_checkpoint_hash"],
        "candidate_checkpoint_hash": bounded["candidate_checkpoint_hash"],
        "bounded_residual_success_rate": bounded_success,
        "expanded_residual_success_rate": expanded_success,
        "success_rate_delta": expanded_success - bounded_success,
        "parent_cross_process_metric_deltas": parent_metric_deltas,
        "parent_cross_process_reproducibility_passed": parent_reproducibility_passed,
        "lockstep_authority_counterfactual_required": True,
        "preliminary_mainline": decision.value,
        "success_metric_boundary": "LEGACY_100_STEP_STABLE_READY_PROXY",
        "missing_audit_arms": [
            "PARENT_FREE_PRIVILEGED_PPO_ORACLE",
            "MODERN_MUJOCO_PORT_OF_HUMANUP_OR_HOST",
        ],
        "final_architecture_decision_allowed": False,
        "claim_boundary": "AUTHORITY_COUNTERFACTUAL_ONLY_NOT_FULL_SUCCESSOR_READY_AUDIT",
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target, report)
    return report


def write_recovery_moe_reachability_aggregate(
    *,
    shard_report_paths: Sequence[Path],
    failure_state_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Bind all GPU shards and preserve route-conditional reachability truth.

    A high aggregate rate is not sufficient when a dominant upright stratum
    hides total failure of the get-up route.  This writer therefore validates
    every state identity against the frozen bank and makes the weakest route,
    rather than the micro-average, control the architecture decision.
    """

    paths = tuple(path.expanduser().resolve() for path in shard_report_paths)
    manifest_path = failure_state_manifest_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if target.exists():
        raise ValueError("recovery MoE aggregate refuses to overwrite evidence")
    if len(paths) != 4 or len(set(paths)) != 4:
        raise ValueError("recovery MoE aggregate requires four unique GPU shards")
    manifest = validate_recovery_mjx_failure_state_manifest(manifest_path)
    expected_count = int(manifest["collected_state_count"])
    selection_rows = manifest.get("selection_rows")
    if not isinstance(selection_rows, list) or len(selection_rows) != expected_count:
        raise ValueError("recovery MoE bank does not bind per-state identities")

    shards: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("recovery MoE shard must be a JSON object")
        claimed_hash = payload.get("report_hash")
        hash_payload = dict(payload)
        hash_payload.pop("report_hash", None)
        if (
            payload.get("schema_version")
            != "rosclaw_soccer.mjlab_recovery_moe_reachability_probe.v1"
            or claimed_hash != hash_json(hash_payload)
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("promotion_eligible") is not False
        ):
            raise ValueError("recovery MoE shard contract is invalid")
        shards.append(payload)

    shared_fields = (
        "contract_hash",
        "checkpoint_hash",
        "source_hash",
        "body_hash",
        "physics_scene_hash",
        "failure_state_manifest_hash",
        "failure_state_manifest_file_hash",
        "source_compiled_model_contract_hash",
        "routing",
        "capture_router_config",
        "termination_semantics",
        "getup_reference_phase_alignment",
        "getup_reference_phase_estimator_weights",
    )
    first = shards[0]
    if any(
        any(shard.get(name) != first.get(name) for name in shared_fields) for shard in shards[1:]
    ):
        raise ValueError("recovery MoE shards do not share one physical contract")
    if first.get("failure_state_manifest_hash") != manifest.get("report_hash") or first.get(
        "failure_state_manifest_file_hash"
    ) != hash_bytes(manifest_path.read_bytes()):
        raise ValueError("recovery MoE shards are not bound to the selected bank")

    devices = {str(shard.get("physics_device")) for shard in shards}
    if devices != {"cuda:0", "cuda:1", "cuda:2", "cuda:3"}:
        raise ValueError("recovery MoE audit did not execute on four bound CUDA devices")
    seeds = [int(shard["random_seed"]) for shard in shards]
    if len(set(seeds)) != 4 or any(not 0 <= seed < 2**31 for seed in seeds):
        raise ValueError("recovery MoE audit does not bind four valid shard seeds")
    state_results: list[dict[str, Any]] = []
    for shard in shards:
        start = int(shard["failure_state_start_index"])
        stop = int(shard["failure_state_stop_index"])
        rows = shard.get("state_results")
        if (
            not isinstance(rows, list)
            or stop - start != int(shard["environment_count"])
            or len(rows) != stop - start
        ):
            raise ValueError("recovery MoE shard state range is invalid")
        athlete_count = sum(row.get("route") == "ATHLETE" for row in rows)
        capture_count = sum(row.get("route") == "CAPTURE" for row in rows)
        getup_count = sum(row.get("route") == "GET_UP" for row in rows)
        success_count = sum(row.get("final_stable_recovery") is True for row in rows)
        handoff_count = sum(row.get("handoff_completed") is True for row in rows)
        if (
            athlete_count != int(shard["athlete_route_count"])
            or capture_count != int(shard.get("capture_route_count", 0))
            or getup_count != int(shard["getup_route_count"])
            or success_count != int(shard["final_stable_recovery_count"])
            or handoff_count != int(shard["handoff_completed_count"])
        ):
            raise ValueError("recovery MoE shard summary differs from its state results")
        state_results.extend(dict(row) for row in rows)
    state_results.sort(key=lambda row: int(row["failure_state_index"]))
    indices = [int(row["failure_state_index"]) for row in state_results]
    if indices != list(range(expected_count)):
        raise ValueError("recovery MoE shards do not exactly cover the failure bank")
    if any(
        row.get("state_identity") != selection_rows[index].get("state_identity")
        for index, row in enumerate(state_results)
    ):
        raise ValueError("recovery MoE state identity differs from the frozen bank")

    route_metrics: dict[str, dict[str, Any]] = {}
    for route in ("ATHLETE", "CAPTURE", "GET_UP"):
        rows = [row for row in state_results if row.get("route") == route]
        success_count = sum(row.get("final_stable_recovery") is True for row in rows)
        handoff_count = sum(row.get("handoff_completed") is True for row in rows)
        route_metrics[route] = {
            "state_count": len(rows),
            "final_stable_recovery_count": success_count,
            "final_stable_recovery_rate": success_count / len(rows) if rows else None,
            "handoff_completed_count": handoff_count,
            "handoff_completed_rate": handoff_count / len(rows) if rows else None,
        }
    if sum(metric["state_count"] for metric in route_metrics.values()) != expected_count:
        raise ValueError("recovery MoE shard contains an unknown route")
    overall_success_count = sum(row.get("final_stable_recovery") is True for row in state_results)
    weakest_populated_route_rate = min(
        float(metric["final_stable_recovery_rate"])
        for metric in route_metrics.values()
        if metric["state_count"]
    )
    getup_rate = route_metrics["GET_UP"]["final_stable_recovery_rate"]
    athlete_rate = route_metrics["ATHLETE"]["final_stable_recovery_rate"]
    if athlete_rate is not None and athlete_rate >= 0.90 and getup_rate == 0.0:
        architecture_decision = "KEEP_ATHLETE_FOUNDATION_TRAIN_PREDECESSOR_CONDITIONED_GETUP"
    elif weakest_populated_route_rate >= 0.70:
        architecture_decision = "MOE_RECOVERY_REACHABILITY_SUPPORTED"
    else:
        architecture_decision = "TRAIN_ROUTE_EXPERTS_BEFORE_END_TO_END_RETRY"
    if weakest_populated_route_rate >= 0.70:
        main_finding = "ROUTED_ATHLETE_CAPTURE_GETUP_RECOVERY_IS_REACHABLE_ON_THE_BOUND_BANK"
    elif athlete_rate is not None and athlete_rate >= 0.90 and getup_rate == 0.0:
        main_finding = (
            "UPRIGHT_MOMENTUM_CAPTURE_IS_REACHABLE_BUT_IMPORTED_GETUP_DOES_NOT_COVER_"
            "THE_REAL_FAILURE_PREDECESSOR_DISTRIBUTION"
        )
    else:
        main_finding = "AT_LEAST_ONE_RECOVERY_ROUTE_REMAINS_BELOW_REACHABILITY_THRESHOLD"

    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_moe_reachability_aggregate.v1",
        "failure_state_manifest_hash": manifest["report_hash"],
        "failure_state_manifest_file_hash": hash_bytes(manifest_path.read_bytes()),
        "shard_file_hashes": [hash_bytes(path.read_bytes()) for path in paths],
        "shard_report_hashes": [shard["report_hash"] for shard in shards],
        "physics_devices": sorted(devices),
        "random_seeds": seeds,
        "expert_contract": {name: first.get(name) for name in shared_fields},
        "state_count": expected_count,
        "overall_final_stable_recovery_count": overall_success_count,
        "overall_final_stable_recovery_rate": overall_success_count / expected_count,
        "route_metrics": route_metrics,
        "getup_reference_phase_alignment": first.get("getup_reference_phase_alignment"),
        "getup_reference_phase_estimator_weights": first.get(
            "getup_reference_phase_estimator_weights"
        ),
        "weakest_populated_route_recovery_rate": weakest_populated_route_rate,
        "architecture_decision": architecture_decision,
        "main_finding": main_finding,
        "micro_average_may_not_authorize_claim": True,
        "bank_contact_topology_coverage": manifest.get("contact_topology_coverage"),
        "bank_stratification_complete": manifest.get("stratification_complete"),
        "required_next_experiments": [
            "CPU_MUJOCO_CONTACT_ENRICHED_POST_DIVE_FAILURE_BANK",
            "PARENT_FREE_PRIVILEGED_RECOVERY_ORACLE",
            "GETUP_PHASE_ESTIMATOR_OR_PREDECESSOR_CONDITIONED_EXPERT",
            "LOCKSTEP_FULL_CHAIN_SUCCESSOR_READY_EXAM",
        ],
        "successor_boundary": "LOCOMOTION_READY_PROXY_NOT_GOALKEEPER_READY",
        "claim_boundary": "CROSS_SCENE_REACHABILITY_DIAGNOSTIC_NOT_PROMOTION_EVIDENCE",
        "state_results": state_results,
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target, report)
    return report


def write_recovery_phase_alignment_ab_decision(
    *,
    baseline_aggregate_path: Path,
    aligned_aggregate_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Emit a fail-closed, state-paired phase-entry A/B decision."""

    baseline_path = baseline_aggregate_path.expanduser().resolve()
    aligned_path = aligned_aggregate_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if target.exists() or baseline_path == aligned_path:
        raise ValueError("recovery phase-alignment A/B evidence paths are invalid")

    def load(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("recovery phase-alignment aggregate must be a JSON object")
        claimed_hash = payload.get("report_hash")
        hash_payload = dict(payload)
        hash_payload.pop("report_hash", None)
        if (
            payload.get("schema_version") != "rosclaw_soccer.recovery_moe_reachability_aggregate.v1"
            or claimed_hash != hash_json(hash_payload)
            or payload.get("promotion_eligible") is not False
            or payload.get("activation_ceiling") != "SIM_ONLY"
        ):
            raise ValueError("recovery phase-alignment aggregate contract is invalid")
        return payload

    baseline = load(baseline_path)
    aligned = load(aligned_path)
    if (
        baseline.get("failure_state_manifest_hash") != aligned.get("failure_state_manifest_hash")
        or baseline.get("state_count") != aligned.get("state_count")
        or baseline.get("physics_devices") != aligned.get("physics_devices")
        or baseline.get("random_seeds") != aligned.get("random_seeds")
    ):
        raise ValueError("recovery phase-alignment A/B is not state/seed/device paired")
    baseline_contract = dict(baseline.get("expert_contract", {}))
    aligned_contract = dict(aligned.get("expert_contract", {}))
    baseline_alignment = baseline_contract.pop("getup_reference_phase_alignment", None)
    aligned_alignment = aligned_contract.pop("getup_reference_phase_alignment", None)
    baseline_contract.pop("getup_reference_phase_estimator_weights", None)
    aligned_weights = aligned_contract.pop("getup_reference_phase_estimator_weights", None)
    if (
        baseline_alignment is not False
        or aligned_alignment is not True
        or baseline_contract != aligned_contract
        or not isinstance(aligned_weights, dict)
    ):
        raise ValueError("recovery phase-alignment A/B changed more than the entry adapter")
    baseline_rows = baseline.get("state_results")
    aligned_rows = aligned.get("state_results")
    if not isinstance(baseline_rows, list) or not isinstance(aligned_rows, list):
        raise ValueError("recovery phase-alignment A/B lacks per-state results")
    paired_rows = []
    for baseline_row, aligned_row in zip(baseline_rows, aligned_rows, strict=True):
        shared = ("failure_state_index", "state_identity", "route")
        if any(baseline_row.get(name) != aligned_row.get(name) for name in shared):
            raise ValueError("recovery phase-alignment A/B state identity changed")
        paired_rows.append(
            {
                "failure_state_index": baseline_row["failure_state_index"],
                "state_identity": baseline_row["state_identity"],
                "route": baseline_row["route"],
                "baseline_success": baseline_row["final_stable_recovery"],
                "aligned_success": aligned_row["final_stable_recovery"],
                "aligned_initial_reference_frame": aligned_row["initial_reference_frame"],
            }
        )
    if len(paired_rows) != int(baseline["state_count"]):
        raise ValueError("recovery phase-alignment A/B state count changed")

    route_deltas: dict[str, dict[str, Any]] = {}
    for route in ("ATHLETE", "GET_UP"):
        rows = [row for row in paired_rows if row["route"] == route]
        baseline_success = sum(row["baseline_success"] is True for row in rows)
        aligned_success = sum(row["aligned_success"] is True for row in rows)
        route_deltas[route] = {
            "state_count": len(rows),
            "baseline_success_count": baseline_success,
            "baseline_success_rate": baseline_success / len(rows) if rows else None,
            "aligned_success_count": aligned_success,
            "aligned_success_rate": aligned_success / len(rows) if rows else None,
            "success_count_delta": aligned_success - baseline_success,
        }
    regressions = sum(
        row["baseline_success"] is True and row["aligned_success"] is not True
        for row in paired_rows
    )
    improvements = sum(
        row["baseline_success"] is not True and row["aligned_success"] is True
        for row in paired_rows
    )
    getup = route_deltas["GET_UP"]
    confirmed = bool(
        getup["baseline_success_rate"] is not None
        and getup["baseline_success_rate"] <= 0.20
        and getup["aligned_success_rate"] >= 0.70
        and regressions == 0
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_phase_alignment_ab_decision.v1",
        "baseline_aggregate_file_hash": hash_bytes(baseline_path.read_bytes()),
        "baseline_aggregate_report_hash": baseline["report_hash"],
        "aligned_aggregate_file_hash": hash_bytes(aligned_path.read_bytes()),
        "aligned_aggregate_report_hash": aligned["report_hash"],
        "failure_state_manifest_hash": baseline["failure_state_manifest_hash"],
        "physics_devices": baseline["physics_devices"],
        "random_seeds": baseline["random_seeds"],
        "phase_estimator_weights": aligned_weights,
        "route_deltas": route_deltas,
        "state_improvement_count": improvements,
        "state_regression_count": regressions,
        "decision": (
            "PHASE_ENTRY_ADAPTER_BREAKTHROUGH_CONFIRMED"
            if confirmed
            else "PHASE_ENTRY_ADAPTER_NOT_CONFIRMED"
        ),
        "paired_state_results": paired_rows,
        "claim_boundary": "EXACT_FAILURE_BANK_PHASE_ENTRY_AB_NOT_FULL_CHAIN_PROMOTION",
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target, report)
    return report


__all__ = [
    "ReachabilityMainline",
    "RecoveryReachabilityBankConfig",
    "RecoveryReachabilityDecisionConfig",
    "build_recovery_reachability_bank",
    "decide_recovery_mainline",
    "write_recovery_moe_reachability_aggregate",
    "write_recovery_phase_alignment_ab_decision",
    "write_preliminary_reachability_decision",
]
