"""Evidence contracts for a short-horizon recovery corrective teacher.

The teacher is deliberately simulation privileged: it may branch MuJoCo/MJX
state and compare counterfactual futures, while the student inputs stored in
the resulting corpus remain deployable proprioception only.  The corpus has no
promotion authority and preserves rejected searches as counterexamples.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_JOINT_COUNT = 29
_EFFECT_METRICS = (
    "root_body_backward_speed_cost",
    "root_body_lateral_speed_cost",
    "pelvis_yaw_speed_cost",
    "stability_deficit",
)
_ARCHIVE_ARRAYS = (
    "actor_observation",
    "baseline_action",
    "corrective_action_increment",
    "teacher_action",
    "teacher_plan",
    "baseline_cost",
    "teacher_cost",
    "cost_improvement_fraction",
    "teacher_accepted",
    "finite_rollout",
    "failure_state_index",
    "control_step",
    "baseline_effect_metrics",
    "teacher_effect_metrics",
    "action_effect_jacobian",
)


@dataclass(frozen=True)
class RecoveryCorrectiveTeacherConfig:
    """Bounded CEM and finite-difference contract for failure-state teaching."""

    state_count: int = 16
    horizon_steps: int = 20
    action_chunk_steps: int = 5
    candidate_count: int = 64
    elite_fraction: float = 0.125
    cem_iterations: int = 3
    initial_action_std: float = 0.15
    minimum_action_std: float = 0.015
    maximum_action_increment: float = 0.50
    finite_difference_increment: float = 0.10
    backward_cost_weight: float = 3.0
    lateral_cost_weight: float = 1.0
    yaw_cost_weight: float = 0.5
    stability_deficit_weight: float = 0.35
    action_magnitude_cost_weight: float = 0.03
    action_slew_cost_weight: float = 0.02
    minimum_cost_improvement_fraction: float = 0.01
    maximum_directional_cost_regression_fraction: float = 0.02
    maximum_directional_cost_regression_absolute: float = 0.002
    maximum_stability_deficit_regression_absolute: float = 0.0
    minimum_accepted_fraction: float = 0.25
    minimum_teacher_action_rms: float = 0.001
    required_gpu_count: int = 4
    random_seed: int = 5_500
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_corrective_teacher_config.v2"

    def __post_init__(self) -> None:
        finite_values = (
            self.elite_fraction,
            self.initial_action_std,
            self.minimum_action_std,
            self.maximum_action_increment,
            self.finite_difference_increment,
            self.backward_cost_weight,
            self.lateral_cost_weight,
            self.yaw_cost_weight,
            self.stability_deficit_weight,
            self.action_magnitude_cost_weight,
            self.action_slew_cost_weight,
            self.minimum_cost_improvement_fraction,
            self.maximum_directional_cost_regression_fraction,
            self.maximum_directional_cost_regression_absolute,
            self.maximum_stability_deficit_regression_absolute,
            self.minimum_accepted_fraction,
            self.minimum_teacher_action_rms,
        )
        if (
            isinstance(self.state_count, bool)
            or not 4 <= self.state_count <= 384
            or self.state_count % self.required_gpu_count
            or not 4 <= self.horizon_steps <= 80
            or not 1 <= self.action_chunk_steps <= self.horizon_steps
            or self.horizon_steps % self.action_chunk_steps
            or not 64 <= self.candidate_count <= 512
            or self.candidate_count % 4
            or not 0.02 <= self.elite_fraction <= 0.5
            or int(self.candidate_count * self.elite_fraction) < 2
            or not 1 <= self.cem_iterations <= 8
            or any(not math.isfinite(value) for value in finite_values)
            or not 0.01 <= self.minimum_action_std <= self.initial_action_std <= 0.5
            or not self.initial_action_std <= self.maximum_action_increment <= 1.0
            or not 0.01 <= self.finite_difference_increment <= self.maximum_action_increment
            or any(value < 0.0 for value in finite_values[5:11])
            or not 0.0 <= self.minimum_cost_improvement_fraction <= 0.5
            or not 0.0 <= self.maximum_directional_cost_regression_fraction <= 0.25
            or not 0.0 <= self.maximum_directional_cost_regression_absolute <= 0.1
            or not 0.0 <= self.maximum_stability_deficit_regression_absolute <= 0.25
            or not 0.0 <= self.minimum_accepted_fraction <= 1.0
            or not 0.0 < self.minimum_teacher_action_rms <= self.maximum_action_increment
            or self.required_gpu_count != 4
            or not 0 <= self.random_seed < 2**31
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.schema_version
            not in {
                "rosclaw_soccer.recovery_corrective_teacher_config.v1",
                "rosclaw_soccer.recovery_corrective_teacher_config.v2",
            }
        ):
            raise ValueError("recovery corrective teacher config is invalid")

    @property
    def action_chunk_count(self) -> int:
        return self.horizon_steps // self.action_chunk_steps

    @property
    def elite_count(self) -> int:
        return int(self.candidate_count * self.elite_fraction)

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray[Any, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _finite_array(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray[Any, Any]:
    array = np.asarray(value, dtype=np.dtype(np.float64) if dtype is None else dtype)
    if not np.all(np.isfinite(array)):
        raise ValueError("recovery corrective teacher array is non-finite")
    return array


def _summarize_action_effect_jacobian(
    jacobian: np.ndarray[Any, Any],
) -> dict[str, Any]:
    if jacobian.ndim != 3 or jacobian.shape[1:] != (len(_EFFECT_METRICS), _JOINT_COUNT):
        raise ValueError("recovery corrective teacher Jacobian shape is invalid")
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    maximum = np.maximum(singular_values[:, :1], 1.0e-12)
    effective_rank = np.sum(singular_values >= 0.05 * maximum, axis=1)
    joint_effect = np.linalg.norm(jacobian, axis=1)
    median_joint_effect = np.median(joint_effect, axis=0)
    active_threshold = max(float(np.max(median_joint_effect)) * 0.05, 1.0e-9)
    active_joint_indices = np.flatnonzero(median_joint_effect >= active_threshold)
    metric_sensitivity = np.median(np.linalg.norm(jacobian, axis=2), axis=0)
    return {
        "metric_names": list(_EFFECT_METRICS),
        "median_metric_sensitivity_l2": {
            name: float(value)
            for name, value in zip(_EFFECT_METRICS, metric_sensitivity, strict=True)
        },
        "median_effective_rank": float(np.median(effective_rank)),
        "minimum_effective_rank": int(np.min(effective_rank)),
        "maximum_effective_rank": int(np.max(effective_rank)),
        "active_joint_count": int(active_joint_indices.size),
        "active_joint_indices": [int(value) for value in active_joint_indices],
        "median_joint_effect_l2": [float(value) for value in median_joint_effect],
        "locally_controllable": bool(
            float(np.median(effective_rank)) >= 2.0 and active_joint_indices.size >= 4
        ),
    }


def _jacobian_summary_matches(
    observed: Any,
    expected: Mapping[str, Any],
) -> bool:
    """Compare derived linear-algebra evidence within machine precision.

    LAPACK reductions may choose a different last bit after an NPZ round trip
    on a larger batch.  Structural and discrete fields remain exact; only the
    derived floating-point medians receive a tight numerical tolerance.
    """

    if not isinstance(observed, dict) or set(observed) != set(expected):
        return False
    exact_names = (
        "metric_names",
        "minimum_effective_rank",
        "maximum_effective_rank",
        "active_joint_count",
        "active_joint_indices",
        "locally_controllable",
    )
    if any(observed.get(name) != expected.get(name) for name in exact_names):
        return False
    observed_metric = observed.get("median_metric_sensitivity_l2")
    expected_metric = expected.get("median_metric_sensitivity_l2")
    if (
        not isinstance(observed_metric, dict)
        or not isinstance(expected_metric, dict)
        or set(observed_metric) != set(expected_metric)
        or not all(
            math.isclose(
                float(observed_metric[name]),
                float(expected_metric[name]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            for name in expected_metric
        )
    ):
        return False
    return bool(
        math.isclose(
            float(observed.get("median_effective_rank", -1.0)),
            float(expected["median_effective_rank"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and np.allclose(
            np.asarray(observed.get("median_joint_effect_l2"), dtype=np.float64),
            np.asarray(expected["median_joint_effect_l2"], dtype=np.float64),
            rtol=1.0e-12,
            atol=1.0e-15,
        )
    )


def _teacher_retention_mask(
    *,
    baseline_effect: np.ndarray[Any, Any],
    teacher_effect: np.ndarray[Any, Any],
    config: RecoveryCorrectiveTeacherConfig,
) -> np.ndarray[Any, Any]:
    """Require v2 labels to improve without hiding directional regressions."""

    if config.schema_version == "rosclaw_soccer.recovery_corrective_teacher_config.v1":
        return np.ones((baseline_effect.shape[0],), dtype=np.bool_)
    directional_tolerance = np.maximum(
        np.abs(baseline_effect[:, :3]) * config.maximum_directional_cost_regression_fraction,
        config.maximum_directional_cost_regression_absolute,
    )
    directional_passed = np.all(
        teacher_effect[:, :3] <= baseline_effect[:, :3] + directional_tolerance,
        axis=1,
    )
    stability_passed = (
        teacher_effect[:, 3]
        <= baseline_effect[:, 3] + config.maximum_stability_deficit_regression_absolute
    )
    return np.asarray(directional_passed & stability_passed, dtype=np.bool_)


def write_recovery_corrective_teacher_evidence(
    *,
    output_dir: Path,
    config: RecoveryCorrectiveTeacherConfig,
    arrays: Mapping[str, Any],
    lineage: Mapping[str, Any],
    devices: tuple[str, ...],
    compiled_model_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Write an integrity-bound corrective corpus and diagnostic report."""

    destination = output_dir.expanduser().resolve()
    if destination.exists():
        raise ValueError("recovery corrective teacher refuses to overwrite evidence")
    required_lineage = {
        "failure_state_manifest_hash",
        "failure_state_manifest_file_hash",
        "failure_state_archive_hash",
        "parent_training_report_hash",
        "parent_checkpoint_hash",
        "teacher_checkpoint_hash",
        "snapshot_manifest_hash",
        "route_manifest_hash",
        "route_group_hash",
        "motion_archive_hash",
    }
    if set(lineage) != required_lineage or any(
        not isinstance(lineage[name], str) or not _HASH.fullmatch(str(lineage[name]))
        for name in required_lineage
    ):
        raise ValueError("recovery corrective teacher lineage is invalid")
    if len(devices) != config.required_gpu_count or len(set(devices)) != len(devices):
        raise ValueError("recovery corrective teacher device map is invalid")

    cast: dict[str, np.ndarray[Any, Any]] = {
        name: np.asarray(arrays[name]) for name in _ARCHIVE_ARRAYS if name in arrays
    }
    if set(cast) != set(_ARCHIVE_ARRAYS):
        raise ValueError("recovery corrective teacher arrays are incomplete")
    state_count = config.state_count
    observation = _finite_array(cast["actor_observation"], dtype=np.dtype(np.float32))
    baseline_action = _finite_array(cast["baseline_action"], dtype=np.dtype(np.float32))
    increment = _finite_array(cast["corrective_action_increment"], dtype=np.dtype(np.float32))
    teacher_action = _finite_array(cast["teacher_action"], dtype=np.dtype(np.float32))
    teacher_plan = _finite_array(cast["teacher_plan"], dtype=np.dtype(np.float32))
    baseline_cost = _finite_array(cast["baseline_cost"])
    teacher_cost = _finite_array(cast["teacher_cost"])
    improvement = _finite_array(cast["cost_improvement_fraction"])
    accepted = np.asarray(cast["teacher_accepted"], dtype=np.bool_)
    finite_rollout = np.asarray(cast["finite_rollout"], dtype=np.bool_)
    state_indices = np.asarray(cast["failure_state_index"], dtype=np.int32)
    control_steps = np.asarray(cast["control_step"], dtype=np.int32)
    baseline_effect = _finite_array(cast["baseline_effect_metrics"])
    teacher_effect = _finite_array(cast["teacher_effect_metrics"])
    jacobian = _finite_array(cast["action_effect_jacobian"])
    vector_shape = (state_count, _JOINT_COUNT)
    if (
        observation.ndim != 2
        or observation.shape[0] != state_count
        or observation.shape[1] < 1
        or baseline_action.shape != vector_shape
        or increment.shape != vector_shape
        or teacher_action.shape != vector_shape
        or teacher_plan.shape != (state_count, config.action_chunk_count, _JOINT_COUNT)
        or any(
            value.shape != (state_count,)
            for value in (
                baseline_cost,
                teacher_cost,
                improvement,
                accepted,
                finite_rollout,
                state_indices,
                control_steps,
            )
        )
        or baseline_effect.shape != (state_count, len(_EFFECT_METRICS))
        or teacher_effect.shape != baseline_effect.shape
        or jacobian.shape != (state_count, len(_EFFECT_METRICS), _JOINT_COUNT)
        or np.any(np.abs(baseline_action) > 1.0 + 1.0e-6)
        or np.any(np.abs(increment) > config.maximum_action_increment + 1.0e-6)
        or np.any(np.abs(teacher_plan) > config.maximum_action_increment + 1.0e-6)
        or np.any(np.abs(teacher_action) > 1.0 + 1.0e-6)
        or np.any(baseline_cost < 0.0)
        or np.any(teacher_cost < 0.0)
        or np.any(state_indices < 0)
        or np.any(control_steps < 0)
    ):
        raise ValueError("recovery corrective teacher array contract is invalid")
    expected_action = np.clip(baseline_action + increment, -1.0, 1.0)
    expected_increment = np.clip(baseline_action + teacher_plan[:, 0], -1.0, 1.0) - baseline_action
    expected_improvement = (baseline_cost - teacher_cost) / np.maximum(
        np.abs(baseline_cost), 1.0e-12
    )
    retention_passed = _teacher_retention_mask(
        baseline_effect=baseline_effect,
        teacher_effect=teacher_effect,
        config=config,
    )
    expected_accepted = (
        finite_rollout
        & retention_passed
        & (expected_improvement >= config.minimum_cost_improvement_fraction)
    )
    if (
        not np.allclose(teacher_action, expected_action, rtol=0.0, atol=2.0e-6)
        or not np.allclose(increment, expected_increment, rtol=0.0, atol=2.0e-6)
        or not np.allclose(improvement, expected_improvement, rtol=1.0e-6, atol=1.0e-7)
        or not np.array_equal(accepted, expected_accepted)
    ):
        raise ValueError("recovery corrective teacher labels are inconsistent")

    normalized_arrays: dict[str, np.ndarray[Any, Any]] = {
        "actor_observation": observation.astype(np.float32),
        "baseline_action": baseline_action.astype(np.float32),
        "corrective_action_increment": increment.astype(np.float32),
        "teacher_action": teacher_action.astype(np.float32),
        "teacher_plan": teacher_plan.astype(np.float32),
        "baseline_cost": baseline_cost.astype(np.float64),
        "teacher_cost": teacher_cost.astype(np.float64),
        "cost_improvement_fraction": improvement.astype(np.float64),
        "teacher_accepted": accepted,
        "finite_rollout": finite_rollout,
        "failure_state_index": state_indices,
        "control_step": control_steps,
        "baseline_effect_metrics": baseline_effect.astype(np.float64),
        "teacher_effect_metrics": teacher_effect.astype(np.float64),
        "action_effect_jacobian": jacobian.astype(np.float64),
    }
    destination.mkdir(parents=True)
    archive_path = destination / "corrective-teacher-corpus.npz"
    _atomic_npz(archive_path, normalized_arrays)
    accepted_count = int(np.sum(accepted))
    accepted_fraction = float(np.mean(accepted))
    authority_rms = float(np.sqrt(np.mean(np.square(increment))))
    accepted_improvements = improvement[accepted]
    windows: list[dict[str, Any]] = []
    for control_step in sorted({int(value) for value in control_steps.tolist()}):
        mask = control_steps == control_step
        windows.append(
            {
                "control_step": control_step,
                "sample_count": int(np.sum(mask)),
                "accepted_count": int(np.sum(accepted[mask])),
                "accepted_fraction": float(np.mean(accepted[mask])),
                "median_cost_improvement_fraction": float(np.median(improvement[mask])),
            }
        )
    jacobian_summary = _summarize_action_effect_jacobian(jacobian)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_corrective_teacher_evidence.v2",
        "config": asdict(config),
        "config_hash": config.config_hash,
        **dict(lineage),
        "compiled_model_contract": dict(compiled_model_contract),
        "rollout_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO_REQUIRED_FOR_PROMOTION",
        "search_algorithm": "SHORT_HORIZON_CHUNKED_CEM_WITH_ZERO_ACTION_CONTROL",
        "counterfactual_reset_contract": "IDENTICAL_EXACT_POLICY_CONTEXT_STATE",
        "actor_observation": "DEPLOYABLE_PROPRIOCEPTION_HISTORY_ONLY",
        "actor_observation_dim": int(observation.shape[1]),
        "teacher_privilege": "SIMULATION_STATE_BRANCHING_AND_COUNTERFACTUAL_ROLLOUT_ONLY",
        "label_semantics": "NORMALIZED_RESIDUAL_INCREMENT_OVER_FROZEN_PARENT",
        "devices": list(devices),
        "device_count": len(devices),
        "all_devices_used": True,
        "state_count": state_count,
        "unique_failure_state_count": int(np.unique(state_indices).size),
        "corpus_archive": archive_path.name,
        "corpus_archive_hash": hash_bytes(archive_path.read_bytes()),
        "corpus_arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in normalized_arrays.items()
        },
        "teacher_search": {
            "accepted_count": accepted_count,
            "rejected_counterexample_count": state_count - accepted_count,
            "accepted_fraction": accepted_fraction,
            "median_cost_improvement_fraction": float(np.median(improvement)),
            "median_accepted_cost_improvement_fraction": (
                float(np.median(accepted_improvements)) if accepted_count else 0.0
            ),
            "mean_corrective_action_increment_rms": authority_rms,
            "maximum_corrective_action_increment_rms": float(
                np.max(np.sqrt(np.mean(np.square(increment), axis=1)))
            ),
            "finite_rollout_fraction": float(np.mean(finite_rollout)),
            "retention_constraint_pass_fraction": float(np.mean(retention_passed)),
            "per_failure_window": windows,
        },
        "action_effect_jacobian": jacobian_summary,
        "teacher_retention_constraints": {
            "directional_cost_metrics": list(_EFFECT_METRICS[:3]),
            "maximum_directional_cost_regression_fraction": (
                config.maximum_directional_cost_regression_fraction
            ),
            "maximum_directional_cost_regression_absolute": (
                config.maximum_directional_cost_regression_absolute
            ),
            "maximum_stability_deficit_regression_absolute": (
                config.maximum_stability_deficit_regression_absolute
            ),
            "enforced_during_search": True,
        },
        "supervised_warm_start_eligible": bool(
            accepted_fraction >= config.minimum_accepted_fraction
            and authority_rms >= config.minimum_teacher_action_rms
            and bool(jacobian_summary["locally_controllable"])
        ),
        "counterexamples_preserved": True,
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    report_path = destination / "teacher-report.json"
    _atomic_json(report_path, report)
    return validate_recovery_corrective_teacher_evidence(report_path)


def validate_recovery_corrective_teacher_evidence(path: Path) -> dict[str, Any]:
    """Validate corpus bytes, authority ceiling, shapes, and derived labels."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery corrective teacher evidence is invalid")
    report_hash = payload.pop("report_hash", None)
    valid_hash = report_hash == hash_json(payload)
    payload["report_hash"] = report_hash
    config_payload = payload.get("config")
    try:
        config = (
            RecoveryCorrectiveTeacherConfig(**config_payload)
            if isinstance(config_payload, dict)
            else None
        )
    except (TypeError, ValueError):
        config = None
    archive_name = payload.get("corpus_archive")
    archive_path = path.parent / str(archive_name)
    archive_valid = bool(
        isinstance(archive_name, str)
        and archive_name == "corrective-teacher-corpus.npz"
        and archive_path.is_file()
        and payload.get("corpus_archive_hash") == hash_bytes(archive_path.read_bytes())
    )
    arrays_valid = False
    derived_valid = False
    summary_valid = False
    if archive_valid and config is not None:
        try:
            with np.load(archive_path, allow_pickle=False) as archive:
                arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
            arrays_valid = set(arrays) == set(_ARCHIVE_ARRAYS) and all(
                np.all(np.isfinite(value))
                for name, value in arrays.items()
                if name not in {"teacher_accepted", "finite_rollout"}
            )
            state_count = config.state_count
            shape_valid = bool(
                arrays["actor_observation"].ndim == 2
                and arrays["actor_observation"].shape[0] == state_count
                and arrays["baseline_action"].shape == (state_count, _JOINT_COUNT)
                and arrays["corrective_action_increment"].shape == (state_count, _JOINT_COUNT)
                and arrays["teacher_action"].shape == (state_count, _JOINT_COUNT)
                and arrays["teacher_plan"].shape
                == (state_count, config.action_chunk_count, _JOINT_COUNT)
                and arrays["action_effect_jacobian"].shape
                == (state_count, len(_EFFECT_METRICS), _JOINT_COUNT)
            )
            expected_improvement = (arrays["baseline_cost"] - arrays["teacher_cost"]) / np.maximum(
                np.abs(arrays["baseline_cost"]), 1.0e-12
            )
            retention_passed = _teacher_retention_mask(
                baseline_effect=arrays["baseline_effect_metrics"],
                teacher_effect=arrays["teacher_effect_metrics"],
                config=config,
            )
            expected_accepted = (
                arrays["finite_rollout"].astype(np.bool_)
                & retention_passed
                & (expected_improvement >= config.minimum_cost_improvement_fraction)
            )
            expected_action = np.clip(
                arrays["baseline_action"] + arrays["corrective_action_increment"],
                -1.0,
                1.0,
            )
            expected_increment = (
                np.clip(
                    arrays["baseline_action"] + arrays["teacher_plan"][:, 0],
                    -1.0,
                    1.0,
                )
                - arrays["baseline_action"]
            )
            derived_valid = bool(
                shape_valid
                and np.allclose(
                    arrays["cost_improvement_fraction"],
                    expected_improvement,
                    rtol=1.0e-6,
                    atol=1.0e-7,
                )
                and np.array_equal(arrays["teacher_accepted"], expected_accepted)
                and np.allclose(arrays["teacher_action"], expected_action, atol=2.0e-6)
                and np.allclose(
                    arrays["corrective_action_increment"], expected_increment, atol=2.0e-6
                )
                and np.all(
                    np.abs(arrays["corrective_action_increment"])
                    <= config.maximum_action_increment + 1.0e-6
                )
            )
            accepted_count = int(np.sum(expected_accepted))
            accepted_fraction = float(np.mean(expected_accepted))
            authority_rms = float(
                np.sqrt(np.mean(np.square(arrays["corrective_action_increment"])))
            )
            jacobian_summary = _summarize_action_effect_jacobian(arrays["action_effect_jacobian"])
            search_summary = payload.get("teacher_search")
            evidence_schema = payload.get("schema_version")
            summary_valid = bool(
                isinstance(search_summary, dict)
                and search_summary.get("accepted_count") == accepted_count
                and search_summary.get("rejected_counterexample_count")
                == state_count - accepted_count
                and math.isclose(
                    float(search_summary.get("accepted_fraction", -1.0)),
                    accepted_fraction,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and math.isclose(
                    float(search_summary.get("mean_corrective_action_increment_rms", -1.0)),
                    authority_rms,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and (
                    evidence_schema == "rosclaw_soccer.recovery_corrective_teacher_evidence.v1"
                    or math.isclose(
                        float(search_summary.get("retention_constraint_pass_fraction", -1.0)),
                        float(np.mean(retention_passed)),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                )
                and _jacobian_summary_matches(
                    payload.get("action_effect_jacobian"), jacobian_summary
                )
                and payload.get("supervised_warm_start_eligible")
                is bool(
                    accepted_fraction >= config.minimum_accepted_fraction
                    and authority_rms >= config.minimum_teacher_action_rms
                    and bool(jacobian_summary["locally_controllable"])
                )
            )
        except (KeyError, OSError, TypeError, ValueError):
            arrays_valid = False
            derived_valid = False
            summary_valid = False
    lineage_names = (
        "failure_state_manifest_hash",
        "failure_state_manifest_file_hash",
        "failure_state_archive_hash",
        "parent_training_report_hash",
        "parent_checkpoint_hash",
        "teacher_checkpoint_hash",
        "snapshot_manifest_hash",
        "route_manifest_hash",
        "route_group_hash",
        "motion_archive_hash",
    )
    evidence_schema = payload.get("schema_version")
    retention_contract = payload.get("teacher_retention_constraints")
    retention_contract_valid = bool(
        (
            evidence_schema == "rosclaw_soccer.recovery_corrective_teacher_evidence.v1"
            and retention_contract is None
        )
        or (
            evidence_schema == "rosclaw_soccer.recovery_corrective_teacher_evidence.v2"
            and config is not None
            and isinstance(retention_contract, dict)
            and retention_contract.get("directional_cost_metrics") == list(_EFFECT_METRICS[:3])
            and retention_contract.get("maximum_directional_cost_regression_fraction")
            == config.maximum_directional_cost_regression_fraction
            and retention_contract.get("maximum_directional_cost_regression_absolute")
            == config.maximum_directional_cost_regression_absolute
            and retention_contract.get("maximum_stability_deficit_regression_absolute")
            == config.maximum_stability_deficit_regression_absolute
            and retention_contract.get("enforced_during_search") is True
        )
    )
    if (
        not valid_hash
        or config is None
        or evidence_schema
        not in {
            "rosclaw_soccer.recovery_corrective_teacher_evidence.v1",
            "rosclaw_soccer.recovery_corrective_teacher_evidence.v2",
        }
        or payload.get("config_hash")
        != (
            hash_json(config_payload)
            if config.schema_version == "rosclaw_soccer.recovery_corrective_teacher_config.v1"
            else config.config_hash
        )
        or not retention_contract_valid
        or not all(_HASH.fullmatch(str(payload.get(name, ""))) for name in lineage_names)
        or payload.get("rollout_backend") != "MUJOCO_MJX"
        or payload.get("physics_truth_backend") != "CPU_MUJOCO_REQUIRED_FOR_PROMOTION"
        or payload.get("actor_observation") != "DEPLOYABLE_PROPRIOCEPTION_HISTORY_ONLY"
        or payload.get("teacher_privilege")
        != "SIMULATION_STATE_BRANCHING_AND_COUNTERFACTUAL_ROLLOUT_ONLY"
        or payload.get("label_semantics") != "NORMALIZED_RESIDUAL_INCREMENT_OVER_FROZEN_PARENT"
        or payload.get("device_count") != 4
        or len(payload.get("devices", ())) != 4
        or payload.get("all_devices_used") is not True
        or payload.get("counterexamples_preserved") is not True
        or payload.get("deployment_candidate") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("promotion_authority") != "NONE"
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
        or payload.get("hardware_command_sent") is not False
        or not archive_valid
        or not arrays_valid
        or not derived_valid
        or not summary_valid
    ):
        raise ValueError("recovery corrective teacher evidence is invalid")
    return payload


__all__ = [
    "RecoveryCorrectiveTeacherConfig",
    "validate_recovery_corrective_teacher_evidence",
    "write_recovery_corrective_teacher_evidence",
]
