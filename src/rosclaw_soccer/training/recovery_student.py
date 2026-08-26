"""Proprio-only recovery student data and deployment contracts.

The teacher may use a reference trajectory during data generation.  The
student contract deliberately excludes reference pose, reference velocity,
reference phase and teacher identity, and predicts bounded absolute 29-DoF
motor targets so deployment does not need reference-conditioned action
semantics.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_JOINT_COUNT = 29
_PROPRIO_DIM = 93


def _valid_hash(value: str) -> bool:
    return bool(_SHA256.fullmatch(value))


def _finite_array(
    value: NDArray[np.floating[Any]] | Sequence[float],
    *,
    shape: tuple[int, ...],
    name: str,
) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return np.asarray(array, dtype=np.float32)


def _array_contract(array: NDArray[Any]) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "data_hash": hash_bytes(contiguous.tobytes()),
    }


@dataclass(frozen=True)
class RecoveryProprioceptionSpec:
    """Frozen deployable observation schema with no reference signal."""

    history_steps: int = 8
    joint_count: int = _JOINT_COUNT
    observation_dim: int = _PROPRIO_DIM
    gyro_scale: float = 0.05
    joint_velocity_scale: float = 0.05
    features: tuple[str, ...] = (
        "projected_gravity_body_3",
        "pelvis_gyro_scaled_3",
        "joint_position_from_default_29",
        "joint_velocity_scaled_29",
        "last_absolute_motor_target_29",
    )
    forbidden_features: tuple[str, ...] = (
        "reference_joint_position",
        "reference_joint_velocity",
        "reference_root_height",
        "reference_feet_height",
        "reference_phase",
        "teacher_id",
    )
    output_semantics: str = "ABSOLUTE_JOINT_POSITION_TARGET_NORMALIZED_TO_LIMITS"
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_proprioception_spec.v1"

    def __post_init__(self) -> None:
        if (
            not 2 <= self.history_steps <= 64
            or self.joint_count != _JOINT_COUNT
            or self.observation_dim != _PROPRIO_DIM
            or not math.isfinite(self.gyro_scale)
            or not math.isfinite(self.joint_velocity_scale)
            or min(self.gyro_scale, self.joint_velocity_scale) <= 0.0
            or len(self.features) != len(set(self.features))
            or len(self.forbidden_features) != len(set(self.forbidden_features))
            or set(self.features) & set(self.forbidden_features)
            or any(not _IDENTIFIER.fullmatch(item) for item in self.features)
            or any(not _IDENTIFIER.fullmatch(item) for item in self.forbidden_features)
            or self.output_semantics != "ABSOLUTE_JOINT_POSITION_TARGET_NORMALIZED_TO_LIMITS"
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery proprioception spec is invalid")

    @property
    def spec_hash(self) -> str:
        return str(hash_json(asdict(self)))


def build_recovery_proprioception(
    *,
    projected_gravity_body: NDArray[np.floating[Any]] | Sequence[float],
    pelvis_gyro_rad_s: NDArray[np.floating[Any]] | Sequence[float],
    joint_position_rad: NDArray[np.floating[Any]] | Sequence[float],
    joint_velocity_rad_s: NDArray[np.floating[Any]] | Sequence[float],
    last_motor_target_rad: NDArray[np.floating[Any]] | Sequence[float],
    default_joint_position_rad: NDArray[np.floating[Any]] | Sequence[float],
    spec: RecoveryProprioceptionSpec | None = None,
) -> NDArray[np.float32]:
    """Build one deployable frame in the frozen feature order."""

    active = spec or RecoveryProprioceptionSpec()
    gravity = _finite_array(
        projected_gravity_body,
        shape=(3,),
        name="projected gravity",
    )
    gravity_norm = float(np.linalg.norm(gravity))
    if not 0.95 <= gravity_norm <= 1.05:
        raise ValueError("projected gravity must be normalized")
    gyro = _finite_array(pelvis_gyro_rad_s, shape=(3,), name="pelvis gyro")
    joint_position = _finite_array(
        joint_position_rad,
        shape=(_JOINT_COUNT,),
        name="joint position",
    )
    joint_velocity = _finite_array(
        joint_velocity_rad_s,
        shape=(_JOINT_COUNT,),
        name="joint velocity",
    )
    last_target = _finite_array(
        last_motor_target_rad,
        shape=(_JOINT_COUNT,),
        name="last motor target",
    )
    default = _finite_array(
        default_joint_position_rad,
        shape=(_JOINT_COUNT,),
        name="default joint position",
    )
    result = np.concatenate(
        (
            gravity,
            gyro * active.gyro_scale,
            joint_position - default,
            joint_velocity * active.joint_velocity_scale,
            last_target,
        )
    ).astype(np.float32)
    if result.shape != (active.observation_dim,) or not np.all(np.isfinite(result)):
        raise ValueError("recovery proprioception frame is invalid")
    return np.asarray(result, dtype=np.float32)


def normalize_absolute_motor_targets(
    targets_rad: NDArray[np.floating[Any]],
    *,
    joint_lower_rad: NDArray[np.floating[Any]] | Sequence[float],
    joint_upper_rad: NDArray[np.floating[Any]] | Sequence[float],
) -> NDArray[np.float32]:
    targets = np.asarray(targets_rad, dtype=np.float32)
    if targets.ndim not in {1, 2} or targets.shape[-1] != _JOINT_COUNT:
        raise ValueError("absolute motor targets have an invalid shape")
    lower = _finite_array(joint_lower_rad, shape=(_JOINT_COUNT,), name="joint lower bound")
    upper = _finite_array(joint_upper_rad, shape=(_JOINT_COUNT,), name="joint upper bound")
    if np.any(upper <= lower) or not np.all(np.isfinite(targets)):
        raise ValueError("absolute motor target bounds are invalid")
    center = 0.5 * (lower + upper)
    radius = 0.5 * (upper - lower)
    normalized = (targets - center) / radius
    return np.asarray(np.clip(normalized, -1.0, 1.0), dtype=np.float32)


def denormalize_absolute_motor_targets(
    normalized_targets: NDArray[np.floating[Any]],
    *,
    joint_lower_rad: NDArray[np.floating[Any]] | Sequence[float],
    joint_upper_rad: NDArray[np.floating[Any]] | Sequence[float],
) -> NDArray[np.float32]:
    normalized = np.asarray(normalized_targets, dtype=np.float32)
    if (
        normalized.ndim not in {1, 2}
        or normalized.shape[-1] != _JOINT_COUNT
        or not np.all(np.isfinite(normalized))
    ):
        raise ValueError("normalized motor targets have an invalid shape")
    lower = _finite_array(joint_lower_rad, shape=(_JOINT_COUNT,), name="joint lower bound")
    upper = _finite_array(joint_upper_rad, shape=(_JOINT_COUNT,), name="joint upper bound")
    if np.any(upper <= lower):
        raise ValueError("absolute motor target bounds are invalid")
    center = 0.5 * (lower + upper)
    radius = 0.5 * (upper - lower)
    return np.asarray(
        center + np.clip(normalized, -1.0, 1.0) * radius,
        dtype=np.float32,
    )


@dataclass(frozen=True)
class RecoveryTeacherEpisode:
    """One successful privileged-teacher rollout converted to deployable IO."""

    episode_id: str
    base_snapshot_hash: str
    initial_snapshot_hash: str
    fixed_route_trial_hash: str
    perturbation_hash: str | None
    proprio: NDArray[np.float32]
    absolute_motor_targets_rad: NDArray[np.float32]
    ready_handoff: NDArray[np.bool_]
    time_dilation: int
    teacher_succeeded: bool
    rollout_controller: str = "PRIVILEGED_TEACHER"
    rollout_succeeded: bool = True
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.recovery_teacher_episode.v1"

    def __post_init__(self) -> None:
        proprio = np.asarray(self.proprio, dtype=np.float32)
        targets = np.asarray(self.absolute_motor_targets_rad, dtype=np.float32)
        ready = np.asarray(self.ready_handoff, dtype=np.bool_)
        if (
            not _IDENTIFIER.fullmatch(self.episode_id)
            or any(
                not _valid_hash(value)
                for value in (
                    self.base_snapshot_hash,
                    self.initial_snapshot_hash,
                    self.fixed_route_trial_hash,
                )
            )
            or (self.perturbation_hash is not None and not _valid_hash(self.perturbation_hash))
            or proprio.ndim != 2
            or proprio.shape[1] != _PROPRIO_DIM
            or targets.shape != (proprio.shape[0], _JOINT_COUNT)
            or ready.shape != (proprio.shape[0],)
            or proprio.shape[0] < 100
            or not np.all(np.isfinite(proprio))
            or not np.all(np.isfinite(targets))
            or not 1 <= self.time_dilation <= 4
            or not self.teacher_succeeded
            or self.rollout_controller not in {"PRIVILEGED_TEACHER", "MIXED_STUDENT_TEACHER"}
            or not isinstance(self.rollout_succeeded, bool)
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_command_sent
        ):
            raise ValueError("recovery teacher episode is invalid")
        object.__setattr__(self, "proprio", proprio)
        object.__setattr__(self, "absolute_motor_targets_rad", targets)
        object.__setattr__(self, "ready_handoff", ready)

    @property
    def episode_hash(self) -> str:
        return str(
            hash_json(
                self.metadata()
                | {
                    "proprio": _array_contract(self.proprio),
                    "absolute_motor_targets_rad": _array_contract(self.absolute_motor_targets_rad),
                    "ready_handoff": _array_contract(self.ready_handoff),
                }
            )
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "base_snapshot_hash": self.base_snapshot_hash,
            "initial_snapshot_hash": self.initial_snapshot_hash,
            "fixed_route_trial_hash": self.fixed_route_trial_hash,
            "perturbation_hash": self.perturbation_hash,
            "time_dilation": self.time_dilation,
            "teacher_succeeded": self.teacher_succeeded,
            "rollout_controller": self.rollout_controller,
            "rollout_succeeded": self.rollout_succeeded,
            "activation_ceiling": self.activation_ceiling,
            "hardware_command_sent": self.hardware_command_sent,
        }


@dataclass(frozen=True)
class RecoveryDistillationCorpus:
    proprio: NDArray[np.float32]
    absolute_motor_targets_rad: NDArray[np.float32]
    ready_handoff: NDArray[np.bool_]
    episode_index: NDArray[np.int32]
    control_step: NDArray[np.int32]
    default_joint_position_rad: NDArray[np.float32]
    joint_lower_rad: NDArray[np.float32]
    joint_upper_rad: NDArray[np.float32]
    rows: tuple[Mapping[str, Any], ...]
    manifest_hash: str
    proprioception_spec: RecoveryProprioceptionSpec

    def __post_init__(self) -> None:
        count = self.proprio.shape[0]
        if (
            self.proprio.shape != (count, _PROPRIO_DIM)
            or self.absolute_motor_targets_rad.shape != (count, _JOINT_COUNT)
            or self.ready_handoff.shape != (count,)
            or self.episode_index.shape != (count,)
            or self.control_step.shape != (count,)
            or not _valid_hash(self.manifest_hash)
        ):
            raise ValueError("recovery distillation corpus arrays are invalid")


def write_recovery_distillation_corpus(
    *,
    episodes: Sequence[RecoveryTeacherEpisode],
    output_dir: Path,
    corpus_name: str,
    proprioception_spec: RecoveryProprioceptionSpec,
    teacher_policy_hash: str,
    body_hash: str,
    physics_scene_hash: str,
    development_report_hash: str,
    default_joint_position_rad: NDArray[np.floating[Any]] | Sequence[float],
    joint_lower_rad: NDArray[np.floating[Any]] | Sequence[float],
    joint_upper_rad: NDArray[np.floating[Any]] | Sequence[float],
) -> dict[str, Any]:
    if (
        not episodes
        or len({item.episode_id for item in episodes}) != len(episodes)
        or not _IDENTIFIER.fullmatch(corpus_name)
        or any(
            not _valid_hash(value)
            for value in (
                teacher_policy_hash,
                body_hash,
                physics_scene_hash,
                development_report_hash,
            )
        )
    ):
        raise ValueError("recovery distillation corpus identity is invalid")
    default = _finite_array(
        default_joint_position_rad,
        shape=(_JOINT_COUNT,),
        name="default joint position",
    ).astype(np.float32)
    lower = _finite_array(
        joint_lower_rad,
        shape=(_JOINT_COUNT,),
        name="joint lower bound",
    ).astype(np.float32)
    upper = _finite_array(
        joint_upper_rad,
        shape=(_JOINT_COUNT,),
        name="joint upper bound",
    ).astype(np.float32)
    if np.any(default < lower) or np.any(default > upper) or np.any(upper <= lower):
        raise ValueError("recovery distillation joint bounds are invalid")
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / f"{corpus_name}.npz"
    manifest_path = destination / f"{corpus_name}.json"
    if archive_path.exists() or manifest_path.exists():
        raise ValueError("recovery distillation writer refuses to overwrite")

    proprio = np.concatenate([item.proprio for item in episodes]).astype(np.float32)
    targets = np.concatenate([item.absolute_motor_targets_rad for item in episodes]).astype(
        np.float32
    )
    ready = np.concatenate([item.ready_handoff for item in episodes]).astype(np.uint8)
    episode_index = np.concatenate(
        [
            np.full(item.proprio.shape[0], index, dtype=np.int32)
            for index, item in enumerate(episodes)
        ]
    )
    control_step = np.concatenate(
        [np.arange(item.proprio.shape[0], dtype=np.int32) for item in episodes]
    )
    arrays: dict[str, NDArray[Any]] = {
        "proprio": proprio,
        "absolute_motor_targets_rad": targets,
        "ready_handoff": ready,
        "episode_index": episode_index,
        "control_step": control_step,
        "default_joint_position_rad": default,
        "joint_lower_rad": lower,
        "joint_upper_rad": upper,
    }
    temporary_archive = destination / f".{corpus_name}.npz.tmp"
    with temporary_archive.open("wb") as stream:
        np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
    temporary_archive.replace(archive_path)
    rows: list[dict[str, Any]] = []
    offset = 0
    for index, episode in enumerate(episodes):
        count = episode.proprio.shape[0]
        rows.append(
            episode.metadata()
            | {
                "episode_index": index,
                "start_row": offset,
                "row_count": count,
                "episode_hash": episode.episode_hash,
            }
        )
        offset += count
    manifest: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_distillation_corpus.v1",
        "corpus_name": corpus_name,
        "archive": archive_path.name,
        "archive_hash": hash_bytes(archive_path.read_bytes()),
        "array_contracts": {name: _array_contract(value) for name, value in arrays.items()},
        "episode_count": len(episodes),
        "sample_count": proprio.shape[0],
        "rows": rows,
        "proprioception_spec": asdict(proprioception_spec),
        "proprioception_spec_hash": proprioception_spec.spec_hash,
        "teacher_policy_hash": teacher_policy_hash,
        "body_hash": body_hash,
        "physics_scene_hash": physics_scene_hash,
        "development_report_hash": development_report_hash,
        "contains_reference_features": False,
        "target_semantics": proprioception_spec.output_semantics,
        "training_use_only": True,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    manifest["manifest_hash"] = hash_json(manifest)
    temporary_manifest = destination / f".{corpus_name}.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def load_recovery_distillation_corpus(
    manifest_path: Path,
) -> RecoveryDistillationCorpus:
    resolved = manifest_path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    declared_hash = payload.pop("manifest_hash", None)
    if (
        payload.get("schema_version") != "rosclaw_soccer.recovery_distillation_corpus.v1"
        or declared_hash != hash_json(payload)
        or payload.get("contains_reference_features") is not False
        or payload.get("training_use_only") is not True
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("recovery distillation manifest integrity failed")
    raw_spec = payload.get("proprioception_spec")
    if not isinstance(raw_spec, dict):
        raise ValueError("recovery distillation proprioception spec is absent")
    raw_spec = dict(raw_spec)
    raw_spec["features"] = tuple(raw_spec["features"])
    raw_spec["forbidden_features"] = tuple(raw_spec["forbidden_features"])
    spec = RecoveryProprioceptionSpec(**raw_spec)
    if spec.spec_hash != payload.get("proprioception_spec_hash"):
        raise ValueError("recovery distillation proprioception spec hash mismatch")
    archive_path = resolved.parent / str(payload.get("archive"))
    if hash_bytes(archive_path.read_bytes()) != payload.get("archive_hash"):
        raise ValueError("recovery distillation archive hash mismatch")
    expected_names = {
        "proprio",
        "absolute_motor_targets_rad",
        "ready_handoff",
        "episode_index",
        "control_step",
        "default_joint_position_rad",
        "joint_lower_rad",
        "joint_upper_rad",
    }
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_names:
            raise ValueError("recovery distillation archive arrays are incomplete")
        arrays = {name: np.asarray(archive[name]) for name in expected_names}
    contracts = payload.get("array_contracts")
    if not isinstance(contracts, dict) or any(
        contracts.get(name) != _array_contract(value) for name, value in arrays.items()
    ):
        raise ValueError("recovery distillation array contract mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != payload.get("episode_count"):
        raise ValueError("recovery distillation episode rows are invalid")
    count = int(payload.get("sample_count", -1))
    if arrays["proprio"].shape[0] != count:
        raise ValueError("recovery distillation sample count is invalid")
    return RecoveryDistillationCorpus(
        proprio=np.asarray(arrays["proprio"], dtype=np.float32),
        absolute_motor_targets_rad=np.asarray(
            arrays["absolute_motor_targets_rad"], dtype=np.float32
        ),
        ready_handoff=np.asarray(arrays["ready_handoff"], dtype=np.bool_),
        episode_index=np.asarray(arrays["episode_index"], dtype=np.int32),
        control_step=np.asarray(arrays["control_step"], dtype=np.int32),
        default_joint_position_rad=np.asarray(
            arrays["default_joint_position_rad"], dtype=np.float32
        ),
        joint_lower_rad=np.asarray(arrays["joint_lower_rad"], dtype=np.float32),
        joint_upper_rad=np.asarray(arrays["joint_upper_rad"], dtype=np.float32),
        rows=tuple(MappingProxyType(dict(row)) for row in rows),
        manifest_hash=str(declared_hash),
        proprioception_spec=spec,
    )


def recovery_teacher_episodes_from_corpus(
    corpus: RecoveryDistillationCorpus,
) -> tuple[RecoveryTeacherEpisode, ...]:
    """Rehydrate verified rows without weakening the pickle-free boundary."""

    episodes: list[RecoveryTeacherEpisode] = []
    for row in corpus.rows:
        start = int(row["start_row"])
        count = int(row["row_count"])
        stop = start + count
        episodes.append(
            RecoveryTeacherEpisode(
                episode_id=str(row["episode_id"]),
                base_snapshot_hash=str(row["base_snapshot_hash"]),
                initial_snapshot_hash=str(row["initial_snapshot_hash"]),
                fixed_route_trial_hash=str(row["fixed_route_trial_hash"]),
                perturbation_hash=(
                    None if row.get("perturbation_hash") is None else str(row["perturbation_hash"])
                ),
                proprio=np.asarray(corpus.proprio[start:stop], dtype=np.float32),
                absolute_motor_targets_rad=np.asarray(
                    corpus.absolute_motor_targets_rad[start:stop], dtype=np.float32
                ),
                ready_handoff=np.asarray(corpus.ready_handoff[start:stop], dtype=np.bool_),
                time_dilation=int(row["time_dilation"]),
                teacher_succeeded=bool(row["teacher_succeeded"]),
                rollout_controller=str(row.get("rollout_controller", "PRIVILEGED_TEACHER")),
                rollout_succeeded=bool(row.get("rollout_succeeded", True)),
            )
        )
    return tuple(episodes)


__all__ = [
    "RecoveryDistillationCorpus",
    "RecoveryProprioceptionSpec",
    "RecoveryTeacherEpisode",
    "build_recovery_proprioception",
    "denormalize_absolute_motor_targets",
    "load_recovery_distillation_corpus",
    "normalize_absolute_motor_targets",
    "recovery_teacher_episodes_from_corpus",
    "write_recovery_distillation_corpus",
]
