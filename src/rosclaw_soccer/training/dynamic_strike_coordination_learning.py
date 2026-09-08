"""Train a bounded S209 coordination actor from verified physics probes."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.dynamic_strike_coordination import (
    STRIKE_COORDINATION_FEATURES,
    DynamicStrikeCoordinationActor,
    StrikeCoordinationObservation,
)
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_FEATURE_STD_FLOOR = np.asarray((0.03, 0.03, 0.03, 0.03, 0.03, 0.10), dtype=np.float64)


@dataclass(frozen=True)
class DynamicStrikeCoordinationDataset:
    observations: NDArray[np.float64]
    actions: NDArray[np.float64]
    source_index: NDArray[np.int64]
    source_probe_paths: tuple[str, ...]
    source_report_hashes: tuple[str, ...]
    source_trajectory_digests: tuple[str, ...]
    dataset_snapshot_hash: str

    def __post_init__(self) -> None:
        count = len(self.observations)
        if (
            self.observations.shape != (count, len(STRIKE_COORDINATION_FEATURES))
            or self.actions.shape != (count, 2)
            or self.source_index.shape != (count,)
            or count < len(self.source_probe_paths)
            or len(self.source_probe_paths) != len(self.source_report_hashes)
            or len(self.source_probe_paths) != len(self.source_trajectory_digests)
            or not np.all(np.isfinite(self.observations))
            or not np.all(np.isfinite(self.actions))
            or np.any(self.source_index < 0)
            or np.any(self.source_index >= len(self.source_probe_paths))
            or not self.dataset_snapshot_hash.startswith("sha256:")
        ):
            raise ValueError("dynamic strike coordination dataset is invalid")


def load_dynamic_strike_coordination_dataset(
    probe_directories: tuple[Path, ...],
) -> DynamicStrikeCoordinationDataset:
    """Load only successful teachers and verify every persisted byte binding."""

    if len(probe_directories) < 2:
        raise ValueError("at least two successful physics probes are required")
    observation_parts: list[NDArray[np.float64]] = []
    action_parts: list[NDArray[np.float64]] = []
    source_parts: list[NDArray[np.int64]] = []
    paths: list[str] = []
    report_hashes: list[str] = []
    trajectory_digests: list[str] = []
    for source_index, raw_root in enumerate(probe_directories):
        root = raw_root.expanduser().resolve()
        report_path = root / "probe.json"
        report = _read_json(report_path)
        stored_report_hash = str(report.get("report_hash", ""))
        hash_payload = dict(report)
        hash_payload.pop("report_hash", None)
        if stored_report_hash != hash_json(hash_payload):
            raise ValueError(f"probe report hash mismatch: {report_path}")
        if (
            report.get("schema_version") != "rosclaw_soccer.dynamic_strike_coordination_probe.v1"
            or report.get("activation_ceiling") != "SIM_ONLY"
            or report.get("hardware_command_sent") is not False
            or report.get("world_safe") is not True
            or report.get("shot_projection", {}).get("whole_ball_inside_goal") is not True
            or report.get("assessment", {}).get("phase_sequence") != [1, 2, 3, 4, 5, 6]
            or report.get("assessment", {})
            .get("gates", {})
            .get("physical_foot_strike_in_strike_phase")
            is not True
            or report.get("assessment", {}).get("gates", {}).get("stable_recovery_completed")
            is not True
        ):
            raise ValueError(f"probe is not a successful SIM-only teacher: {report_path}")
        artifact = report.get("trajectory_artifact")
        if not isinstance(artifact, dict) or artifact.get("file") != "trajectory.npz":
            raise ValueError(f"probe trajectory manifest is invalid: {report_path}")
        trajectory_path = root / "trajectory.npz"
        if hash_bytes(trajectory_path.read_bytes()) != artifact.get("file_hash"):
            raise ValueError(f"probe trajectory file hash mismatch: {trajectory_path}")
        with np.load(trajectory_path, allow_pickle=False) as archive:
            trajectory = {name: np.asarray(archive[name]) for name in archive.files}
        digest = trajectory_digest(trajectory)
        if digest != artifact.get("trajectory_digest") or digest != report.get("trajectory_digest"):
            raise ValueError(f"probe trajectory digest mismatch: {trajectory_path}")
        actor = _actor_from_mapping(report.get("actor"))
        if actor.actor_hash != report.get("actor_hash") or actor.policy_type != "parameter":
            raise ValueError(f"probe teacher actor commitment is invalid: {report_path}")
        active = np.asarray(trajectory["strike_coordination_actor_active"], dtype=np.bool_)
        observations = np.asarray(trajectory["strike_coordination_observation"], dtype=np.float64)[
            active
        ]
        actions = np.column_stack(
            (
                np.asarray(trajectory["strike_coordination_stance_blend"], dtype=np.float64)[
                    active
                ],
                np.asarray(trajectory["strike_coordination_goal_yaw_blend"], dtype=np.float64)[
                    active
                ],
            )
        )
        if len(observations) == 0:
            raise ValueError(f"probe has no active coordination observations: {report_path}")
        recomputed = np.asarray(
            [
                tuple(asdict(actor.act(StrikeCoordinationObservation(*row))).values())
                for row in observations
            ],
            dtype=np.float64,
        )
        if actions.shape != recomputed.shape or not np.allclose(
            actions, recomputed, atol=1.0e-10, rtol=0.0
        ):
            raise ValueError(f"probe action trace does not match its actor: {trajectory_path}")
        observation_parts.append(observations)
        action_parts.append(actions)
        source_parts.append(np.full(len(observations), source_index, dtype=np.int64))
        paths.append(str(root))
        report_hashes.append(stored_report_hash)
        trajectory_digests.append(digest)
    all_observations = np.concatenate(observation_parts)
    all_actions = np.concatenate(action_parts)
    all_sources = np.concatenate(source_parts)
    snapshot_hash = _dataset_snapshot_hash(
        observations=all_observations,
        actions=all_actions,
        source_index=all_sources,
        source_report_hashes=tuple(report_hashes),
        source_trajectory_digests=tuple(trajectory_digests),
    )
    return DynamicStrikeCoordinationDataset(
        observations=all_observations,
        actions=all_actions,
        source_index=all_sources,
        source_probe_paths=tuple(paths),
        source_report_hashes=tuple(report_hashes),
        source_trajectory_digests=tuple(trajectory_digests),
        dataset_snapshot_hash=snapshot_hash,
    )


def fit_dynamic_strike_coordination_actor(
    dataset: DynamicStrikeCoordinationDataset,
    *,
    ridge: float = 0.03,
) -> tuple[DynamicStrikeCoordinationActor, dict[str, float]]:
    """Fit a two-head linear sigmoid actor in normalized proprioceptive space."""

    if not math.isfinite(ridge) or not 0.0 < ridge <= 10.0:
        raise ValueError("dynamic strike ridge must be in (0, 10]")
    mean = np.mean(dataset.observations, axis=0)
    std = np.maximum(np.std(dataset.observations, axis=0), _FEATURE_STD_FLOOR)
    features = (dataset.observations - mean) / std
    design = np.column_stack((features, np.ones(len(features), dtype=np.float64)))
    maximum = np.asarray((0.80, 1.0), dtype=np.float64)
    probabilities = np.clip(dataset.actions / maximum, 1.0e-5, 1.0 - 1.0e-5)
    targets = np.log(probabilities / (1.0 - probabilities))
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    weights = np.clip(coefficients[:-1].T, -12.0, 12.0)
    biases = np.clip(coefficients[-1], -12.0, 12.0)
    actor = DynamicStrikeCoordinationActor(
        weights=tuple(float(value) for value in weights.reshape(-1)),
        biases=(float(biases[0]), float(biases[1])),
        feature_mean=tuple(float(value) for value in mean),
        feature_std=tuple(float(value) for value in std),
        policy_type="learned_linear",
        dataset_snapshot_hash=dataset.dataset_snapshot_hash,
    )
    predicted = np.asarray(
        [
            tuple(asdict(actor.act(StrikeCoordinationObservation(*row))).values())
            for row in dataset.observations
        ],
        dtype=np.float64,
    )
    residual = predicted - dataset.actions
    metrics = {
        "stance_blend_rmse": float(np.sqrt(np.mean(residual[:, 0] ** 2))),
        "goal_yaw_blend_rmse": float(np.sqrt(np.mean(residual[:, 1] ** 2))),
        "maximum_absolute_error": float(np.max(np.abs(residual))),
    }
    return actor, metrics


def train_dynamic_strike_coordination_actor(
    *,
    probe_directories: tuple[Path, ...],
    output_dir: Path,
    ridge: float = 0.03,
) -> dict[str, Any]:
    """Persist the verified dataset and learned actor outside the checkout."""

    root = output_dir.expanduser().resolve()
    checkout = Path(__file__).parents[3]
    if root == checkout or checkout in root.parents:
        raise ValueError("dynamic strike learning evidence must remain outside the checkout")
    if root.exists():
        raise FileExistsError("dynamic strike learning output already exists")
    dataset = load_dynamic_strike_coordination_dataset(probe_directories)
    actor, fit_metrics = fit_dynamic_strike_coordination_actor(dataset, ridge=ridge)
    root.mkdir(parents=True)
    dataset_path = root / "dataset.npz"
    _atomic_npz(
        dataset_path,
        {
            "observations": dataset.observations,
            "actions": dataset.actions,
            "source_index": dataset.source_index,
        },
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.dynamic_strike_coordination_learning.v1",
        "feature_names": list(STRIKE_COORDINATION_FEATURES),
        "sample_count": len(dataset.observations),
        "source_probe_paths": list(dataset.source_probe_paths),
        "source_report_hashes": list(dataset.source_report_hashes),
        "source_trajectory_digests": list(dataset.source_trajectory_digests),
        "dataset_snapshot_hash": dataset.dataset_snapshot_hash,
        "dataset_artifact": {
            "file": dataset_path.name,
            "file_hash": hash_bytes(dataset_path.read_bytes()),
        },
        "ridge": ridge,
        "fit_metrics": fit_metrics,
        "actor": asdict(actor),
        "actor_hash": actor.actor_hash,
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(root / "learning.json", report)
    return report


def load_dynamic_strike_coordination_actor_artifact(
    artifact_directory: Path,
) -> tuple[DynamicStrikeCoordinationActor, dict[str, Any]]:
    """Integrity-check a learned artifact before any simulation evaluation."""

    root = artifact_directory.expanduser().resolve()
    report_path = root / "learning.json"
    report = _read_json(report_path)
    stored_report_hash = str(report.get("report_hash", ""))
    hash_payload = dict(report)
    hash_payload.pop("report_hash", None)
    artifact = report.get("dataset_artifact")
    if (
        stored_report_hash != hash_json(hash_payload)
        or report.get("schema_version") != "rosclaw_soccer.dynamic_strike_coordination_learning.v1"
        or report.get("activation_ceiling") != "SIM_ONLY"
        or report.get("hardware_command_sent") is not False
        or not isinstance(artifact, dict)
        or artifact.get("file") != "dataset.npz"
    ):
        raise ValueError(f"dynamic strike learning report is invalid: {report_path}")
    dataset_path = root / "dataset.npz"
    if hash_bytes(dataset_path.read_bytes()) != artifact.get("file_hash"):
        raise ValueError(f"dynamic strike dataset file hash mismatch: {dataset_path}")
    with np.load(dataset_path, allow_pickle=False) as archive:
        observations = np.asarray(archive["observations"], dtype=np.float64)
        actions = np.asarray(archive["actions"], dtype=np.float64)
        source_index = np.asarray(archive["source_index"], dtype=np.int64)
    report_hashes = tuple(str(value) for value in report.get("source_report_hashes", ()))
    trajectory_digests = tuple(str(value) for value in report.get("source_trajectory_digests", ()))
    snapshot_hash = _dataset_snapshot_hash(
        observations=observations,
        actions=actions,
        source_index=source_index,
        source_report_hashes=report_hashes,
        source_trajectory_digests=trajectory_digests,
    )
    actor = _actor_from_mapping(report.get("actor"))
    if (
        snapshot_hash != report.get("dataset_snapshot_hash")
        or actor.policy_type != "learned_linear"
        or actor.dataset_snapshot_hash != snapshot_hash
        or actor.actor_hash != report.get("actor_hash")
    ):
        raise ValueError(f"dynamic strike learned actor commitment is invalid: {report_path}")
    return actor, report


def _dataset_snapshot_hash(
    *,
    observations: NDArray[np.float64],
    actions: NDArray[np.float64],
    source_index: NDArray[np.int64],
    source_report_hashes: tuple[str, ...],
    source_trajectory_digests: tuple[str, ...],
) -> str:
    return str(
        hash_json(
            {
                "schema_version": "rosclaw_soccer.dynamic_strike_coordination_dataset.v1",
                "feature_names": list(STRIKE_COORDINATION_FEATURES),
                "observations_hash": hash_bytes(
                    np.ascontiguousarray(observations, dtype="<f8").tobytes()
                ),
                "actions_hash": hash_bytes(np.ascontiguousarray(actions, dtype="<f8").tobytes()),
                "source_index_hash": hash_bytes(
                    np.ascontiguousarray(source_index, dtype="<i8").tobytes()
                ),
                "source_report_hashes": list(source_report_hashes),
                "source_trajectory_digests": list(source_trajectory_digests),
            }
        )
    )


def _actor_from_mapping(raw: object) -> DynamicStrikeCoordinationActor:
    if not isinstance(raw, Mapping):
        raise ValueError("probe actor payload is invalid")
    values = dict(raw)
    for name in ("weights", "biases", "feature_mean", "feature_std"):
        value = values.get(name)
        if not isinstance(value, list):
            raise ValueError("probe actor vectors must be JSON arrays")
        values[name] = tuple(float(item) for item in value)
    return DynamicStrikeCoordinationActor(**values)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _atomic_npz(path: Path, arrays: dict[str, NDArray[Any]]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(output, **arrays)  # type: ignore[arg-type]
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-probe", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ridge", default=0.03, type=float)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = train_dynamic_strike_coordination_actor(
        probe_directories=tuple(arguments.teacher_probe),
        output_dir=arguments.output_dir,
        ridge=arguments.ridge,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DynamicStrikeCoordinationDataset",
    "fit_dynamic_strike_coordination_actor",
    "load_dynamic_strike_coordination_actor_artifact",
    "load_dynamic_strike_coordination_dataset",
    "train_dynamic_strike_coordination_actor",
]
