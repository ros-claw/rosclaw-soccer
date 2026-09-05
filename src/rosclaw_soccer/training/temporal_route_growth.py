"""S123 temporal route acquisition from released S122 physics traces."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.reactive_route_actor import load_reactive_route_actor
from rosclaw_soccer.growth.tactical_2v1 import TacticalAction
from rosclaw_soccer.growth.tactical_2v1_actor import load_two_vs_one_tactical_actor
from rosclaw_soccer.growth.temporal_route_actor import (
    G1TemporalRouteActor,
    TemporalRouteSequence,
    fit_temporal_route_actor,
    load_temporal_route_actor,
    save_temporal_route_actor,
)
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.full_body_tactical_2v1 import (
    FullBodyTwoVsOneConfig,
    FullBodyTwoVsOneScenario,
)
from rosclaw_soccer.training.reactive_route_growth import (
    ReactiveRouteCase,
    simulate_reactive_route_episode,
)
from rosclaw_soccer.training.tactical_2v1_physics import FrozenTacticalSkillBundle


@dataclass(frozen=True)
class TemporalRouteSourceSnapshot:
    source_stage_hash: str
    source_actor_hash: str
    source_report_hash: str
    source_actor_file_hash: str
    sequences: tuple[TemporalRouteSequence, ...]
    trajectory_hashes: tuple[str, ...]
    schema_version: str = "rosclaw_soccer.temporal_route_source_snapshot.v1"

    @property
    def snapshot_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class TemporalRouteRetentionManifest:
    cases: tuple[ReactiveRouteCase, ...]
    suite_id: str = "s123.temporal-route.sealed-retention"
    training_access_allowed: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.temporal_route_retention_manifest.v1"

    def __post_init__(self) -> None:
        hashes = tuple(case.case_hash for case in self.cases)
        if (
            len(self.cases) != 8
            or len(set(hashes)) != len(hashes)
            or self.training_access_allowed
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("temporal route retention must be unique, sealed and SIM_ONLY")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "cases": [asdict(case) for case in self.cases],
            "case_hashes": [case.case_hash for case in self.cases],
            "training_access_allowed": self.training_access_allowed,
            "activation_ceiling": self.activation_ceiling,
        }
        if include_hash:
            value["manifest_hash"] = hash_json(value)
        return value

    @property
    def manifest_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))


def load_released_s122_temporal_snapshot(source_stage_dir: Path) -> TemporalRouteSourceSnapshot:
    """Verify released S122 artifacts before treating their frames as training data."""

    root = source_stage_dir.expanduser().resolve()
    stage = _load_json(root / "stage-summary.json")
    _verify_hash(stage, "stage_hash", "S122 stage")
    if stage.get("status") != "PASS_REACTIVE_MULTI_AGENT_GROWTH":
        raise ValueError("S122 temporal source is not a passing released stage")
    report = _load_json(root / "retention-v2/retention-exam.json")
    _verify_hash(report, "report_hash", "S122 retention")
    if report.get("status") != "PASS_REACTIVE_MULTI_AGENT_GROWTH" or report.get(
        "report_hash"
    ) != stage.get("retention_report_hash"):
        raise ValueError("S122 temporal source retention binding is invalid")
    actor_path = root / "reactive-route-champion-v2.json"
    actor = load_reactive_route_actor(actor_path)
    actor_file_hash = hash_bytes(actor_path.read_bytes())
    if actor.actor_hash != stage.get("actor_hash") or actor_file_hash != stage.get(
        "actor_file_hash"
    ):
        raise ValueError("S122 temporal source actor binding is invalid")

    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("S122 temporal source requires eight released trajectories")
    sequences: list[TemporalRouteSequence] = []
    trajectory_hashes: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not (
            row.get("qualified") is True
            and row.get("safe") is True
            and row.get("exact_replay") is True
        ):
            raise ValueError("S122 temporal source contains an unqualified trajectory")
        artifact = row.get("primary_artifact")
        if not isinstance(artifact, dict):
            raise ValueError("S122 temporal source artifact is absent")
        path = root / "retention-v2" / f"case-{index:03d}" / str(artifact.get("file"))
        file_hash = hash_bytes(path.read_bytes())
        if file_hash != artifact.get("file_hash"):
            raise ValueError("S122 temporal source file hash changed")
        with np.load(path, allow_pickle=False) as archive:
            trajectory = {name: np.asarray(archive[name]) for name in archive.files}
        if trajectory_digest(trajectory) != artifact.get("trajectory_digest"):
            raise ValueError("S122 temporal source trajectory digest changed")
        trajectory_hashes.append(file_hash)
        for role in ("passer", "goalkeeper"):
            features = np.asarray(trajectory[f"{role}_reactive_route_features"], dtype=np.float64)
            commands = np.asarray(trajectory[f"{role}_tactical_world_command"], dtype=np.float64)[
                :, :2
            ]
            sequences.append(
                TemporalRouteSequence(
                    sequence_id=f"{row['case_id']}.{role}",
                    features=tuple(tuple(float(value) for value in frame) for frame in features),
                    teacher_world_commands_xy_mps=tuple(
                        (float(command[0]), float(command[1])) for command in commands
                    ),
                )
            )
    return TemporalRouteSourceSnapshot(
        source_stage_hash=str(stage["stage_hash"]),
        source_actor_hash=actor.actor_hash,
        source_report_hash=str(report["report_hash"]),
        source_actor_file_hash=actor_file_hash,
        sequences=tuple(sequences),
        trajectory_hashes=tuple(trajectory_hashes),
    )


def train_temporal_route_population(
    *,
    source_stage_dir: Path,
    output_dir: Path,
    devices: tuple[str, ...] = ("cpu",),
    seeds: tuple[int, ...] = (1230, 1231, 1232, 1233),
    epochs: int = 800,
    residual_ceiling_mps: float = 0.04,
    maximum_parent_attenuation_fraction: float = 0.15,
    maximum_residual_to_parent_fraction: float = 0.20,
    development_sequences: tuple[TemporalRouteSequence, ...] = (),
) -> tuple[G1TemporalRouteActor, dict[str, Any]]:
    """Train independent seeds, select on released validation traces, then consolidate."""

    if len(devices) not in {1, len(seeds)} or len(set(seeds)) != len(seeds):
        raise ValueError("temporal route population device and seed mapping is invalid")
    if any(
        not sequence.sequence_id.startswith("s123.development.")
        for sequence in development_sequences
    ) or len({item.sequence_id for item in development_sequences}) != len(development_sequences):
        raise ValueError("temporal route development sequences violate isolation naming")
    destination = output_dir.expanduser().resolve()
    if destination.exists():
        raise ValueError("temporal route population output must be new")
    destination.mkdir(parents=True)
    snapshot = load_released_s122_temporal_snapshot(source_stage_dir)
    parent = load_reactive_route_actor(
        source_stage_dir.expanduser().resolve() / "reactive-route-champion-v2.json"
    )
    _write_json(destination / "source-snapshot.json", _snapshot_summary(snapshot))
    source_training, validation = _sequence_split(snapshot.sequences)
    training = (*source_training, *development_sequences)
    candidates: list[tuple[G1TemporalRouteActor, dict[str, Any]]] = []
    for index, seed in enumerate(seeds):
        device = devices[0] if len(devices) == 1 else devices[index]
        actor = fit_temporal_route_actor(
            training,
            source_stage_hash=snapshot.source_stage_hash,
            source_actor_hash=snapshot.source_actor_hash,
            parent_actor=parent,
            seed=seed,
            epochs=epochs,
            residual_ceiling_mps=residual_ceiling_mps,
            maximum_parent_attenuation_fraction=maximum_parent_attenuation_fraction,
            maximum_residual_to_parent_fraction=maximum_residual_to_parent_fraction,
            device=device,
        )
        metrics = _sequence_metrics(actor, validation)
        candidate_path = destination / f"candidate-{index:02d}.json"
        save_temporal_route_actor(actor, candidate_path)
        candidates.append(
            (
                actor,
                {
                    "candidate_index": index,
                    "seed": seed,
                    "device": device,
                    "actor_hash": actor.actor_hash,
                    "actor_file": candidate_path.name,
                    "actor_file_hash": hash_bytes(candidate_path.read_bytes()),
                    "training_rmse_mps": actor.training_rmse_mps,
                    "validation": metrics,
                },
            )
        )
    candidates.sort(
        key=lambda item: (
            float(item[1]["validation"]["rmse_mps"]),
            float(item[1]["validation"]["command_acceleration_rms_mps_per_step"]),
            int(item[1]["seed"]),
        )
    )
    selected_seed = int(candidates[0][1]["seed"])
    champion = fit_temporal_route_actor(
        (*snapshot.sequences, *development_sequences),
        source_stage_hash=snapshot.source_stage_hash,
        source_actor_hash=snapshot.source_actor_hash,
        parent_actor=parent,
        seed=selected_seed,
        epochs=epochs,
        residual_ceiling_mps=residual_ceiling_mps,
        maximum_parent_attenuation_fraction=maximum_parent_attenuation_fraction,
        maximum_residual_to_parent_fraction=maximum_residual_to_parent_fraction,
        device="cpu",
    )
    champion_path = destination / "temporal-route-champion.json"
    save_temporal_route_actor(champion, champion_path)
    champion_metrics = _sequence_metrics(champion, snapshot.sequences)
    parent_metrics = _memoryless_sequence_metrics(parent, snapshot.sequences)
    development_metrics = (
        None if not development_sequences else _sequence_metrics(champion, development_sequences)
    )
    gates = {
        "four_unique_candidates": len(candidates) == 4
        and len({item[0].actor_hash for item in candidates}) == 4,
        "validation_rmse_below_0p08_mps": float(candidates[0][1]["validation"]["rmse_mps"]) <= 0.08,
        "consolidated_rmse_within_0p005_of_parent": champion_metrics["rmse_mps"]
        <= parent_metrics["rmse_mps"] + 0.005,
        "command_acceleration_reduced_10_percent": champion_metrics[
            "command_acceleration_rms_mps_per_step"
        ]
        <= 0.90 * parent_metrics["command_acceleration_rms_mps_per_step"],
        "released_s122_and_isolated_development_only": all(
            item.sequence_id.startswith("s123.development.") for item in development_sequences
        ),
        "no_current_retention_access": True,
        "numpy_runtime_artifact": True,
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.temporal_route_population.v1",
        "status": (
            "PASS_TEMPORAL_ROUTE_ACQUISITION"
            if all(gates.values())
            else "REJECTED_TEMPORAL_ROUTE_ACQUISITION"
        ),
        "source_snapshot_hash": snapshot.snapshot_hash,
        "source_stage_hash": snapshot.source_stage_hash,
        "source_actor_hash": snapshot.source_actor_hash,
        "training_sequence_ids": [item.sequence_id for item in training],
        "validation_sequence_ids": [item.sequence_id for item in validation],
        "development_sequence_ids": [item.sequence_id for item in development_sequences],
        "candidate_rows": [item[1] for item in candidates],
        "selected_seed": selected_seed,
        "champion_hash": champion.actor_hash,
        "champion_file_hash": hash_bytes(champion_path.read_bytes()),
        "champion_metrics": champion_metrics,
        "parent_metrics": parent_metrics,
        "development_metrics": development_metrics,
        "gates": gates,
        "training_devices": list(devices),
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(destination / "population-report.json", report)
    return champion, report


def build_temporal_failure_sequences(
    trajectory: dict[str, NDArray[Any]],
    *,
    episode_id: str,
    teammate_lateral_bias_m: float,
) -> tuple[TemporalRouteSequence, ...]:
    """Relabel an on-policy development trace while preserving temporal order."""

    if not episode_id.startswith("s123.development."):
        raise ValueError("temporal failure sequence must be a development episode")
    if not -0.12 <= teammate_lateral_bias_m <= 0.12:
        raise ValueError("temporal failure relabel bias must remain in [-0.12, 0.12] m")
    sequences = []
    for role, maximum_speed, gain, damping in (
        ("passer", 0.45, 1.35, 0.12),
        ("goalkeeper", 0.38, 1.05, 0.15),
    ):
        features = np.asarray(trajectory[f"{role}_reactive_route_features"], dtype=np.float64)
        error = features[:, :2].copy()
        if role == "passer":
            error[:, 1] += teammate_lateral_bias_m
        commands = gain * error - damping * features[:, 2:4]
        speed = np.linalg.norm(commands, axis=1)
        commands *= np.minimum(1.0, maximum_speed / np.maximum(speed, 1.0e-9))[:, None]
        sequences.append(
            TemporalRouteSequence(
                sequence_id=f"{episode_id}.{role}",
                features=tuple(tuple(float(value) for value in row) for row in features),
                teacher_world_commands_xy_mps=tuple(
                    (float(command[0]), float(command[1])) for command in commands
                ),
            )
        )
    return tuple(sequences)


def default_temporal_route_retention_manifest() -> TemporalRouteRetentionManifest:
    """Declare S123 cases disjoint from all S122 acquisition and retention layouts."""

    layouts = (
        ((5.42, -0.38, 0.0), (1.98, 0.38, 0.0), (0.15, -0.12), (-0.13, 0.10)),
        ((5.52, -0.43, 0.0), (2.12, 0.52, 0.0), (-0.15, 0.11), (0.13, -0.12)),
        ((5.61, -0.46, 0.0), (2.18, 0.58, 0.0), (0.16, 0.09), (-0.14, -0.11)),
        ((5.70, -0.50, 0.0), (2.00, 0.40, 0.0), (-0.14, -0.15), (0.12, 0.14)),
        ((5.32, 0.32, 0.0), (4.10, 0.31, 0.0), (0.14, -0.13), (-0.12, 0.11)),
        ((5.43, 0.42, 0.0), (4.30, 0.41, 0.0), (-0.15, 0.12), (0.14, -0.13)),
        ((5.54, 0.52, 0.0), (4.50, 0.51, 0.0), (0.16, 0.09), (-0.15, -0.11)),
        ((5.65, 0.62, 0.0), (4.70, 0.61, 0.0), (-0.14, -0.15), (0.12, 0.16)),
    )
    cases = tuple(
        ReactiveRouteCase(
            scenario=FullBodyTwoVsOneScenario(
                scenario_id=f"s123.retention.{index:03d}",
                seed=123_700 + index,
                teammate_origin_m=teammate,
                defender_origin_m=defender,
            ),
            teammate_origin_offset_m=teammate_offset,
            defender_origin_offset_m=defender_offset,
        )
        for index, (teammate, defender, teammate_offset, defender_offset) in enumerate(layouts)
    )
    return TemporalRouteRetentionManifest(cases=cases)


def default_temporal_route_retention_manifest_v2() -> TemporalRouteRetentionManifest:
    """Keep the frozen tactical state in support while hardening body offsets."""

    layouts = (
        ((5.47, -0.40, 0.0), (2.04, 0.44, 0.0), (0.15, -0.12), (-0.13, 0.10)),
        ((5.51, -0.40, 0.0), (2.10, 0.50, 0.0), (-0.15, 0.11), (0.13, -0.12)),
        ((5.56, -0.40, 0.0), (2.15, 0.55, 0.0), (0.16, 0.09), (-0.14, -0.11)),
        ((5.60, -0.40, 0.0), (2.03, 0.43, 0.0), (-0.14, -0.15), (0.12, 0.14)),
        ((5.38, 0.38, 0.0), (4.18, 0.38, 0.0), (0.14, -0.13), (-0.12, 0.11)),
        ((5.48, 0.48, 0.0), (4.38, 0.48, 0.0), (-0.15, 0.12), (0.14, -0.13)),
        ((5.58, 0.58, 0.0), (4.58, 0.58, 0.0), (0.16, 0.09), (-0.15, -0.11)),
        ((5.68, 0.58, 0.0), (4.78, 0.48, 0.0), (-0.14, -0.15), (0.12, 0.16)),
    )
    cases = tuple(
        ReactiveRouteCase(
            scenario=FullBodyTwoVsOneScenario(
                scenario_id=f"s123.retention-v2.{index:03d}",
                seed=123_900 + index,
                teammate_origin_m=teammate,
                defender_origin_m=defender,
            ),
            teammate_origin_offset_m=teammate_offset,
            defender_origin_offset_m=defender_offset,
        )
        for index, (teammate, defender, teammate_offset, defender_offset) in enumerate(layouts)
    )
    return TemporalRouteRetentionManifest(
        cases=cases,
        suite_id="s123.temporal-route.sealed-retention-v2",
    )


def default_temporal_route_retention_manifest_v3() -> TemporalRouteRetentionManifest:
    """Fresh suite for the mirrored weak-side execution generation."""

    layouts = (
        ((5.47, -0.40, 0.0), (2.04, 0.44, 0.0), (0.15, -0.14), (-0.14, 0.12)),
        ((5.51, -0.40, 0.0), (2.10, 0.50, 0.0), (-0.16, 0.10), (0.14, -0.13)),
        ((5.56, -0.40, 0.0), (2.15, 0.55, 0.0), (0.16, 0.11), (-0.15, -0.10)),
        ((5.60, -0.40, 0.0), (2.03, 0.43, 0.0), (-0.16, -0.14), (0.13, 0.15)),
        ((5.38, 0.38, 0.0), (4.18, 0.38, 0.0), (0.15, -0.14), (-0.13, 0.12)),
        ((5.48, 0.48, 0.0), (4.38, 0.48, 0.0), (-0.16, 0.11), (0.15, -0.14)),
        ((5.58, 0.58, 0.0), (4.58, 0.58, 0.0), (0.16, 0.10), (-0.16, -0.10)),
        ((5.68, 0.58, 0.0), (4.78, 0.48, 0.0), (-0.15, -0.14), (0.13, 0.16)),
    )
    cases = tuple(
        ReactiveRouteCase(
            scenario=FullBodyTwoVsOneScenario(
                scenario_id=f"s123.retention-v3.{index:03d}",
                seed=123_950 + index,
                teammate_origin_m=teammate,
                defender_origin_m=defender,
            ),
            teammate_origin_offset_m=teammate_offset,
            defender_origin_offset_m=defender_offset,
        )
        for index, (teammate, defender, teammate_offset, defender_offset) in enumerate(layouts)
    )
    return TemporalRouteRetentionManifest(
        cases=cases,
        suite_id="s123.temporal-route.sealed-retention-v3",
    )


def default_temporal_route_retention_manifest_v4() -> TemporalRouteRetentionManifest:
    """Fresh suite for follow-through and collision-shield qualification."""

    layouts = (
        ((5.47, -0.40, 0.0), (2.04, 0.44, 0.0), (0.14, -0.16), (-0.15, 0.11)),
        ((5.51, -0.40, 0.0), (2.10, 0.50, 0.0), (-0.17, 0.09), (0.15, -0.12)),
        ((5.56, -0.40, 0.0), (2.15, 0.55, 0.0), (0.17, 0.08), (-0.16, -0.09)),
        ((5.60, -0.40, 0.0), (2.03, 0.43, 0.0), (-0.15, -0.16), (0.14, 0.13)),
        ((5.38, 0.38, 0.0), (4.18, 0.38, 0.0), (0.14, -0.16), (-0.15, 0.11)),
        ((5.48, 0.48, 0.0), (4.38, 0.48, 0.0), (-0.17, 0.09), (0.16, -0.13)),
        ((5.58, 0.58, 0.0), (4.58, 0.58, 0.0), (0.17, 0.08), (-0.17, -0.09)),
        ((5.68, 0.58, 0.0), (4.78, 0.48, 0.0), (-0.16, -0.13), (0.14, 0.15)),
    )
    cases = tuple(
        ReactiveRouteCase(
            scenario=FullBodyTwoVsOneScenario(
                scenario_id=f"s123.retention-v4.{index:03d}",
                seed=123_980 + index,
                teammate_origin_m=teammate,
                defender_origin_m=defender,
            ),
            teammate_origin_offset_m=teammate_offset,
            defender_origin_offset_m=defender_offset,
        )
        for index, (teammate, defender, teammate_offset, defender_offset) in enumerate(layouts)
    )
    return TemporalRouteRetentionManifest(
        cases=cases,
        suite_id="s123.temporal-route.sealed-retention-v4",
    )


def default_temporal_route_retention_manifest_v5() -> TemporalRouteRetentionManifest:
    """Fresh suite for the conservative parent-anchored temporal actor."""

    layouts = (
        ((5.47, -0.40, 0.0), (2.04, 0.44, 0.0), (0.13, -0.17), (-0.16, 0.10)),
        ((5.51, -0.40, 0.0), (2.10, 0.50, 0.0), (-0.16, 0.12), (0.16, -0.11)),
        ((5.56, -0.40, 0.0), (2.15, 0.55, 0.0), (0.17, 0.07), (-0.17, -0.08)),
        ((5.60, -0.40, 0.0), (2.03, 0.43, 0.0), (-0.17, -0.13), (0.15, 0.12)),
        ((5.38, 0.38, 0.0), (4.18, 0.38, 0.0), (0.13, -0.17), (-0.16, 0.10)),
        ((5.48, 0.48, 0.0), (4.38, 0.48, 0.0), (-0.16, 0.12), (0.17, -0.12)),
        ((5.58, 0.58, 0.0), (4.58, 0.58, 0.0), (0.17, 0.07), (-0.17, -0.08)),
        ((5.68, 0.58, 0.0), (4.78, 0.48, 0.0), (-0.17, -0.12), (0.15, 0.14)),
    )
    cases = tuple(
        ReactiveRouteCase(
            scenario=FullBodyTwoVsOneScenario(
                scenario_id=f"s123.retention-v5.{index:03d}",
                seed=124_000 + index,
                teammate_origin_m=teammate,
                defender_origin_m=defender,
            ),
            teammate_origin_offset_m=teammate_offset,
            defender_origin_offset_m=defender_offset,
        )
        for index, (teammate, defender, teammate_offset, defender_offset) in enumerate(layouts)
    )
    return TemporalRouteRetentionManifest(
        cases=cases,
        suite_id="s123.temporal-route.sealed-retention-v5",
    )


def default_temporal_route_retention_manifest_v6() -> TemporalRouteRetentionManifest:
    """Fresh suite after freezing the predictive pass-reception reflex."""

    layouts = (
        ((5.47, -0.40, 0.0), (2.04, 0.44, 0.0), (0.12, -0.18), (-0.17, 0.09)),
        ((5.51, -0.40, 0.0), (2.10, 0.50, 0.0), (-0.18, 0.13), (0.17, -0.10)),
        ((5.56, -0.40, 0.0), (2.15, 0.55, 0.0), (0.18, 0.06), (-0.18, -0.07)),
        ((5.60, -0.40, 0.0), (2.03, 0.43, 0.0), (-0.18, -0.11), (0.16, 0.11)),
        ((5.38, 0.38, 0.0), (4.18, 0.38, 0.0), (0.12, -0.18), (-0.17, 0.09)),
        ((5.48, 0.48, 0.0), (4.38, 0.48, 0.0), (-0.18, 0.13), (0.18, -0.11)),
        ((5.58, 0.58, 0.0), (4.58, 0.58, 0.0), (0.18, 0.06), (-0.18, -0.07)),
        ((5.68, 0.58, 0.0), (4.78, 0.48, 0.0), (-0.18, -0.11), (0.16, 0.13)),
    )
    cases = tuple(
        ReactiveRouteCase(
            scenario=FullBodyTwoVsOneScenario(
                scenario_id=f"s123.retention-v6.{index:03d}",
                seed=124_020 + index,
                teammate_origin_m=teammate,
                defender_origin_m=defender,
            ),
            teammate_origin_offset_m=teammate_offset,
            defender_origin_offset_m=defender_offset,
        )
        for index, (teammate, defender, teammate_offset, defender_offset) in enumerate(layouts)
    )
    return TemporalRouteRetentionManifest(
        cases=cases,
        suite_id="s123.temporal-route.sealed-retention-v6",
    )


def default_temporal_route_retention_manifest_v7() -> TemporalRouteRetentionManifest:
    """Fresh suite after freezing diagonal braking and the all-action joint guard."""

    layouts = (
        ((5.47, -0.40, 0.0), (2.04, 0.44, 0.0), (0.09, -0.18), (-0.18, 0.07)),
        ((5.51, -0.40, 0.0), (2.10, 0.50, 0.0), (-0.18, 0.14), (0.18, -0.08)),
        ((5.56, -0.40, 0.0), (2.15, 0.55, 0.0), (0.18, 0.03), (-0.18, -0.05)),
        ((5.60, -0.40, 0.0), (2.03, 0.43, 0.0), (-0.18, -0.08), (0.17, 0.09)),
        ((5.38, 0.38, 0.0), (4.18, 0.38, 0.0), (0.09, -0.18), (-0.18, 0.07)),
        ((5.48, 0.48, 0.0), (4.38, 0.48, 0.0), (-0.18, 0.14), (0.18, -0.09)),
        ((5.58, 0.58, 0.0), (4.58, 0.58, 0.0), (0.18, 0.03), (-0.18, -0.05)),
        ((5.68, 0.58, 0.0), (4.78, 0.48, 0.0), (-0.18, -0.08), (0.17, 0.11)),
    )
    cases = tuple(
        ReactiveRouteCase(
            scenario=FullBodyTwoVsOneScenario(
                scenario_id=f"s123.retention-v7.{index:03d}",
                seed=124_060 + index,
                teammate_origin_m=teammate,
                defender_origin_m=defender,
            ),
            teammate_origin_offset_m=teammate_offset,
            defender_origin_offset_m=defender_offset,
        )
        for index, (teammate, defender, teammate_offset, defender_offset) in enumerate(layouts)
    )
    return TemporalRouteRetentionManifest(
        cases=cases,
        suite_id="s123.temporal-route.sealed-retention-v7",
    )


def default_temporal_route_retention_manifest_v8() -> TemporalRouteRetentionManifest:
    """Fresh final suite for the continuous-confidence braking generation."""

    layouts = (
        ((5.47, -0.40, 0.0), (2.04, 0.44, 0.0), (0.08, -0.17), (-0.17, 0.06)),
        ((5.51, -0.40, 0.0), (2.10, 0.50, 0.0), (-0.17, 0.16), (0.17, -0.07)),
        ((5.56, -0.40, 0.0), (2.15, 0.55, 0.0), (0.17, 0.02), (-0.17, -0.04)),
        ((5.60, -0.40, 0.0), (2.03, 0.43, 0.0), (-0.17, -0.07), (0.16, 0.08)),
        ((5.38, 0.38, 0.0), (4.18, 0.38, 0.0), (0.08, -0.17), (-0.17, 0.06)),
        ((5.48, 0.48, 0.0), (4.38, 0.48, 0.0), (-0.17, 0.16), (0.17, -0.08)),
        ((5.58, 0.58, 0.0), (4.58, 0.58, 0.0), (0.17, 0.02), (-0.17, -0.04)),
        ((5.68, 0.58, 0.0), (4.78, 0.48, 0.0), (-0.17, -0.07), (0.16, 0.10)),
    )
    cases = tuple(
        ReactiveRouteCase(
            scenario=FullBodyTwoVsOneScenario(
                scenario_id=f"s123.retention-v8.{index:03d}",
                seed=124_100 + index,
                teammate_origin_m=teammate,
                defender_origin_m=defender,
            ),
            teammate_origin_offset_m=teammate_offset,
            defender_origin_offset_m=defender_offset,
        )
        for index, (teammate, defender, teammate_offset, defender_offset) in enumerate(layouts)
    )
    return TemporalRouteRetentionManifest(
        cases=cases,
        suite_id="s123.temporal-route.sealed-retention-v8",
    )


def default_temporal_failure_development_cases() -> tuple[ReactiveRouteCase, ...]:
    """Adjacent cases derived from the failure class, never from sealed trajectories."""

    offsets = (
        ((-0.12, -0.13), (0.10, 0.12)),
        ((-0.13, -0.12), (0.11, 0.10)),
        ((-0.11, -0.15), (0.09, 0.13)),
        ((-0.15, -0.11), (0.12, 0.09)),
    )
    return tuple(
        ReactiveRouteCase(
            scenario=FullBodyTwoVsOneScenario(
                scenario_id=f"s123.development.boundary-{index:02d}",
                seed=123_400 + index,
                teammate_origin_m=(5.60, -0.40, 0.0),
                defender_origin_m=(2.03, 0.43, 0.0),
            ),
            teammate_origin_offset_m=teammate_offset,
            defender_origin_offset_m=defender_offset,
        )
        for index, (teammate_offset, defender_offset) in enumerate(offsets)
    )


def collect_temporal_failure_development(
    *,
    output_dir: Path,
    asset_root: Path,
    temporal_actor_path: Path,
    tactical_actor_path: Path,
    skill_bundle: FrozenTacticalSkillBundle,
    cases: tuple[ReactiveRouteCase, ...] | None = None,
    teammate_lateral_bias_m: float = -0.08,
    config: FullBodyTwoVsOneConfig | None = None,
) -> tuple[tuple[TemporalRouteSequence, ...], dict[str, Any]]:
    """Collect a disjoint on-policy curriculum and persist its bounded labels."""

    destination = output_dir.expanduser().resolve()
    if destination.exists():
        raise ValueError("temporal failure development output must be new")
    destination.mkdir(parents=True)
    actor = load_temporal_route_actor(temporal_actor_path)
    tactical_actor = load_two_vs_one_tactical_actor(tactical_actor_path)
    active = config or FullBodyTwoVsOneConfig()
    selected_cases = cases or default_temporal_failure_development_cases()
    sequences: list[TemporalRouteSequence] = []
    rows = []
    for index, case in enumerate(selected_cases):
        decision = tactical_actor.decide(
            case.scenario.state(skill_bundle=skill_bundle, config=active)
        )
        if not decision.accepted or decision.action != TacticalAction.PASS:
            raise RuntimeError("temporal failure curriculum requires supported PASS cases")
        result, trajectory = simulate_reactive_route_episode(
            asset_root=asset_root,
            case=case,
            action=decision.action,
            actor_path=temporal_actor_path,
            actor=actor,
            skill_bundle=skill_bundle,
            config=active,
        )
        artifact = _save_trajectory(destination / f"development-{index:03d}.npz", trajectory)
        labeled = build_temporal_failure_sequences(
            trajectory,
            episode_id=case.scenario.scenario_id,
            teammate_lateral_bias_m=teammate_lateral_bias_m,
        )
        sequences.extend(labeled)
        rows.append(
            {
                "case": asdict(case),
                "case_hash": case.case_hash,
                "on_policy_result": result.to_dict(),
                "on_policy_qualified": result.qualified,
                "trajectory_artifact": artifact,
                "labeled_sequence_ids": [item.sequence_id for item in labeled],
            }
        )
    sequence_payload = [asdict(sequence) for sequence in sequences]
    sequence_path = destination / "development-sequences.json"
    _write_json(
        sequence_path,
        {
            "schema_version": "rosclaw_soccer.temporal_route_development_sequences.v1",
            "sequences": sequence_payload,
            "sequence_hash": hash_json(sequence_payload),
            "sealed_retention_trajectory_used": False,
            "activation_ceiling": "SIM_ONLY",
        },
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.temporal_route_failure_development.v1",
        "actor_hash": actor.actor_hash,
        "actor_file_hash": hash_bytes(temporal_actor_path.read_bytes()),
        "case_count": len(selected_cases),
        "on_policy_qualified_count": sum(row["on_policy_qualified"] for row in rows),
        "teammate_lateral_bias_m": teammate_lateral_bias_m,
        "rows": rows,
        "development_sequence_file": sequence_path.name,
        "development_sequence_file_hash": hash_bytes(sequence_path.read_bytes()),
        "development_sequence_hash": hash_json(sequence_payload),
        "sealed_retention_trajectory_used": False,
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(destination / "development-report.json", report)
    return tuple(sequences), report


def load_temporal_development_sequences(
    development_dir: Path,
) -> tuple[TemporalRouteSequence, ...]:
    """Load only content-bound, explicitly non-retention development labels."""

    root = development_dir.expanduser().resolve()
    report = _load_json(root / "development-report.json")
    _verify_hash(report, "report_hash", "temporal development report")
    sequence_path = root / str(report.get("development_sequence_file"))
    if hash_bytes(sequence_path.read_bytes()) != report.get("development_sequence_file_hash"):
        raise ValueError("temporal development sequence file hash changed")
    payload = _load_json(sequence_path)
    rows = payload.get("sequences")
    if (
        payload.get("sealed_retention_trajectory_used") is not False
        or not isinstance(rows, list)
        or hash_json(rows) != payload.get("sequence_hash")
        or payload.get("sequence_hash") != report.get("development_sequence_hash")
    ):
        raise ValueError("temporal development sequence boundary is invalid")
    sequences = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("temporal development sequence row is invalid")
        value = dict(row)
        value["features"] = tuple(tuple(frame) for frame in value["features"])
        value["teacher_world_commands_xy_mps"] = tuple(
            tuple(command) for command in value["teacher_world_commands_xy_mps"]
        )
        sequences.append(TemporalRouteSequence(**value))
    return tuple(sequences)


def run_temporal_route_retention_exam(
    *,
    output_dir: Path,
    asset_root: Path,
    temporal_actor_path: Path,
    parent_actor_path: Path,
    tactical_actor_path: Path,
    skill_bundle: FrozenTacticalSkillBundle,
    manifest: TemporalRouteRetentionManifest,
    config: FullBodyTwoVsOneConfig | None = None,
) -> dict[str, Any]:
    """Compare the frozen temporal actor to S122 on unseen CPU MuJoCo cases."""

    destination = output_dir.expanduser().resolve()
    if destination.exists():
        raise ValueError("temporal route retention output must be new")
    destination.mkdir(parents=True)
    temporal_actor = load_temporal_route_actor(temporal_actor_path)
    parent_actor = load_reactive_route_actor(parent_actor_path)
    tactical_actor = load_two_vs_one_tactical_actor(tactical_actor_path)
    active = config or FullBodyTwoVsOneConfig()
    _write_json(destination / "sealed-retention.json", manifest.to_dict())
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(manifest.cases):
        decision = tactical_actor.decide(
            case.scenario.state(skill_bundle=skill_bundle, config=active)
        )
        if not decision.accepted or decision.action not in {
            TacticalAction.PASS,
            TacticalAction.SHOOT,
        }:
            raise RuntimeError("frozen tactical actor rejected a temporal retention case")
        temporal_result, temporal_trace = simulate_reactive_route_episode(
            asset_root=asset_root,
            case=case,
            action=decision.action,
            actor_path=temporal_actor_path,
            actor=temporal_actor,
            skill_bundle=skill_bundle,
            config=active,
        )
        replay_result, replay_trace = simulate_reactive_route_episode(
            asset_root=asset_root,
            case=case,
            action=decision.action,
            actor_path=temporal_actor_path,
            actor=temporal_actor,
            skill_bundle=skill_bundle,
            config=active,
        )
        parent_result, parent_trace = simulate_reactive_route_episode(
            asset_root=asset_root,
            case=case,
            action=decision.action,
            actor_path=parent_actor_path,
            actor=parent_actor,
            skill_bundle=skill_bundle,
            config=active,
        )
        case_dir = destination / f"case-{index:03d}"
        case_dir.mkdir(parents=True)
        temporal_artifact = _save_trajectory(case_dir / "temporal-primary.npz", temporal_trace)
        replay_artifact = _save_trajectory(case_dir / "temporal-replay.npz", replay_trace)
        parent_artifact = _save_trajectory(case_dir / "parent.npz", parent_trace)
        temporal_command_delta = _route_command_delta_rms(temporal_trace)
        parent_command_delta = _route_command_delta_rms(parent_trace)
        rows.append(
            {
                "case_id": case.scenario.scenario_id,
                "case_hash": case.case_hash,
                "action": decision.action.value,
                "temporal": temporal_result.to_dict(),
                "parent": parent_result.to_dict(),
                "temporal_qualified": temporal_result.qualified,
                "parent_qualified": parent_result.qualified,
                "temporal_safe": temporal_result.base.safe,
                "parent_safe": parent_result.base.safe,
                "exact_replay": bool(
                    temporal_result.to_dict() == replay_result.to_dict()
                    and temporal_artifact["trajectory_digest"]
                    == replay_artifact["trajectory_digest"]
                ),
                "temporal_command_delta_rms_mps_per_step": temporal_command_delta,
                "parent_command_delta_rms_mps_per_step": parent_command_delta,
                "temporal_collision_shield_frames": int(
                    np.count_nonzero(temporal_trace["passer_reactive_collision_shield_active"])
                    + np.count_nonzero(
                        temporal_trace["goalkeeper_reactive_collision_shield_active"]
                    )
                ),
                "parent_collision_shield_frames": int(
                    np.count_nonzero(parent_trace["passer_reactive_collision_shield_active"])
                    + np.count_nonzero(parent_trace["goalkeeper_reactive_collision_shield_active"])
                ),
                "temporal_reception_reflex_frames": int(
                    np.count_nonzero(temporal_trace["passer_reception_interception_active"])
                ),
                "parent_reception_reflex_frames": int(
                    np.count_nonzero(parent_trace["passer_reception_interception_active"])
                ),
                "temporal_reception_reflex_peak_torque_nm": float(
                    np.max(np.abs(temporal_trace["passer_reception_interception_torque"]))
                ),
                "parent_reception_reflex_peak_torque_nm": float(
                    np.max(np.abs(parent_trace["passer_reception_interception_torque"]))
                ),
                "temporal_velocity_braking_frames": int(
                    np.count_nonzero(
                        np.linalg.norm(
                            temporal_trace["passer_reactive_velocity_braking_correction"],
                            axis=1,
                        )
                        > 1.0e-9
                    )
                ),
                "temporal_velocity_braking_peak_correction_mps": float(
                    np.max(
                        np.linalg.norm(
                            temporal_trace["passer_reactive_velocity_braking_correction"],
                            axis=1,
                        )
                    )
                ),
                "temporal_minimum_role_separation_m": float(
                    min(
                        np.min(temporal_trace["passer_reactive_role_separation_m"]),
                        np.min(temporal_trace["goalkeeper_reactive_role_separation_m"]),
                    )
                ),
                "parent_minimum_role_separation_m": float(
                    min(
                        np.min(parent_trace["passer_reactive_role_separation_m"]),
                        np.min(parent_trace["goalkeeper_reactive_role_separation_m"]),
                    )
                ),
                "temporal_artifact": temporal_artifact,
                "replay_artifact": replay_artifact,
                "parent_artifact": parent_artifact,
            }
        )
    count = len(rows)
    metrics = {
        "case_count": count,
        "temporal_qualified_rate": sum(row["temporal_qualified"] for row in rows) / count,
        "parent_qualified_rate": sum(row["parent_qualified"] for row in rows) / count,
        "temporal_safe_rate": sum(row["temporal_safe"] for row in rows) / count,
        "parent_safe_rate": sum(row["parent_safe"] for row in rows) / count,
        "exact_replay_rate": sum(row["exact_replay"] for row in rows) / count,
        "selected_action_counts": {
            action: sum(row["action"] == action for row in rows) for action in ("pass", "shoot")
        },
        "temporal_mean_command_delta_rms_mps_per_step": float(
            np.mean([row["temporal_command_delta_rms_mps_per_step"] for row in rows])
        ),
        "parent_mean_command_delta_rms_mps_per_step": float(
            np.mean([row["parent_command_delta_rms_mps_per_step"] for row in rows])
        ),
        "temporal_mean_teammate_root_jerk_mps3": float(
            np.mean(
                [row["temporal"]["teammate_motion"]["root_speed_jerk_rms_mps3"] for row in rows]
            )
        ),
        "parent_mean_teammate_root_jerk_mps3": float(
            np.mean([row["parent"]["teammate_motion"]["root_speed_jerk_rms_mps3"] for row in rows])
        ),
        "maximum_temporal_robot_contact_steps": max(
            row["temporal"]["base"]["robot_robot_contact_count"] for row in rows
        ),
        "collision_shield_case_count": sum(
            row["temporal_collision_shield_frames"] > 0 for row in rows
        ),
        "reception_reflex_case_count": sum(
            row["temporal_reception_reflex_frames"] > 0 for row in rows
        ),
        "maximum_temporal_reception_reflex_torque_nm": max(
            row["temporal_reception_reflex_peak_torque_nm"] for row in rows
        ),
        "diagonal_velocity_braking_case_count": sum(
            row["temporal_velocity_braking_frames"] > 0 for row in rows
        ),
        "maximum_temporal_velocity_braking_correction_mps": max(
            row["temporal_velocity_braking_peak_correction_mps"] for row in rows
        ),
    }
    gates = {
        "all_temporal_cases_qualified": metrics["temporal_qualified_rate"] == 1.0,
        "all_temporal_cases_safe": metrics["temporal_safe_rate"] == 1.0,
        "candidate_dominates_parent": metrics["temporal_qualified_rate"]
        >= metrics["parent_qualified_rate"]
        and metrics["temporal_safe_rate"] >= metrics["parent_safe_rate"],
        "all_exact_replay": metrics["exact_replay_rate"] == 1.0,
        "pass_and_shoot_coverage": metrics["selected_action_counts"] == {"pass": 4, "shoot": 4},
        "temporal_command_not_materially_rougher_than_parent": (
            metrics["temporal_mean_command_delta_rms_mps_per_step"]
            <= 1.03 * metrics["parent_mean_command_delta_rms_mps_per_step"]
        ),
        "collision_contact_bounded": metrics["maximum_temporal_robot_contact_steps"] <= 40,
        "collision_shield_exercised": metrics["collision_shield_case_count"] >= 1,
        "predictive_reception_reflex_exercised": metrics["reception_reflex_case_count"] >= 1,
        "predictive_reception_reflex_torque_bounded": (
            metrics["maximum_temporal_reception_reflex_torque_nm"] <= 20.0 + 1.0e-8
        ),
        "diagonal_velocity_braking_exercised": metrics["diagonal_velocity_braking_case_count"] >= 1,
        "diagonal_velocity_braking_bounded": metrics[
            "maximum_temporal_velocity_braking_correction_mps"
        ]
        <= 0.18 + 1.0e-8,
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.temporal_route_retention_exam.v1",
        "status": (
            "PASS_TEMPORAL_MULTI_AGENT_GROWTH"
            if all(gates.values())
            else "REJECTED_TEMPORAL_MULTI_AGENT_GROWTH"
        ),
        "manifest_hash": manifest.manifest_hash,
        "temporal_actor_hash": temporal_actor.actor_hash,
        "temporal_actor_file_hash": hash_bytes(temporal_actor_path.read_bytes()),
        "parent_actor_hash": parent_actor.actor_hash,
        "parent_actor_file_hash": hash_bytes(parent_actor_path.read_bytes()),
        "metrics": metrics,
        "gates": gates,
        "rows": rows,
        "evidence_boundary": {
            "physics_authority": "CPU_MUJOCO",
            "whole_body_g1_count": 3,
            "shared_solver_and_ball": True,
            "temporal_memory_reset_per_episode": True,
            "movement_executed_by_frozen_neural_locomotion": True,
            "route_actor_pose_joint_torque_or_ball_authority": False,
            "predictive_reception_reflex_safety_projected": True,
            "diagonal_velocity_braking_bounded": True,
            "retention_used_for_training": False,
            "pixels_used_for_scoring": False,
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(destination / "retention-exam.json", report)
    return report


def _sequence_split(
    sequences: tuple[TemporalRouteSequence, ...],
) -> tuple[tuple[TemporalRouteSequence, ...], tuple[TemporalRouteSequence, ...]]:
    validation_case_ids = {
        sequence.sequence_id.rsplit(".", 1)[0]
        for sequence in sequences
        if sequence.sequence_id.endswith((".passer", ".goalkeeper"))
    }
    held_cases = set(sorted(validation_case_ids)[-2:])
    training = tuple(
        sequence
        for sequence in sequences
        if sequence.sequence_id.rsplit(".", 1)[0] not in held_cases
    )
    validation = tuple(
        sequence for sequence in sequences if sequence.sequence_id.rsplit(".", 1)[0] in held_cases
    )
    if len(training) < 8 or len(validation) < 2:
        raise ValueError("temporal route split lost role or scenario coverage")
    return training, validation


def _sequence_metrics(
    actor: G1TemporalRouteActor,
    sequences: tuple[TemporalRouteSequence, ...],
) -> dict[str, float | int]:
    errors: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    accepted = 0
    frames = 0
    for sequence in sequences:
        memory = None
        previous = np.zeros(2, dtype=np.float64)
        for features, target in zip(
            sequence.features, sequence.teacher_world_commands_xy_mps, strict=True
        ):
            decision = actor.decide(np.asarray(features, dtype=np.float64), memory)
            prediction = np.asarray(decision.world_command_xy_mps, dtype=np.float64)
            errors.append(prediction - np.asarray(target, dtype=np.float64))
            deltas.append(prediction - previous)
            accepted += int(decision.accepted)
            frames += 1
            previous = prediction
            memory = decision.next_memory
    return {
        "sequence_count": len(sequences),
        "frame_count": frames,
        "rmse_mps": float(np.sqrt(np.mean(np.square(np.asarray(errors))))),
        "command_acceleration_rms_mps_per_step": float(
            np.sqrt(np.mean(np.square(np.asarray(deltas))))
        ),
        "support_acceptance_rate": accepted / frames,
    }


def _memoryless_sequence_metrics(
    actor: Any,
    sequences: tuple[TemporalRouteSequence, ...],
) -> dict[str, float | int]:
    errors: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    frames = 0
    previous = np.zeros(2, dtype=np.float64)
    for sequence in sequences:
        previous.fill(0.0)
        for features, target in zip(
            sequence.features, sequence.teacher_world_commands_xy_mps, strict=True
        ):
            decision = actor.decide(np.asarray(features, dtype=np.float64))
            prediction = np.asarray(decision.world_command_xy_mps, dtype=np.float64)
            errors.append(prediction - np.asarray(target, dtype=np.float64))
            deltas.append(prediction - previous)
            previous = prediction
            frames += 1
    return {
        "sequence_count": len(sequences),
        "frame_count": frames,
        "rmse_mps": float(np.sqrt(np.mean(np.square(np.asarray(errors))))),
        "command_acceleration_rms_mps_per_step": float(
            np.sqrt(np.mean(np.square(np.asarray(deltas))))
        ),
    }


def _snapshot_summary(snapshot: TemporalRouteSourceSnapshot) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": snapshot.schema_version,
        "source_stage_hash": snapshot.source_stage_hash,
        "source_actor_hash": snapshot.source_actor_hash,
        "source_report_hash": snapshot.source_report_hash,
        "source_actor_file_hash": snapshot.source_actor_file_hash,
        "sequence_ids": [item.sequence_id for item in snapshot.sequences],
        "sequence_count": len(snapshot.sequences),
        "frame_count": sum(len(item.features) for item in snapshot.sequences),
        "trajectory_hashes": list(snapshot.trajectory_hashes),
        "snapshot_hash": snapshot.snapshot_hash,
        "released_source_only": True,
        "current_stage_retention_accessed": False,
    }
    value["summary_hash"] = hash_json(value)
    return value


def _route_command_delta_rms(trajectory: dict[str, NDArray[Any]]) -> float:
    deltas = []
    for role in ("passer", "goalkeeper"):
        commands = np.asarray(trajectory[f"{role}_tactical_world_command"], dtype=np.float64)[:, :2]
        deltas.append(np.diff(commands, axis=0))
    combined = np.concatenate(deltas, axis=0)
    return float(np.sqrt(np.mean(np.square(combined))))


def _save_trajectory(path: Path, trajectory: dict[str, NDArray[Any]]) -> dict[str, str]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **trajectory)  # type: ignore[arg-type]
    os.replace(temporary, path)
    return {
        "file": path.name,
        "file_hash": hash_bytes(path.read_bytes()),
        "trajectory_digest": trajectory_digest(trajectory),
    }


def _verify_hash(payload: dict[str, Any], key: str, label: str) -> None:
    claimed = payload.get(key)
    body = dict(payload)
    body.pop(key, None)
    if claimed != hash_json(body):
        raise ValueError(f"{label} content hash changed")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "TemporalRouteRetentionManifest",
    "TemporalRouteSourceSnapshot",
    "build_temporal_failure_sequences",
    "collect_temporal_failure_development",
    "default_temporal_failure_development_cases",
    "default_temporal_route_retention_manifest",
    "default_temporal_route_retention_manifest_v2",
    "default_temporal_route_retention_manifest_v3",
    "default_temporal_route_retention_manifest_v4",
    "default_temporal_route_retention_manifest_v5",
    "default_temporal_route_retention_manifest_v6",
    "default_temporal_route_retention_manifest_v7",
    "default_temporal_route_retention_manifest_v8",
    "load_released_s122_temporal_snapshot",
    "load_temporal_development_sequences",
    "run_temporal_route_retention_exam",
    "train_temporal_route_population",
]
