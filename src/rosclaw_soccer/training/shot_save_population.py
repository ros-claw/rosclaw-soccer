"""Paired population exam for a striker skill family and goalkeeper champion.

This is a downstream, SIM_ONLY development gate.  Eight immutable low-shot
skills challenge the parent and candidate goalkeepers in the same CPU MuJoCo
worlds.  It may replace the *development* champion, but cannot grant product
promotion without a separately sealed holdout.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.paired_champion_gate import (
    ChampionMetricSpec,
    ChampionSnapshot,
    decision_payload,
    evaluate_paired_champion,
)
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world
from rosclaw_soccer.training.shot_save_league import (
    ShotSaveGoalkeeperPolicy,
    ShotSaveLeagueConfig,
    ShotSaveStrikerPolicy,
    _cell,
    _simulation_kwargs,
)


@dataclass(frozen=True)
class ShotSavePopulationConfig:
    shots: tuple[ShotSaveStrikerPolicy, ...] = (
        ShotSaveStrikerPolicy(
            "population-center",
            0.2437420748734585,
            0.13147022232288466,
            0.0,
            0.8,
        ),
        ShotSaveStrikerPolicy(
            "population-inner",
            0.4661039719675122,
            0.12082048585243889,
            0.2,
            0.8,
            foot_yaw_offset_rad=0.04,
        ),
        ShotSaveStrikerPolicy(
            "population-mid",
            0.6481174583568615,
            0.1173229587165438,
            0.35,
            0.8,
        ),
        ShotSaveStrikerPolicy(
            "population-midpitch",
            0.6661285505602982,
            0.12464610090477349,
            0.5,
            0.8,
            foot_pitch_offset_rad=-0.12,
        ),
        ShotSaveStrikerPolicy(
            "population-far",
            0.8363412529331966,
            0.12093718793269315,
            0.5,
            0.8,
        ),
        ShotSaveStrikerPolicy(
            "population-edge",
            1.0856546026055292,
            0.11856537841399575,
            0.35,
            0.5,
        ),
        ShotSaveStrikerPolicy(
            "population-edgephase",
            1.2650946844279862,
            0.12782772619512844,
            0.7,
            1.1,
        ),
        ShotSaveStrikerPolicy(
            "population-edgewide",
            1.3363137719638016,
            0.13078289085723951,
            0.7,
            0.8,
        ),
    )
    parent_goalkeeper: ShotSaveGoalkeeperPolicy = ShotSaveGoalkeeperPolicy(
        "goalkeeper-g0-upright-block",
        0.48,
        0.265,
    )
    candidate_goalkeeper: ShotSaveGoalkeeperPolicy = ShotSaveGoalkeeperPolicy(
        "goalkeeper-g1-low-dive-a",
        0.25,
        0.265,
        -0.40,
        -0.40,
    )
    simulation_duration_sec: float = 10.8
    minimum_shot_speed_mps: float = 6.0
    minimum_save_rate: float = 0.80
    minimum_save_rate_gain: float = 0.125
    minimum_goalkeeper_pelvis_height_m: float = 0.65
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.shot_save_population_config.v1"

    def __post_init__(self) -> None:
        identifiers = tuple(item.policy_id for item in self.shots)
        if (
            len(self.shots) < 8
            or len(set(identifiers)) != len(identifiers)
            or self.parent_goalkeeper.policy_hash == self.candidate_goalkeeper.policy_hash
            or not 9.0 <= self.simulation_duration_sec <= 15.0
            or not 4.0 <= self.minimum_shot_speed_mps <= 15.0
            or not 0.5 <= self.minimum_save_rate <= 1.0
            or not 0.0 < self.minimum_save_rate_gain <= 0.5
            or not 0.55 <= self.minimum_goalkeeper_pelvis_height_m <= 0.80
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("shot-save population config is invalid")

    @property
    def scenario_suite_hash(self) -> str:
        return str(
            hash_json(
                {
                    "shot_policy_hashes": [item.policy_hash for item in self.shots],
                    "simulation_duration_sec": self.simulation_duration_sec,
                    "physics_authority": "CPU_MUJOCO",
                    "shared_ball_state": True,
                }
            )
        )

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        savez_compressed = cast(Any, np.savez_compressed)
        savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__),
        root / "training/shot_save_league.py",
        root / "skills/team/shared_world.py",
        root / "providers/g1/mujoco_primitives.py",
        root / "world/field.py",
        root / "growth/paired_champion_gate.py",
    )
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _run_cell(
    *,
    asset_root: Path,
    shot: ShotSaveStrikerPolicy,
    goalkeeper: ShotSaveGoalkeeperPolicy,
    stage: str,
    config: ShotSavePopulationConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    league_config = ShotSaveLeagueConfig(
        minimum_goalkeeper_pelvis_height_m=config.minimum_goalkeeper_pelvis_height_m
    )
    kwargs = _simulation_kwargs(striker=shot, goalkeeper=goalkeeper)
    kwargs["simulation_duration_sec"] = config.simulation_duration_sec
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    cell = _cell(
        stage=stage,
        striker=shot,
        goalkeeper=goalkeeper,
        result=result,
        trajectory=trajectory,
        config=league_config,
    )
    challenge_valid = bool(
        cell["safe"]
        and result.pass_contact_observed
        and result.shot_contact_observed
        and result.shot_peak_ball_speed_mps >= config.minimum_shot_speed_mps
        and result.shooter_min_pelvis_height_m >= league_config.minimum_striker_pelvis_height_m
    )
    cell["challenge_valid"] = challenge_valid
    cell["keeper_exam_passed"] = bool(challenge_valid and cell["goalkeeper_eligible"])
    return cell, trajectory


def _metrics(cells: tuple[dict[str, Any], ...]) -> dict[str, float]:
    count = len(cells)
    pelvis = tuple(
        float(cell["result"]["goalkeeper_min_pelvis_height_m"] or 0.0) for cell in cells
    )
    return {
        "save_rate": sum(bool(cell["keeper_exam_passed"]) for cell in cells) / count,
        "safety_rate": sum(bool(cell["safe"]) for cell in cells) / count,
        "contact_rate": sum(
            bool(cell["result"]["goalkeeper_ball_contact_observed"]) for cell in cells
        )
        / count,
        "valid_challenge_rate": sum(bool(cell["challenge_valid"]) for cell in cells) / count,
        "minimum_goalkeeper_pelvis_height_m": min(pelvis),
    }


def _metric_specs(config: ShotSavePopulationConfig) -> tuple[ChampionMetricSpec, ...]:
    return (
        ChampionMetricSpec(
            "save_rate",
            "MAXIMIZE",
            minimum_improvement=config.minimum_save_rate_gain,
            hard_lower_bound=config.minimum_save_rate,
        ),
        ChampionMetricSpec(
            "safety_rate",
            "MAXIMIZE",
            hard_lower_bound=1.0,
        ),
        ChampionMetricSpec(
            "contact_rate",
            "MAXIMIZE",
            maximum_regression=0.0,
            hard_lower_bound=config.minimum_save_rate,
        ),
        ChampionMetricSpec(
            "valid_challenge_rate",
            "MAXIMIZE",
            hard_lower_bound=1.0,
        ),
        ChampionMetricSpec(
            "minimum_goalkeeper_pelvis_height_m",
            "MAXIMIZE",
            maximum_regression=0.02,
            hard_lower_bound=config.minimum_goalkeeper_pelvis_height_m,
        ),
    )


def run_shot_save_population_exam(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
    config: ShotSavePopulationConfig | None = None,
) -> dict[str, Any]:
    """Run paired parent/candidate matrices and strict replay every cell."""

    active = config or ShotSavePopulationConfig()
    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("population evidence output must be new and outside the checkout")
    output.mkdir(parents=True)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()

    matrices: dict[str, tuple[dict[str, Any], ...]] = {}
    strict_replay = True
    archives: dict[str, dict[str, Any]] = {}
    for role, goalkeeper in (
        ("parent", active.parent_goalkeeper),
        ("candidate", active.candidate_goalkeeper),
    ):
        cells: list[dict[str, Any]] = []
        for shot in active.shots:
            cell, _ = _run_cell(
                asset_root=asset_root,
                shot=shot,
                goalkeeper=goalkeeper,
                stage=f"{role.upper()}_DISCOVERY",
                config=active,
            )
            replay, trajectory = _run_cell(
                asset_root=asset_root,
                shot=shot,
                goalkeeper=goalkeeper,
                stage=f"{role.upper()}_STRICT_REPLAY",
                config=active,
            )
            strict_replay &= bool(
                replay["result"] == cell["result"]
                and replay["trajectory_digest"] == cell["trajectory_digest"]
                and replay["challenge_valid"] == cell["challenge_valid"]
                and replay["keeper_exam_passed"] == cell["keeper_exam_passed"]
            )
            archive_path = output / f"{role}-{shot.policy_id}-trajectory.npz"
            _atomic_npz(archive_path, trajectory)
            archives[f"{role}:{shot.policy_id}"] = {
                "path": archive_path.name,
                "hash": hash_bytes(archive_path.read_bytes()),
                "trajectory_digest": replay["trajectory_digest"],
                "frame_count": int(len(trajectory["time"])),
                "contains_scored_physics_state": True,
            }
            cells.append(cell)
        matrices[role] = tuple(cells)

    parent_metrics = _metrics(matrices["parent"])
    candidate_metrics = _metrics(matrices["candidate"])
    parent_qualified = bool(
        parent_metrics["save_rate"] >= active.minimum_save_rate
        and parent_metrics["safety_rate"] == 1.0
        and parent_metrics["valid_challenge_rate"] == 1.0
    )
    candidate_qualified = bool(
        candidate_metrics["save_rate"] >= active.minimum_save_rate
        and candidate_metrics["safety_rate"] == 1.0
        and candidate_metrics["valid_challenge_rate"] == 1.0
        and strict_replay
    )
    parent_snapshot = ChampionSnapshot(
        artifact_hash=active.parent_goalkeeper.policy_hash,
        parent_artifact_hash=None,
        scenario_suite_hash=active.scenario_suite_hash,
        episode_count=len(active.shots),
        metrics=tuple(parent_metrics.items()),
        qualified=parent_qualified,
    )
    candidate_snapshot = ChampionSnapshot(
        artifact_hash=active.candidate_goalkeeper.policy_hash,
        parent_artifact_hash=active.parent_goalkeeper.policy_hash,
        scenario_suite_hash=active.scenario_suite_hash,
        episode_count=len(active.shots),
        metrics=tuple(candidate_metrics.items()),
        qualified=candidate_qualified,
    )
    metric_specs = _metric_specs(active)
    decision = evaluate_paired_champion(
        parent=parent_snapshot,
        candidate=candidate_snapshot,
        metrics=metric_specs,
    )
    gate = decision_payload(
        decision,
        parent=parent_snapshot,
        candidate=candidate_snapshot,
        metrics=metric_specs,
    )
    failures = [
        {
            "shot_policy_id": cell["striker_policy_id"],
            "shot_policy_hash": cell["striker_policy_hash"],
            "goalkeeper_policy_hash": cell["goalkeeper_policy_hash"],
            "failure": "ATTACKER_BLOCKED_BY_DEVELOPMENT_CHAMPION",
            "trajectory_digest": cell["trajectory_digest"],
            "priority": 1.0 + float(cell["result"]["shot_peak_ball_speed_mps"]) / 20.0,
        }
        for cell in matrices["candidate"]
        if cell["keeper_exam_passed"]
    ]
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.shot_save_population_exam.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "scenario_suite_hash": active.scenario_suite_hash,
        "implementation_hash": _implementation_hash(),
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "backend_commit": qualification.backend_commit,
        "physics_authority": "CPU_MUJOCO",
        "physics_backend": "single_shared_world_three_g1",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "parent_cells": list(matrices["parent"]),
        "candidate_cells": list(matrices["candidate"]),
        "parent_metrics": parent_metrics,
        "candidate_metrics": candidate_metrics,
        "strict_replay": strict_replay,
        "trajectory_archives": archives,
        "development_champion_gate": gate,
        "development_champion_replaced": decision.replace_champion,
        "attacker_failure_memory": failures,
        "next_learner": "STRIKER",
        "sealed_holdout_used": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
    }
    report["report_hash"] = hash_json(report)
    path = output / "shot-save-population-exam.json"
    _atomic_json(path, report)
    return validate_shot_save_population_exam(path)


def validate_shot_save_population_exam(path: Path) -> dict[str, Any]:
    report_path = path.expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("shot-save population report must be a JSON object")
    expected = payload.pop("report_hash", None)
    if expected != hash_json(payload):
        raise ValueError("shot-save population report integrity mismatch")
    archives = payload.get("trajectory_archives")
    if not isinstance(archives, dict) or len(archives) < 16:
        raise ValueError("shot-save population trajectory suite is incomplete")
    for contract in archives.values():
        if not isinstance(contract, dict) or not isinstance(contract.get("path"), str):
            raise ValueError("shot-save population trajectory contract is invalid")
        trajectory = report_path.parent / contract["path"]
        if (
            not trajectory.is_file()
            or contract.get("hash") != hash_bytes(trajectory.read_bytes())
            or contract.get("contains_scored_physics_state") is not True
        ):
            raise ValueError("shot-save population trajectory integrity mismatch")
    if (
        payload.get("schema_version") != "rosclaw_soccer.shot_save_population_exam.v1"
        or payload.get("implementation_hash") != _implementation_hash()
        or payload.get("physics_authority") != "CPU_MUJOCO"
        or payload.get("strict_replay") is not True
        or payload.get("development_champion_replaced") is not True
        or payload.get("sealed_holdout_used") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
        or payload.get("hardware_command_sent") is not False
        or payload.get("pixels_used_for_scoring") is not False
    ):
        raise ValueError("shot-save population authority contract is invalid")
    payload["report_hash"] = expected
    return cast(dict[str, Any], payload)


__all__ = [
    "ShotSavePopulationConfig",
    "run_shot_save_population_exam",
    "validate_shot_save_population_exam",
]
