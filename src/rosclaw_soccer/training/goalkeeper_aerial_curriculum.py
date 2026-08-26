"""Strict low-to-aerial CPU-MuJoCo curriculum for one goalkeeper candidate.

The frozen S79 low-shot population and deterministic aerial launcher share one
goalkeeper configuration.  Low shots retain the qualified block expert while a
causal threat router grants an online PPO actor or a central GMT expert
exclusive authority only for aerial threats.  This is a development exam, not
product promotion or hardware authorization.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.goalkeeper_v2.policy import load_goalkeeper_actor_artifact
from rosclaw_soccer.skills.team.development_evidence import three_role_development_kwargs
from rosclaw_soccer.skills.team.shared_world import G1GoalkeeperConfig, simulate_shared_world
from rosclaw_soccer.training.shot_save_league import _simulation_kwargs
from rosclaw_soccer.training.shot_save_population import ShotSavePopulationConfig


@dataclass(frozen=True)
class GoalkeeperAerialCurriculumCase:
    case_id: str
    band: str
    target_y_m: float
    target_z_m: float
    flight_sec: float
    required: bool = True
    launch_vertical_bias_mps: float = 0.0
    minimum_contact_height_m: float = 0.0
    schema_version: str = "rosclaw_soccer.goalkeeper_aerial_curriculum_case.v2"

    def __post_init__(self) -> None:
        values = (
            self.target_y_m,
            self.target_z_m,
            self.flight_sec,
            self.launch_vertical_bias_mps,
            self.minimum_contact_height_m,
        )
        if (
            not self.case_id
            or len(self.case_id) > 64
            or not self.case_id.replace("-", "").isalnum()
            or self.band not in {"MID", "HIGH_CENTER", "HIGH_INNER", "HIGH_CORNER"}
            or not all(math.isfinite(value) for value in values)
            or not -1.20 <= self.target_y_m <= 1.20
            or not 0.80 <= self.target_z_m <= 1.70
            or not 0.80 <= self.flight_sec <= 2.00
            or not 0.0 <= self.launch_vertical_bias_mps <= 0.50
            or not 0.0 <= self.minimum_contact_height_m <= 1.80
            or (self.minimum_contact_height_m > 0.0 and self.band != "HIGH_CORNER")
        ):
            raise ValueError("goalkeeper aerial curriculum case is invalid")

    @property
    def case_hash(self) -> str:
        return str(hash_json(asdict(self)))


_DEFAULT_AERIAL_CASES = (
    GoalkeeperAerialCurriculumCase("mid-left", "MID", -0.70, 1.20, 1.20),
    GoalkeeperAerialCurriculumCase("mid-right", "MID", 0.70, 1.20, 1.20),
    GoalkeeperAerialCurriculumCase("high-center", "HIGH_CENTER", 0.0, 1.40, 1.20),
    GoalkeeperAerialCurriculumCase("high-inner-left", "HIGH_INNER", -0.70, 1.40, 1.40),
    GoalkeeperAerialCurriculumCase("high-inner-right", "HIGH_INNER", 0.70, 1.40, 1.40),
    GoalkeeperAerialCurriculumCase("high-corner-left", "HIGH_CORNER", -1.0, 1.50, 1.60),
    GoalkeeperAerialCurriculumCase("high-corner-right", "HIGH_CORNER", 1.0, 1.50, 1.60),
    GoalkeeperAerialCurriculumCase(
        "true-high-corner-left",
        "HIGH_CORNER",
        -1.0,
        1.50,
        1.60,
        launch_vertical_bias_mps=0.30,
        minimum_contact_height_m=1.30,
    ),
    GoalkeeperAerialCurriculumCase(
        "true-high-corner-right",
        "HIGH_CORNER",
        1.0,
        1.50,
        1.60,
        launch_vertical_bias_mps=0.30,
        minimum_contact_height_m=1.30,
    ),
    GoalkeeperAerialCurriculumCase(
        "frontier-corner-left", "HIGH_CORNER", -1.0, 1.50, 1.40, required=False
    ),
    GoalkeeperAerialCurriculumCase(
        "frontier-corner-right", "HIGH_CORNER", 1.0, 1.50, 1.40, required=False
    ),
)


@dataclass(frozen=True)
class GoalkeeperAerialCurriculumConfig:
    aerial_cases: tuple[GoalkeeperAerialCurriculumCase, ...] = _DEFAULT_AERIAL_CASES
    actor_minimum_target_height_m: float = 0.80
    actor_minimum_current_ball_height_m: float = 0.30
    actor_minimum_incoming_ball_speed_mps: float = 2.50
    actor_threat_warmup_sec: float = 0.04
    actor_minimum_intercept_confidence: float = 0.25
    actor_operational_space_reach_local_x_m: float = 0.0
    actor_operational_space_reach_side_offset_m: float = 0.10
    gmt_maximum_lateral_error_m: float = 0.18
    minimum_pelvis_height_m: float = 0.65
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_aerial_curriculum_config.v2"

    def __post_init__(self) -> None:
        identifiers = tuple(case.case_id for case in self.aerial_cases)
        required_bands = {case.band for case in self.aerial_cases if case.required}
        if (
            len(self.aerial_cases) < 7
            or len(set(identifiers)) != len(identifiers)
            or required_bands != {"MID", "HIGH_CENTER", "HIGH_INNER", "HIGH_CORNER"}
            or not 0.0 <= self.actor_minimum_target_height_m <= 1.60
            or not 0.0 <= self.actor_minimum_current_ball_height_m <= 1.0
            or not 0.10 <= self.actor_minimum_incoming_ball_speed_mps <= 15.0
            or not 0.04 <= self.actor_threat_warmup_sec <= 0.20
            or not 0.25 <= self.actor_minimum_intercept_confidence <= 0.80
            or not -0.35 <= self.actor_operational_space_reach_local_x_m <= 0.35
            or not -0.20 <= self.actor_operational_space_reach_side_offset_m <= 0.20
            or any(
                not -0.35 <= value <= 0.35
                for value in (
                    self.actor_operational_space_reach_local_x_m
                    - self.actor_operational_space_reach_side_offset_m,
                    self.actor_operational_space_reach_local_x_m
                    + self.actor_operational_space_reach_side_offset_m,
                )
            )
            or not 0.10 <= self.gmt_maximum_lateral_error_m <= 1.50
            or not 0.55 <= self.minimum_pelvis_height_m <= 0.80
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("goalkeeper aerial curriculum config is invalid")

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
        save = cast(Any, np.savez_compressed)
        save(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _implementation_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        root / "skills/team/shared_world.py",
        root / "skills/goalkeeper_v2/observations.py",
        root / "skills/goalkeeper_v2/policy.py",
        root / "training/shot_save_league.py",
        root / "training/shot_save_population.py",
    ):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _hybrid_goalkeeper_config(
    parent: G1GoalkeeperConfig,
    *,
    actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    config: GoalkeeperAerialCurriculumConfig,
) -> G1GoalkeeperConfig:
    return replace(
        parent,
        actor_observation_mode="visible_ball_history_v3",
        actor_artifact_path=actor_path,
        actor_minimum_target_height_m=config.actor_minimum_target_height_m,
        actor_minimum_current_ball_height_m=config.actor_minimum_current_ball_height_m,
        actor_minimum_incoming_ball_speed_mps=(config.actor_minimum_incoming_ball_speed_mps),
        actor_threat_warmup_sec=config.actor_threat_warmup_sec,
        actor_minimum_intercept_confidence=config.actor_minimum_intercept_confidence,
        actor_operational_space_reach_local_x_m=(config.actor_operational_space_reach_local_x_m),
        actor_operational_space_reach_side_offset_m=(
            config.actor_operational_space_reach_side_offset_m
        ),
        block_action_timing_mode="shooter_phase",
        mosaic_gmt_model_path=gmt_model_path,
        mosaic_gmt_skill_path=gmt_skill_path,
        mosaic_gmt_blend=1.0,
        mosaic_gmt_maximum_lateral_error_m=config.gmt_maximum_lateral_error_m,
        mosaic_gmt_lower_body_scale=1.0,
        mosaic_gmt_waist_scale=1.0,
        mosaic_gmt_arm_scale=1.0,
    )


def _safe(result: Any, *, minimum_pelvis_height_m: float) -> bool:
    return bool(
        result.finite_state
        and not result.joint_limit_violation
        and not result.torque_limit_violation
        and not result.actuator_saturation
        and not result.goalkeeper_joint_limit_violation
        and float(result.goalkeeper_min_pelvis_height_m or 0.0) >= minimum_pelvis_height_m
    )


def _outcome(
    *,
    case_id: str,
    band: str,
    required: bool,
    result: Any,
    trajectory: dict[str, np.ndarray],
    minimum_pelvis_height_m: float,
    minimum_contact_height_m: float = 0.0,
) -> dict[str, Any]:
    safe = _safe(result, minimum_pelvis_height_m=minimum_pelvis_height_m)
    contact_frames = np.flatnonzero(trajectory["goalkeeper_ball_contact"])
    contact_height_m = (
        None
        if not contact_frames.size
        else float(trajectory["ball_pose"][int(contact_frames[0]), 2])
    )
    passed = bool(
        safe
        and result.goalkeeper_ball_contact_observed
        and result.goalkeeper_save_observed
        and not result.goal_crossed
        and contact_height_m is not None
        and contact_height_m >= minimum_contact_height_m
    )
    return {
        "case_id": case_id,
        "band": band,
        "required": required,
        "safe": safe,
        "passed": passed,
        "contact_height_m": contact_height_m,
        "minimum_contact_height_m": minimum_contact_height_m,
        "trajectory_digest": trajectory_digest(trajectory),
        "result": result.to_dict(),
    }


def run_goalkeeper_aerial_curriculum(
    *,
    asset_root: Path,
    actor_artifact_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    output_dir: Path,
    source_checkout: Path,
    config: GoalkeeperAerialCurriculumConfig | None = None,
) -> dict[str, Any]:
    """Execute frozen low replay and the progressive aerial curriculum twice."""

    active = config or GoalkeeperAerialCurriculumConfig()
    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    actor_path = actor_artifact_path.expanduser().resolve()
    model_path = gmt_model_path.expanduser().resolve()
    skill_path = gmt_skill_path.expanduser().resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("aerial curriculum evidence must be new and outside the checkout")
    if not actor_path.is_file() or not model_path.is_file() or not skill_path.is_file():
        raise ValueError("aerial curriculum artifacts are unavailable")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    actor = load_goalkeeper_actor_artifact(actor_path)
    if actor.body_hash != qualification.body_hash:
        raise ValueError("aerial goalkeeper actor Body hash mismatch")
    output.mkdir(parents=True)

    low_population = ShotSavePopulationConfig()
    low_rows: list[dict[str, Any]] = []
    aerial_rows: list[dict[str, Any]] = []
    archives: dict[str, dict[str, Any]] = {}
    strict_replay = True

    for shot in low_population.shots:
        kwargs = _simulation_kwargs(
            striker=shot,
            goalkeeper=low_population.candidate_goalkeeper,
        )
        parent = kwargs.get("goalkeeper_config")
        if not isinstance(parent, G1GoalkeeperConfig):
            raise RuntimeError("frozen low goalkeeper config is unavailable")
        kwargs["goalkeeper_config"] = _hybrid_goalkeeper_config(
            parent,
            actor_path=actor_path,
            gmt_model_path=model_path,
            gmt_skill_path=skill_path,
            config=active,
        )
        kwargs["simulation_duration_sec"] = low_population.simulation_duration_sec
        first, first_trajectory = simulate_shared_world(asset_root, **kwargs)
        replay, replay_trajectory = simulate_shared_world(asset_root, **kwargs)
        first_row = _outcome(
            case_id=shot.policy_id,
            band="LOW",
            required=True,
            result=first,
            trajectory=first_trajectory,
            minimum_pelvis_height_m=active.minimum_pelvis_height_m,
        )
        replay_row = _outcome(
            case_id=shot.policy_id,
            band="LOW",
            required=True,
            result=replay,
            trajectory=replay_trajectory,
            minimum_pelvis_height_m=active.minimum_pelvis_height_m,
        )
        strict_replay &= bool(first_row == replay_row)
        archive = output / f"low-{shot.policy_id}-trajectory.npz"
        _atomic_npz(archive, replay_trajectory)
        archives[f"LOW:{shot.policy_id}"] = {
            "path": archive.name,
            "hash": hash_bytes(archive.read_bytes()),
            "trajectory_digest": replay_row["trajectory_digest"],
        }
        low_rows.append(first_row)

    base = three_role_development_kwargs()
    base_parent = _simulation_kwargs(
        striker=low_population.shots[0],
        goalkeeper=low_population.candidate_goalkeeper,
    )["goalkeeper_config"]
    if not isinstance(base_parent, G1GoalkeeperConfig):
        raise RuntimeError("aerial goalkeeper parent config is unavailable")
    hybrid = _hybrid_goalkeeper_config(
        base_parent,
        actor_path=actor_path,
        gmt_model_path=model_path,
        gmt_skill_path=skill_path,
        config=active,
    )
    keeper_x = float(base["goal_spec"].plane_x_m - hybrid.depth_from_goal_line_m)
    launcher = (keeper_x - 5.0, 0.0, 0.60)
    for case in active.aerial_cases:
        velocity = (
            5.0 / case.flight_sec,
            case.target_y_m / case.flight_sec,
            (case.target_z_m - 0.60 + 4.905 * case.flight_sec**2) / case.flight_sec
            + case.launch_vertical_bias_mps,
        )
        kwargs = dict(base)
        kwargs.update(
            goalkeeper_config=hybrid,
            ball_launcher_position_m=launcher,
            ball_launcher_velocity_mps=velocity,
            simulation_duration_sec=max(2.0, case.flight_sec + 0.80),
            shooter_start_sec=math.inf,
        )
        first, first_trajectory = simulate_shared_world(asset_root, **kwargs)
        replay, replay_trajectory = simulate_shared_world(asset_root, **kwargs)
        first_row = _outcome(
            case_id=case.case_id,
            band=case.band,
            required=case.required,
            result=first,
            trajectory=first_trajectory,
            minimum_pelvis_height_m=active.minimum_pelvis_height_m,
            minimum_contact_height_m=case.minimum_contact_height_m,
        )
        first_row["target_y_m"] = case.target_y_m
        first_row["target_z_m"] = case.target_z_m
        first_row["flight_sec"] = case.flight_sec
        replay_row = _outcome(
            case_id=case.case_id,
            band=case.band,
            required=case.required,
            result=replay,
            trajectory=replay_trajectory,
            minimum_pelvis_height_m=active.minimum_pelvis_height_m,
            minimum_contact_height_m=case.minimum_contact_height_m,
        )
        replay_row["target_y_m"] = case.target_y_m
        replay_row["target_z_m"] = case.target_z_m
        replay_row["flight_sec"] = case.flight_sec
        strict_replay &= bool(first_row == replay_row)
        archive = output / f"aerial-{case.case_id}-trajectory.npz"
        _atomic_npz(archive, replay_trajectory)
        archives[f"{case.band}:{case.case_id}"] = {
            "path": archive.name,
            "hash": hash_bytes(archive.read_bytes()),
            "trajectory_digest": replay_row["trajectory_digest"],
        }
        aerial_rows.append(first_row)

    required_rows = tuple(row for row in aerial_rows if bool(row["required"]))
    frontier_rows = tuple(row for row in aerial_rows if not bool(row["required"]))
    low_pass_rate = sum(bool(row["passed"]) for row in low_rows) / len(low_rows)
    required_pass_rate = sum(bool(row["passed"]) for row in required_rows) / len(required_rows)
    frontier_pass_rate = sum(bool(row["passed"]) for row in frontier_rows) / max(
        1, len(frontier_rows)
    )
    safety_rate = sum(bool(row["safe"]) for row in (*low_rows, *aerial_rows)) / (
        len(low_rows) + len(aerial_rows)
    )
    true_high_rows = tuple(
        row for row in aerial_rows if float(row["minimum_contact_height_m"]) > 0.0
    )
    true_high_pass_rate = sum(bool(row["passed"]) for row in true_high_rows) / max(
        1, len(true_high_rows)
    )
    minimum_true_high_contact_height_m = min(
        (
            float(row["contact_height_m"])
            for row in true_high_rows
            if row["contact_height_m"] is not None
        ),
        default=0.0,
    )
    passed = bool(
        low_pass_rate == 1.0 and required_pass_rate == 1.0 and safety_rate == 1.0 and strict_replay
    )
    failures = [
        {
            "case_id": row["case_id"],
            "band": row["band"],
            "trajectory_digest": row["trajectory_digest"],
            "failure": (
                "SAFETY_REGRESSION"
                if not bool(row["safe"])
                else "CONTACT_HEIGHT_GATE_FAILED"
                if row["contact_height_m"] is not None
                and float(row["contact_height_m"]) < float(row["minimum_contact_height_m"])
                else "AERIAL_SAVE_MISSED_OR_INSUFFICIENT_DEFLECTION"
            ),
            "priority": 2.0 if row["band"] == "HIGH_CORNER" else 1.0,
        }
        for row in aerial_rows
        if not bool(row["passed"])
    ]
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.goalkeeper_aerial_curriculum_exam.v2",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "implementation_hash": _implementation_hash(),
        "body_hash": qualification.body_hash,
        "actor_policy_hash": actor.policy_hash,
        "actor_artifact_hash": hash_bytes(actor_path.read_bytes()),
        "gmt_model_hash": hash_bytes(model_path.read_bytes()),
        "gmt_skill_artifact_hash": hash_bytes(skill_path.read_bytes()),
        "physics_authority": "CPU_MUJOCO",
        "low_rows": low_rows,
        "aerial_rows": aerial_rows,
        "trajectory_archives": archives,
        "strict_replay": strict_replay,
        "metrics": {
            "low_pass_rate": low_pass_rate,
            "required_aerial_pass_rate": required_pass_rate,
            "frontier_pass_rate": frontier_pass_rate,
            "safety_rate": safety_rate,
            "true_high_corner_pass_rate": true_high_pass_rate,
            "minimum_true_high_contact_height_m": minimum_true_high_contact_height_m,
        },
        "failure_memory": failures,
        "passed": passed,
        "promotion_status": (
            "DEVELOPMENT_CURRICULUM_PASSED_NOT_PROMOTED"
            if passed
            else "REJECTED_CURRICULUM_FAILURE"
        ),
        "sealed_holdout_used": False,
        "promotion_eligible": False,
        "development_only": True,
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
    }
    report["report_hash"] = hash_json(report)
    report_path = output / "goalkeeper-aerial-curriculum-exam.json"
    _atomic_json(report_path, report)
    return validate_goalkeeper_aerial_curriculum(report_path)


def validate_goalkeeper_aerial_curriculum(path: Path) -> dict[str, Any]:
    report_path = path.expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("goalkeeper aerial curriculum report must be an object")
    expected = payload.pop("report_hash", None)
    if expected != hash_json(payload):
        raise ValueError("goalkeeper aerial curriculum report integrity mismatch")
    config = payload.get("config")
    cases = None if not isinstance(config, dict) else config.get("aerial_cases")
    low_rows = payload.get("low_rows")
    aerial_rows = payload.get("aerial_rows")
    archives = payload.get("trajectory_archives")
    if (
        not isinstance(cases, list)
        or not isinstance(low_rows, list)
        or not isinstance(aerial_rows, list)
        or len(low_rows) != 8
        or len(aerial_rows) != len(cases)
        or not isinstance(archives, dict)
        or len(archives) != len(low_rows) + len(aerial_rows)
    ):
        raise ValueError("goalkeeper aerial curriculum archives are incomplete")
    for contract in archives.values():
        if not isinstance(contract, dict) or not isinstance(contract.get("path"), str):
            raise ValueError("goalkeeper aerial curriculum archive contract is invalid")
        archive = report_path.parent / contract["path"]
        if not archive.is_file() or contract.get("hash") != hash_bytes(archive.read_bytes()):
            raise ValueError("goalkeeper aerial curriculum archive integrity mismatch")
    true_high_rows = [
        row
        for row in aerial_rows
        if isinstance(row, dict) and float(row.get("minimum_contact_height_m", 0.0)) > 0.0
    ]
    metrics = payload.get("metrics")
    if (
        payload.get("schema_version") != "rosclaw_soccer.goalkeeper_aerial_curriculum_exam.v2"
        or payload.get("implementation_hash") != _implementation_hash()
        or payload.get("physics_authority") != "CPU_MUJOCO"
        or payload.get("strict_replay") is not True
        or payload.get("passed") is not True
        or payload.get("sealed_holdout_used") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("development_only") is not True
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
        or payload.get("hardware_command_sent") is not False
        or payload.get("pixels_used_for_scoring") is not False
        or len(true_high_rows) != 2
        or any(
            row.get("passed") is not True
            or not isinstance(row.get("contact_height_m"), int | float)
            or float(row["contact_height_m"]) < float(row["minimum_contact_height_m"])
            for row in true_high_rows
        )
        or not isinstance(metrics, dict)
        or metrics.get("low_pass_rate") != 1.0
        or metrics.get("required_aerial_pass_rate") != 1.0
        or metrics.get("safety_rate") != 1.0
        or metrics.get("true_high_corner_pass_rate") != 1.0
        or float(metrics.get("minimum_true_high_contact_height_m", 0.0)) < 1.30
    ):
        raise ValueError("goalkeeper aerial curriculum authority contract is invalid")
    payload["report_hash"] = expected
    return cast(dict[str, Any], payload)


__all__ = [
    "GoalkeeperAerialCurriculumCase",
    "GoalkeeperAerialCurriculumConfig",
    "run_goalkeeper_aerial_curriculum",
    "validate_goalkeeper_aerial_curriculum",
]
