"""Alternating striker/goalkeeper best responses in one MuJoCo world.

The league is a downstream SIM_ONLY Growth dojo.  It proves that each role can
produce a safe best response to a frozen opponent, but it does not promote
either response without disjoint seed and scenario holdouts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
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
from rosclaw_soccer.skills.team.development_evidence import (
    three_role_development_kwargs,
)
from rosclaw_soccer.skills.team.goalkeeper_learning import (
    goalkeeper_block_parent_config,
)
from rosclaw_soccer.skills.team.shared_world import (
    G1GoalkeeperConfig,
    G1SharedWorldResult,
    simulate_shared_world,
)


@dataclass(frozen=True)
class ShotSaveStrikerPolicy:
    policy_id: str
    physical_target_y_m: float
    physical_target_z_m: float
    policy_target_y_m: float
    policy_target_z_m: float
    foot_yaw_offset_rad: float = 0.085
    foot_pitch_offset_rad: float = 0.010
    kick_foot: str = "right"
    # Preserve the qualified right-foot parent's 0.175 rad approach posture.
    # A new option must opt out explicitly instead of silently forgetting it.
    pelvis_yaw_offset_rad: float = 0.175
    loft_synergy: float = 0.0
    schema_version: str = "rosclaw_soccer.shot_save_striker_policy.v1"

    def __post_init__(self) -> None:
        values = (
            self.physical_target_y_m,
            self.physical_target_z_m,
            self.policy_target_y_m,
            self.policy_target_z_m,
            self.foot_yaw_offset_rad,
            self.foot_pitch_offset_rad,
            self.pelvis_yaw_offset_rad,
            self.loft_synergy,
        )
        if (
            not self.policy_id
            or len(self.policy_id) > 64
            or not self.policy_id.replace("-", "").isalnum()
            or not all(math.isfinite(value) for value in values)
            # The population exam uses a 3.0 m goal and a 0.115 m ball.  The
            # complete ball centre may therefore reach |y|=1.385 m.
            or not -1.385 <= self.physical_target_y_m <= 1.385
            or not 0.115 <= self.physical_target_z_m <= 1.885
            or not -1.385 <= self.policy_target_y_m <= 1.385
            or not 0.0 <= self.policy_target_z_m <= 2.0
            or not -0.12 <= self.foot_yaw_offset_rad <= 0.12
            or not -0.18 <= self.foot_pitch_offset_rad <= 0.18
            or self.kick_foot not in {"left", "right"}
            or not -0.20 <= self.pelvis_yaw_offset_rad <= 0.20
            or not 0.0 <= self.loft_synergy <= 0.30
        ):
            raise ValueError("shot-save striker policy is invalid")

    @property
    def policy_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class ShotSaveGoalkeeperPolicy:
    policy_id: str
    depth_from_goal_line_m: float
    block_action_hip_pitch_rad: float
    block_action_shoulder_pitch_rad: float = 0.0
    block_action_shoulder_roll_rad: float = 0.0
    schema_version: str = "rosclaw_soccer.shot_save_goalkeeper_policy.v1"

    def __post_init__(self) -> None:
        values = (
            self.depth_from_goal_line_m,
            self.block_action_hip_pitch_rad,
            self.block_action_shoulder_pitch_rad,
            self.block_action_shoulder_roll_rad,
        )
        if (
            not self.policy_id
            or len(self.policy_id) > 64
            or not self.policy_id.replace("-", "").isalnum()
            or not all(math.isfinite(value) for value in values)
            or not 0.25 <= self.depth_from_goal_line_m <= 0.80
            or not 0.0 <= self.block_action_hip_pitch_rad <= 0.50
            or abs(self.block_action_shoulder_pitch_rad) > 1.0
            or abs(self.block_action_shoulder_roll_rad) > 0.8
        ):
            raise ValueError("shot-save goalkeeper policy is invalid")

    @property
    def policy_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class ShotSaveLeagueConfig:
    parent_striker: ShotSaveStrikerPolicy = ShotSaveStrikerPolicy(
        policy_id="striker-g0-low-corner",
        physical_target_y_m=0.89,
        physical_target_z_m=0.115,
        policy_target_y_m=0.70,
        policy_target_z_m=0.50,
    )
    striker_candidates: tuple[ShotSaveStrikerPolicy, ...] = (
        ShotSaveStrikerPolicy(
            policy_id="striker-g1-yaw-only-failure",
            physical_target_y_m=0.89,
            physical_target_z_m=0.115,
            policy_target_y_m=0.70,
            policy_target_z_m=0.50,
            foot_yaw_offset_rad=0.055,
        ),
        ShotSaveStrikerPolicy(
            policy_id="striker-g1-contact-phase-escape",
            physical_target_y_m=0.8363412529331966,
            physical_target_z_m=0.12093718793269315,
            policy_target_y_m=0.50,
            policy_target_z_m=0.80,
        ),
    )
    parent_goalkeeper: ShotSaveGoalkeeperPolicy = ShotSaveGoalkeeperPolicy(
        policy_id="goalkeeper-g0-upright-block",
        depth_from_goal_line_m=0.48,
        block_action_hip_pitch_rad=0.265,
    )
    goalkeeper_candidates: tuple[ShotSaveGoalkeeperPolicy, ...] = (
        ShotSaveGoalkeeperPolicy(
            policy_id="goalkeeper-g1-depth-only-failure",
            depth_from_goal_line_m=0.25,
            block_action_hip_pitch_rad=0.265,
        ),
        ShotSaveGoalkeeperPolicy(
            policy_id="goalkeeper-g1-downreach-only-failure",
            depth_from_goal_line_m=0.25,
            block_action_hip_pitch_rad=0.265,
            block_action_shoulder_pitch_rad=-0.40,
        ),
        ShotSaveGoalkeeperPolicy(
            policy_id="goalkeeper-g1-low-dive-a",
            depth_from_goal_line_m=0.25,
            block_action_hip_pitch_rad=0.265,
            block_action_shoulder_pitch_rad=-0.40,
            block_action_shoulder_roll_rad=-0.40,
        ),
        ShotSaveGoalkeeperPolicy(
            policy_id="goalkeeper-g1-low-dive-b",
            depth_from_goal_line_m=0.25,
            block_action_hip_pitch_rad=0.265,
            block_action_shoulder_pitch_rad=-0.40,
            block_action_shoulder_roll_rad=-0.20,
        ),
    )
    maximum_target_error_m: float = 0.10
    minimum_striker_pelvis_height_m: float = 0.55
    minimum_goalkeeper_pelvis_height_m: float = 0.65
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.shot_save_league_config.v1"

    def __post_init__(self) -> None:
        striker_ids = (self.parent_striker.policy_id,) + tuple(
            item.policy_id for item in self.striker_candidates
        )
        goalkeeper_ids = (self.parent_goalkeeper.policy_id,) + tuple(
            item.policy_id for item in self.goalkeeper_candidates
        )
        if (
            not self.striker_candidates
            or not self.goalkeeper_candidates
            or len(set(striker_ids)) != len(striker_ids)
            or len(set(goalkeeper_ids)) != len(goalkeeper_ids)
            or not 0.01 <= self.maximum_target_error_m <= 0.50
            or not 0.45 <= self.minimum_striker_pelvis_height_m <= 0.80
            or not 0.45 <= self.minimum_goalkeeper_pelvis_height_m <= 0.80
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("shot-save league config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _safe_result(result: G1SharedWorldResult) -> bool:
    return bool(
        result.finite_state
        and not result.joint_limit_violation
        and not result.torque_limit_violation
        and not result.actuator_saturation
        and not result.shooter_joint_limit_violation
        and not result.goalkeeper_joint_limit_violation
    )


def _cell(
    *,
    stage: str,
    striker: ShotSaveStrikerPolicy,
    goalkeeper: ShotSaveGoalkeeperPolicy,
    result: G1SharedWorldResult,
    trajectory: Mapping[str, NDArray[Any]],
    config: ShotSaveLeagueConfig,
) -> dict[str, Any]:
    safe = _safe_result(result)
    striker_eligible = bool(
        safe
        and result.pass_contact_observed
        and result.shot_contact_observed
        and result.goal_crossed
        and result.target_error_m is not None
        and result.target_error_m <= config.maximum_target_error_m
        and result.shooter_min_pelvis_height_m >= config.minimum_striker_pelvis_height_m
    )
    goalkeeper_eligible = bool(
        safe
        and result.goalkeeper_ball_contact_observed
        and result.goalkeeper_save_observed
        and not result.goal_crossed
        and (result.goalkeeper_min_pelvis_height_m or 0.0)
        >= config.minimum_goalkeeper_pelvis_height_m
    )
    target_error = 10.0 if result.target_error_m is None else result.target_error_m
    striker_score = (
        10.0 * float(striker_eligible)
        + 2.0 * float(result.goal_crossed)
        - min(target_error, 10.0)
        + 0.02 * result.shot_peak_ball_speed_mps
    )
    goalkeeper_score = (
        10.0 * float(goalkeeper_eligible)
        + 2.0 * float(result.goalkeeper_ball_contact_observed)
        + float(not result.goal_plane_crossed)
        - 20.0 * float(not safe)
    )
    return {
        "stage": stage,
        "striker_policy_id": striker.policy_id,
        "striker_policy_hash": striker.policy_hash,
        "goalkeeper_policy_id": goalkeeper.policy_id,
        "goalkeeper_policy_hash": goalkeeper.policy_hash,
        "trajectory_digest": trajectory_digest(cast(dict[str, np.ndarray], trajectory)),
        "safe": safe,
        "striker_eligible": striker_eligible,
        "goalkeeper_eligible": goalkeeper_eligible,
        "striker_score": striker_score,
        "goalkeeper_score": goalkeeper_score,
        "result": result.to_dict(),
    }


def _simulation_kwargs(
    *, striker: ShotSaveStrikerPolicy, goalkeeper: ShotSaveGoalkeeperPolicy
) -> dict[str, Any]:
    kwargs = three_role_development_kwargs()
    parent = kwargs.get("goalkeeper_config")
    if not isinstance(parent, G1GoalkeeperConfig):
        raise RuntimeError("shot-save league goalkeeper parent is unavailable")
    parent = goalkeeper_block_parent_config(parent)
    kwargs["goalkeeper_config"] = replace(
        parent,
        depth_from_goal_line_m=goalkeeper.depth_from_goal_line_m,
        block_action_enabled=True,
        block_action_hip_pitch_rad=goalkeeper.block_action_hip_pitch_rad,
        block_action_shoulder_pitch_rad=goalkeeper.block_action_shoulder_pitch_rad,
        block_action_shoulder_roll_rad=goalkeeper.block_action_shoulder_roll_rad,
    )
    overrides = dict(kwargs.get("shooter_parameter_overrides", {}))
    overrides.update(
        {
            "foot_yaw_offset": striker.foot_yaw_offset_rad,
            "foot_pitch_offset": striker.foot_pitch_offset_rad,
            "kick_foot": striker.kick_foot,
            "pelvis_yaw_offset": striker.pelvis_yaw_offset_rad,
            "loft_synergy": striker.loft_synergy,
        }
    )
    kwargs["shooter_parameter_overrides"] = overrides
    physical_target = (
        7.50,
        striker.physical_target_y_m,
        striker.physical_target_z_m,
    )
    kwargs["shooter_target"] = physical_target
    kwargs["shooter_policy_target"] = (
        7.50,
        striker.policy_target_y_m,
        striker.policy_target_z_m,
    )
    kwargs["goal_spec"] = replace(
        kwargs["goal_spec"],
        target_y_m=striker.physical_target_y_m,
        target_z_m=striker.physical_target_z_m,
    )
    return kwargs


def _run_cell(
    *,
    asset_root: Path,
    stage: str,
    striker: ShotSaveStrikerPolicy,
    goalkeeper: ShotSaveGoalkeeperPolicy,
    config: ShotSaveLeagueConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result, trajectory = simulate_shared_world(
        asset_root, **_simulation_kwargs(striker=striker, goalkeeper=goalkeeper)
    )
    return (
        _cell(
            stage=stage,
            striker=striker,
            goalkeeper=goalkeeper,
            result=result,
            trajectory=trajectory,
            config=config,
        ),
        trajectory,
    )


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
    paths = (
        Path(__file__),
        Path(__file__).resolve().parents[1] / "skills/team/shared_world.py",
        Path(__file__).resolve().parents[1] / "providers/g1/mujoco_primitives.py",
        Path(__file__).resolve().parents[1] / "world/field.py",
    )
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def run_shot_save_growth_round(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
    config: ShotSaveLeagueConfig | None = None,
) -> dict[str, Any]:
    """Run one attacker response and one defender response with strict replay."""

    active = config or ShotSaveLeagueConfig()
    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("shot-save league output must be new and outside the checkout")
    output.mkdir(parents=True)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()

    cells: list[dict[str, Any]] = []
    parent_cell, _ = _run_cell(
        asset_root=asset_root,
        stage="PARENT_MATCH",
        striker=active.parent_striker,
        goalkeeper=active.parent_goalkeeper,
        config=active,
    )
    cells.append(parent_cell)
    attack_rows: list[tuple[dict[str, Any], ShotSaveStrikerPolicy]] = []
    for striker in active.striker_candidates:
        cell, _ = _run_cell(
            asset_root=asset_root,
            stage="STRIKER_BEST_RESPONSE",
            striker=striker,
            goalkeeper=active.parent_goalkeeper,
            config=active,
        )
        cells.append(cell)
        attack_rows.append((cell, striker))
    selected_attack_cell, selected_striker = max(
        attack_rows, key=lambda item: float(item[0]["striker_score"])
    )

    defense_rows: list[tuple[dict[str, Any], ShotSaveGoalkeeperPolicy]] = []
    for goalkeeper in active.goalkeeper_candidates:
        cell, _ = _run_cell(
            asset_root=asset_root,
            stage="GOALKEEPER_BEST_RESPONSE",
            striker=selected_striker,
            goalkeeper=goalkeeper,
            config=active,
        )
        cells.append(cell)
        defense_rows.append((cell, goalkeeper))
    selected_defense_cell, selected_goalkeeper = max(
        defense_rows, key=lambda item: float(item[0]["goalkeeper_score"])
    )

    replay_cells: list[dict[str, Any]] = []
    trajectory_contracts: dict[str, dict[str, Any]] = {}
    finalists = (
        (
            "attacker-best-response",
            "STRIKER_STRICT_REPLAY",
            selected_striker,
            active.parent_goalkeeper,
            selected_attack_cell,
        ),
        (
            "defender-best-response",
            "GOALKEEPER_STRICT_REPLAY",
            selected_striker,
            selected_goalkeeper,
            selected_defense_cell,
        ),
    )
    strict_replay = True
    for name, stage, striker, goalkeeper, discovery_cell in finalists:
        replay_cell, trajectory = _run_cell(
            asset_root=asset_root,
            stage=stage,
            striker=striker,
            goalkeeper=goalkeeper,
            config=active,
        )
        replay_cells.append(replay_cell)
        strict_replay &= bool(
            replay_cell["result"] == discovery_cell["result"]
            and replay_cell["trajectory_digest"] == discovery_cell["trajectory_digest"]
        )
        trajectory_path = output / f"{name}-trajectory.npz"
        _atomic_npz(trajectory_path, trajectory)
        trajectory_contracts[name] = {
            "path": trajectory_path.name,
            "hash": hash_bytes(trajectory_path.read_bytes()),
            "digest": replay_cell["trajectory_digest"],
            "frame_count": int(len(trajectory["time"])),
            "contains_scored_physics_state": True,
        }

    cycle_complete = bool(
        selected_attack_cell["striker_eligible"]
        and selected_defense_cell["goalkeeper_eligible"]
        and strict_replay
    )
    failure_memory = [
        {
            "stage": cell["stage"],
            "learner": (
                "striker"
                if cell["stage"] == "STRIKER_BEST_RESPONSE"
                else "goalkeeper"
            ),
            "striker_policy_hash": cell["striker_policy_hash"],
            "goalkeeper_policy_hash": cell["goalkeeper_policy_hash"],
            "failure": (
                "SAFE_SHOT_SAVED_OR_MISSED"
                if cell["stage"] == "STRIKER_BEST_RESPONSE"
                else "SAFE_GOAL_CONCEDED"
            ),
            "trajectory_digest": cell["trajectory_digest"],
        }
        for cell in cells
        if (
            cell["stage"] == "STRIKER_BEST_RESPONSE"
            and not cell["striker_eligible"]
        )
        or (
            cell["stage"] == "GOALKEEPER_BEST_RESPONSE"
            and not cell["goalkeeper_eligible"]
        )
    ]
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.shot_save_growth_round.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "backend_commit": qualification.backend_commit,
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "physics_backend": "single_shared_world_three_g1",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "cells": cells,
        "strict_replay_cells": replay_cells,
        "strict_replay": strict_replay,
        "selected_striker_policy_id": selected_striker.policy_id,
        "selected_striker_policy_hash": selected_striker.policy_hash,
        "selected_goalkeeper_policy_id": selected_goalkeeper.policy_id,
        "selected_goalkeeper_policy_hash": selected_goalkeeper.policy_hash,
        "attacker_best_response_succeeded": bool(selected_attack_cell["striker_eligible"]),
        "defender_best_response_succeeded": bool(selected_defense_cell["goalkeeper_eligible"]),
        "alternating_best_response_cycle_complete": cycle_complete,
        "failure_memory": failure_memory,
        "trajectory_archives": trajectory_contracts,
        "promotion_eligible": False,
        "promotion_authority": "NONE_DEVELOPMENT_CYCLE_ONLY",
        "sealed_holdout_used": False,
        "pixels_used_for_scoring": False,
        "state_reset_after_start": False,
        "simultaneous_three_body_physics": True,
        "shared_ball_state": True,
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    report_path = output / "shot-save-growth-round.json"
    _atomic_json(report_path, report)
    return validate_shot_save_growth_round(report_path)


def validate_shot_save_growth_round(path: Path) -> dict[str, Any]:
    """Validate report integrity and all development-only authority bounds."""

    report_path = path.expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("shot-save growth report must be a JSON object")
    expected = payload.pop("report_hash", None)
    if expected != hash_json(payload):
        raise ValueError("shot-save growth report integrity mismatch")
    archives = payload.get("trajectory_archives")
    if not isinstance(archives, dict) or set(archives) != {
        "attacker-best-response",
        "defender-best-response",
    }:
        raise ValueError("shot-save growth trajectory contract is incomplete")
    for contract in archives.values():
        if not isinstance(contract, dict) or not isinstance(contract.get("path"), str):
            raise ValueError("shot-save growth trajectory contract is invalid")
        trajectory_path = report_path.parent / contract["path"]
        if (
            not trajectory_path.is_file()
            or contract.get("hash") != hash_bytes(trajectory_path.read_bytes())
            or contract.get("contains_scored_physics_state") is not True
        ):
            raise ValueError("shot-save growth trajectory integrity mismatch")
    if (
        payload.get("schema_version") != "rosclaw_soccer.shot_save_growth_round.v1"
        or payload.get("implementation_hash") != _implementation_hash()
        or payload.get("physics_authority") != "CPU_MUJOCO"
        or payload.get("strict_replay") is not True
        or payload.get("alternating_best_response_cycle_complete") is not True
        or payload.get("promotion_eligible") is not False
        or payload.get("sealed_holdout_used") is not False
        or payload.get("pixels_used_for_scoring") is not False
        or payload.get("state_reset_after_start") is not False
        or payload.get("simultaneous_three_body_physics") is not True
        or payload.get("shared_ball_state") is not True
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_authorized") is not False
        or payload.get("hardware_command_sent") is not False
    ):
        raise ValueError("shot-save growth report authority contract is invalid")
    payload["report_hash"] = expected
    return cast(dict[str, Any], payload)


__all__ = [
    "ShotSaveGoalkeeperPolicy",
    "ShotSaveLeagueConfig",
    "ShotSaveStrikerPolicy",
    "run_shot_save_growth_round",
    "validate_shot_save_growth_round",
]
