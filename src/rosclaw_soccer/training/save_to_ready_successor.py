"""Strict save-to-ready successor exam in the shared three-G1 world.

The exam keeps the qualified pass, shot and airborne glove save intact, then
requires measured absorption, a new lateral command and a second continuous
goalkeeper-ready state.  It never resets or teleports the goalkeeper and
grants no authority outside CPU MuJoCo simulation.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import (
    G1SharedWorldResult,
    simulate_shared_world,
)
from rosclaw_soccer.training.dynamic_corner_save import (
    dynamic_corner_lane_kwargs,
    expanded_dynamic_corner_lanes,
    validate_dynamic_corner_evidence,
)
from rosclaw_soccer.training.dynamic_takeoff_exam import evaluate_dynamic_takeoff_save
from rosclaw_soccer.world.field import G1TrainingGoalSpec

_CLAIM = "STRICT_SAVE_ABSORB_REENGAGE_GOALKEEPER_READY_PORTFOLIO"
_AUTHORITY_BY_LANE = {
    "left-inner": 0.05,
    "left-outer": 0.75,
    "right-inner": 0.50,
    "right-outer": 0.75,
}


@dataclass(frozen=True)
class SaveToReadySuccessorConfig:
    """Fail-closed physical gates for recovery and successor re-engagement."""

    simulation_duration_sec: float = 25.0
    recovery_delay_sec: float = 2.0
    recovery_depth_speed_mps: float = 0.30
    recovery_lateral_deadband_m: float = 0.15
    probe_delay_sec: float = 10.0
    probe_duration_sec: float = 0.80
    probe_speed_mps: float = 0.14
    ready_hold_sec: float = 1.0
    minimum_pelvis_height_m: float = 0.70
    minimum_upright_projection: float = 0.90
    maximum_root_linear_speed_mps: float = 0.25
    maximum_root_angular_speed_rad_s: float = 0.50
    minimum_facing_field_cos: float = 0.90
    maximum_hand_ready_error_rad: float = 0.18
    maximum_depth_error_m: float = 0.40
    minimum_probe_displacement_m: float = 0.01
    minimum_probe_lateral_speed_mps: float = 0.03
    minimum_lateral_acceleration_capacity_mps2: float = 0.50
    minimum_probe_command_fraction: float = 0.90
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.save_to_ready_successor_config.v1"

    def __post_init__(self) -> None:
        values = tuple(
            value
            for value in asdict(self).values()
            if isinstance(value, int | float) and not isinstance(value, bool)
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("save-to-ready successor configuration must be finite")
        if not 24.0 <= self.simulation_duration_sec <= 25.0:
            raise ValueError("save-to-ready successor duration is invalid")
        if not 2.0 <= self.recovery_delay_sec <= 3.0:
            raise ValueError("save-to-ready recovery delay is invalid")
        if not 0.20 <= self.recovery_depth_speed_mps <= 0.30:
            raise ValueError("save-to-ready depth speed is invalid")
        if not 0.08 <= self.recovery_lateral_deadband_m <= 0.20:
            raise ValueError("save-to-ready lateral deadband is invalid")
        if not 8.0 <= self.probe_delay_sec <= 10.0:
            raise ValueError("save-to-ready probe delay is invalid")
        if not 0.60 <= self.probe_duration_sec <= 1.0:
            raise ValueError("save-to-ready probe duration is invalid")
        if not 0.10 <= self.probe_speed_mps <= 0.20:
            raise ValueError("save-to-ready probe speed is invalid")
        if not math.isclose(self.ready_hold_sec, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("goalkeeper readiness must be held for exactly one second")
        if not (
            0.65 <= self.minimum_pelvis_height_m <= 0.75
            and 0.85 <= self.minimum_upright_projection <= 1.0
            and 0.10 <= self.maximum_root_linear_speed_mps <= 0.25
            and 0.20 <= self.maximum_root_angular_speed_rad_s <= 0.50
            and 0.85 <= self.minimum_facing_field_cos <= 1.0
            and 0.05 <= self.maximum_hand_ready_error_rad <= 0.18
            and 0.20 <= self.maximum_depth_error_m <= 0.40
            and 0.005 <= self.minimum_probe_displacement_m <= 0.05
            and 0.02 <= self.minimum_probe_lateral_speed_mps <= 0.10
            and 0.50 <= self.minimum_lateral_acceleration_capacity_mps2 <= 2.0
            and 0.80 <= self.minimum_probe_command_fraction <= 1.0
        ):
            raise ValueError("save-to-ready successor thresholds are invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("save-to-ready successor exam must remain SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _ready_window(
    *,
    trajectory: dict[str, np.ndarray],
    mask: np.ndarray,
    goal: G1TrainingGoalSpec,
    depth_from_goal_line_m: float,
    config: SaveToReadySuccessorConfig,
) -> dict[str, Any]:
    pelvis = np.asarray(trajectory["goalkeeper_pelvis_pose"], dtype=np.float64)[mask]
    velocity = np.asarray(trajectory["goalkeeper_root_velocity"], dtype=np.float64)[mask]
    quaternion = np.asarray(trajectory["goalkeeper_torso_quaternion"], dtype=np.float64)[mask]
    support = np.asarray(trajectory["goalkeeper_foot_contact"], dtype=np.bool_)[mask]
    joints = np.asarray(trajectory["goalkeeper_joint_position"], dtype=np.float64)[mask]
    target = np.asarray(trajectory["goalkeeper_policy_action"], dtype=np.float64)[mask]
    command = np.asarray(trajectory["goalkeeper_command_mps"], dtype=np.float64)[mask]
    upright = 1.0 - 2.0 * (quaternion[:, 1] ** 2 + quaternion[:, 2] ** 2)
    facing = -(1.0 - 2.0 * (quaternion[:, 2] ** 2 + quaternion[:, 3] ** 2))
    linear_speed = np.linalg.norm(velocity[:, :3], axis=1)
    angular_speed = np.linalg.norm(velocity[:, 3:], axis=1)
    hand_error = np.sqrt(np.mean((joints[:, 12:] - target[:, 12:]) ** 2, axis=1))
    desired_depth = goal.plane_x_m - depth_from_goal_line_m
    depth_error = np.abs(pelvis[:, 0] - desired_depth)
    inside_region = (depth_error <= config.maximum_depth_error_m) & (
        np.abs(pelvis[:, 1]) <= goal.width_m * 0.5 - 0.20
    )
    metrics = {
        "minimum_pelvis_height_m": float(np.min(pelvis[:, 2])),
        "minimum_upright_projection": float(np.min(upright)),
        "maximum_root_linear_speed_mps": float(np.max(linear_speed)),
        "maximum_root_angular_speed_rad_s": float(np.max(angular_speed)),
        "minimum_facing_field_cos": float(np.min(facing)),
        "maximum_hand_ready_error_rad": float(np.max(hand_error)),
        "maximum_depth_error_m": float(np.max(depth_error)),
        "bilateral_support_fraction": float(np.mean(np.all(support, axis=1))),
        "inside_keeper_region_fraction": float(np.mean(inside_region)),
        "maximum_absolute_command_mps": float(np.max(np.abs(command))),
        "sample_count": int(np.count_nonzero(mask)),
    }
    gates = {
        "pelvis_height": metrics["minimum_pelvis_height_m"] >= config.minimum_pelvis_height_m,
        "upright": metrics["minimum_upright_projection"] >= config.minimum_upright_projection,
        "low_root_linear_speed": metrics["maximum_root_linear_speed_mps"]
        <= config.maximum_root_linear_speed_mps,
        "low_root_angular_speed": metrics["maximum_root_angular_speed_rad_s"]
        <= config.maximum_root_angular_speed_rad_s,
        "bilateral_support": math.isclose(
            metrics["bilateral_support_fraction"], 1.0, rel_tol=0.0, abs_tol=1e-12
        ),
        "facing_field": metrics["minimum_facing_field_cos"] >= config.minimum_facing_field_cos,
        "inside_keeper_region": math.isclose(
            metrics["inside_keeper_region_fraction"], 1.0, rel_tol=0.0, abs_tol=1e-12
        ),
        "hands_ready": metrics["maximum_hand_ready_error_rad"]
        <= config.maximum_hand_ready_error_rad,
    }
    return {"passed": bool(all(gates.values())), "gates": gates, "metrics": metrics}


def evaluate_save_to_ready_successor(
    *,
    result: G1SharedWorldResult,
    trajectory: dict[str, np.ndarray],
    goal: G1TrainingGoalSpec,
    depth_from_goal_line_m: float,
    expected_probe_command_mps: float,
    config: SaveToReadySuccessorConfig | None = None,
) -> dict[str, Any]:
    """Evaluate absorption, re-engagement and the next goalkeeper-ready state."""

    active = config or SaveToReadySuccessorConfig()
    required = {
        "time",
        "goalkeeper_pelvis_pose",
        "goalkeeper_root_velocity",
        "goalkeeper_torso_quaternion",
        "goalkeeper_foot_contact",
        "goalkeeper_joint_position",
        "goalkeeper_policy_action",
        "goalkeeper_command_mps",
    }
    if not required <= set(trajectory):
        return {"passed": False, "reason": "successor telemetry is incomplete"}
    time = np.asarray(trajectory["time"], dtype=np.float64)
    shapes = {
        "goalkeeper_pelvis_pose": (time.size, 7),
        "goalkeeper_root_velocity": (time.size, 6),
        "goalkeeper_torso_quaternion": (time.size, 4),
        "goalkeeper_foot_contact": (time.size, 2),
        "goalkeeper_joint_position": (time.size, 29),
        "goalkeeper_policy_action": (time.size, 29),
        "goalkeeper_command_mps": (time.size,),
    }
    arrays = {name: np.asarray(trajectory[name]) for name in shapes}
    if (
        time.ndim != 1
        or time.size < 100
        or not np.all(np.isfinite(time))
        or not np.all(np.diff(time) > 0.0)
        or any(array.shape != shapes[name] for name, array in arrays.items())
        or any(
            not np.all(np.isfinite(array))
            for name, array in arrays.items()
            if name != "goalkeeper_foot_contact"
        )
    ):
        return {"passed": False, "reason": "successor telemetry is invalid"}
    contact = result.goalkeeper_ball_contact_time_sec
    if contact is None or not result.goalkeeper_save_observed:
        return {"passed": False, "reason": "successful save contact is absent"}
    probe_start = contact + active.probe_delay_sec
    probe_stop = probe_start + active.probe_duration_sec
    pre_mask = (time >= probe_start - active.ready_hold_sec) & (time < probe_start)
    probe_mask = (time >= probe_start) & (time < probe_stop)
    post_mask = time >= float(time[-1]) - active.ready_hold_sec + 1.0e-9
    if (
        min(np.count_nonzero(pre_mask), np.count_nonzero(probe_mask), np.count_nonzero(post_mask))
        < 30
    ):
        return {"passed": False, "reason": "successor windows are incomplete"}
    pre_ready = _ready_window(
        trajectory=trajectory,
        mask=pre_mask,
        goal=goal,
        depth_from_goal_line_m=depth_from_goal_line_m,
        config=active,
    )
    post_ready = _ready_window(
        trajectory=trajectory,
        mask=post_mask,
        goal=goal,
        depth_from_goal_line_m=depth_from_goal_line_m,
        config=active,
    )
    pelvis = np.asarray(trajectory["goalkeeper_pelvis_pose"], dtype=np.float64)
    velocity = np.asarray(trajectory["goalkeeper_root_velocity"], dtype=np.float64)
    command = np.asarray(trajectory["goalkeeper_command_mps"], dtype=np.float64)
    probe_indices = np.flatnonzero(probe_mask)
    probe_displacement = float(pelvis[probe_indices[-1], 1] - pelvis[probe_indices[0], 1])
    probe_lateral_speed = float(np.max(np.abs(velocity[probe_mask, 1])))
    probe_acceleration = float(
        np.max(np.abs(np.diff(velocity[probe_mask, 1]) / np.diff(time[probe_mask])))
    )
    command_fraction = float(
        np.mean(
            np.isclose(
                command[probe_mask],
                expected_probe_command_mps,
                rtol=0.0,
                atol=1e-12,
            )
        )
    )
    probe_metrics = {
        "expected_command_mps": expected_probe_command_mps,
        "command_fraction": command_fraction,
        "signed_displacement_m": probe_displacement,
        "peak_lateral_speed_mps": probe_lateral_speed,
        "peak_lateral_acceleration_mps2": probe_acceleration,
        "start_sec": probe_start,
        "stop_sec": probe_stop,
    }
    probe_gates = {
        "command_fidelity": command_fraction >= active.minimum_probe_command_fraction,
        "command_direction_response": probe_displacement * expected_probe_command_mps > 0.0,
        "minimum_displacement": abs(probe_displacement) >= active.minimum_probe_displacement_m,
        "minimum_lateral_speed": probe_lateral_speed >= active.minimum_probe_lateral_speed_mps,
        "lateral_acceleration_capacity": probe_acceleration
        >= active.minimum_lateral_acceleration_capacity_mps2,
    }
    gates = {
        "save_observed": result.goalkeeper_save_observed,
        "finite_state": result.finite_state,
        "joint_limits_safe": not result.goalkeeper_joint_limit_violation,
        "pre_probe_goalkeeper_ready": pre_ready["passed"],
        "successor_probe_reengaged": all(probe_gates.values()),
        "post_probe_goalkeeper_ready": post_ready["passed"],
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "pre_probe_ready": pre_ready,
        "probe": {
            "passed": bool(all(probe_gates.values())),
            "gates": probe_gates,
            "metrics": probe_metrics,
        },
        "post_probe_ready": post_ready,
        "reset_or_teleport_used": False,
        "successor_state": "GOALKEEPER_READY",
    }


def run_save_to_ready_successor_evidence(
    *,
    asset_root: Path,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    dive_source_checkout: Path,
    dive_athlete_checkpoint_path: Path,
    dive_athlete_exam_path: Path,
    parent_evidence_path: Path,
    output_dir: Path,
    config: SaveToReadySuccessorConfig | None = None,
) -> dict[str, Any]:
    """Run four bilateral save-to-ready cases twice under strict replay."""

    active = config or SaveToReadySuccessorConfig()
    parent_path = parent_evidence_path.expanduser().resolve()
    parent = validate_dynamic_corner_evidence(parent_path)
    parent_authority = parent.get("dive_athlete", {}).get("authority_by_lane")
    if parent_authority != _AUTHORITY_BY_LANE:
        raise ValueError("parent dive-athlete authority table changed")
    paths = (
        striker_actor_path,
        goalkeeper_actor_path,
        gmt_model_path,
        gmt_skill_path,
        dive_athlete_checkpoint_path,
        dive_athlete_exam_path,
    )
    resolved = tuple(path.expanduser().resolve() for path in paths)
    if not all(path.is_file() for path in resolved):
        raise FileNotFoundError("save-to-ready evidence input artifact is missing")
    striker, keeper, gmt_model, gmt_skill, athlete_checkpoint, athlete_exam = resolved
    dive_source = dive_source_checkout.expanduser().resolve()
    if not (dive_source / ".git").exists() or not (dive_source / "LICENSE").is_file():
        raise ValueError("save-to-ready dive source checkout is incomplete")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    checkout = Path(__file__).parents[3]
    artifacts = {
        "striker_actor_hash": hash_bytes(striker.read_bytes()),
        "goalkeeper_actor_hash": hash_bytes(keeper.read_bytes()),
        "gmt_model_hash": hash_bytes(gmt_model.read_bytes()),
        "gmt_skill_hash": hash_bytes(gmt_skill.read_bytes()),
        "dive_athlete_checkpoint_hash": hash_bytes(athlete_checkpoint.read_bytes()),
        "dive_athlete_exam_hash": hash_bytes(athlete_exam.read_bytes()),
        "dive_source_commit": _git_head(dive_source),
        "dive_source_license_hash": hash_bytes((dive_source / "LICENSE").read_bytes()),
        "parent_evidence_file_hash": hash_bytes(parent_path.read_bytes()),
        "parent_report_hash": parent["report_hash"],
        "parent_implementation_hash": parent["implementation_hash"],
    }
    request = {
        "schema_version": "rosclaw_soccer.save_to_ready_successor_request.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "authority_by_lane": _AUTHORITY_BY_LANE,
        "artifacts": artifacts,
        "source_commit": _git_head(checkout),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "request.json", request)
    cases: dict[str, Any] = {}
    for lane in expanded_dynamic_corner_lanes():
        expected_command = active.probe_speed_mps * (
            1.0 if lane.lane_id.startswith("left") else -1.0
        )
        kwargs = dynamic_corner_lane_kwargs(
            lane=lane,
            striker_actor_path=striker,
            goalkeeper_actor_path=keeper,
            gmt_model_path=gmt_model,
            gmt_skill_path=gmt_skill,
            dive_source_checkout=dive_source,
            dive_athlete_checkpoint_path=athlete_checkpoint,
            dive_athlete_exam_path=athlete_exam,
            dive_athlete_blend=_AUTHORITY_BY_LANE[lane.lane_id],
        )
        goalkeeper_config = kwargs["goalkeeper_config"]
        goal = kwargs["goal_spec"]
        kwargs["simulation_duration_sec"] = active.simulation_duration_sec
        kwargs["goalkeeper_config"] = replace(
            goalkeeper_config,
            maximum_depth_correction_mps=active.recovery_depth_speed_mps,
            post_contact_ready_recovery_enabled=True,
            post_contact_ready_recovery_delay_sec=active.recovery_delay_sec,
            post_contact_ready_lateral_deadband_m=active.recovery_lateral_deadband_m,
            successor_lateral_probe_enabled=True,
            successor_lateral_probe_delay_sec=active.probe_delay_sec,
            successor_lateral_probe_duration_sec=active.probe_duration_sec,
            successor_lateral_probe_command_mps=expected_command,
        )
        first_result, first_trajectory = simulate_shared_world(asset_root, **kwargs)
        replay_result, replay_trajectory = simulate_shared_world(asset_root, **kwargs)
        first_takeoff = evaluate_dynamic_takeoff_save(
            result=first_result,
            trajectory=first_trajectory,
            config=lane.takeoff_config,
        )
        replay_takeoff = evaluate_dynamic_takeoff_save(
            result=replay_result,
            trajectory=replay_trajectory,
            config=lane.takeoff_config,
        )
        first_successor = evaluate_save_to_ready_successor(
            result=first_result,
            trajectory=first_trajectory,
            goal=goal,
            depth_from_goal_line_m=goalkeeper_config.depth_from_goal_line_m,
            expected_probe_command_mps=expected_command,
            config=active,
        )
        replay_successor = evaluate_save_to_ready_successor(
            result=replay_result,
            trajectory=replay_trajectory,
            goal=goal,
            depth_from_goal_line_m=goalkeeper_config.depth_from_goal_line_m,
            expected_probe_command_mps=expected_command,
            config=active,
        )
        strict_replay = bool(
            first_result.to_dict() == replay_result.to_dict()
            and trajectory_digest(first_trajectory) == trajectory_digest(replay_trajectory)
        )
        trajectory_path = output / f"{lane.lane_id}-trajectory.npz"
        np.savez_compressed(trajectory_path, **replay_trajectory)  # type: ignore[arg-type]
        cases[lane.lane_id] = {
            "passed": bool(
                first_takeoff.get("passed")
                and replay_takeoff.get("passed")
                and first_successor.get("passed")
                and replay_successor.get("passed")
                and strict_replay
            ),
            "strict_replay": strict_replay,
            "expected_probe_command_mps": expected_command,
            "trajectory_file": trajectory_path.name,
            "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
            "result": replay_result.to_dict(),
            "takeoff": replay_takeoff,
            "successor": replay_successor,
        }
    portfolio_gates = {
        "all_original_airborne_saves_retained": all(
            case["takeoff"].get("passed") is True for case in cases.values()
        ),
        "all_pre_probe_ready": all(
            case["successor"].get("pre_probe_ready", {}).get("passed") is True
            for case in cases.values()
        ),
        "both_probe_directions_exercised": {
            math.copysign(1.0, float(case["expected_probe_command_mps"])) for case in cases.values()
        }
        == {-1.0, 1.0},
        "all_successor_probes_reengaged": all(
            case["successor"].get("probe", {}).get("passed") is True for case in cases.values()
        ),
        "all_post_probe_goalkeeper_ready": all(
            case["successor"].get("post_probe_ready", {}).get("passed") is True
            for case in cases.values()
        ),
        "all_lanes_strict_replay": all(case["strict_replay"] for case in cases.values()),
        "no_reset_or_teleport": all(
            case["successor"].get("reset_or_teleport_used") is False for case in cases.values()
        ),
    }
    passed = bool(all(case["passed"] for case in cases.values()) and all(portfolio_gates.values()))
    report = {
        "schema_version": "rosclaw_soccer.save_to_ready_successor_evidence.v1",
        "passed": passed,
        "promotion_status": "FROZEN_RESEARCH_DEMO" if passed else "REJECTED_DEVELOPMENT",
        "claim": _CLAIM,
        "portfolio_gates": portfolio_gates,
        "case_count": len(cases),
        "cases": cases,
        "successor_contract": {
            "source_skill_id": "soccer.goalkeeper.save",
            "successor_skill_id": "soccer.goalkeeper.ready",
            "hold_steps": 50,
            "control_period_sec": 0.02,
            "new_lateral_command_required": True,
        },
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "pixels_used_for_scoring": False,
        "single_shared_ball_per_case": True,
        "simultaneous_three_body_physics_per_case": True,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "implementation_hash": _implementation_hash(),
        "artifacts": artifacts,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "evidence.json", report)
    return report


def validate_save_to_ready_successor_evidence(path: Path) -> dict[str, Any]:
    """Validate frozen S104 evidence and every external trajectory binding."""

    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("save-to-ready evidence must be an object")
    claimed_hash = payload.get("report_hash")
    unhashed = dict(payload)
    unhashed.pop("report_hash", None)
    request = source.parent / "request.json"
    request_payload = json.loads(request.read_text(encoding="utf-8")) if request.is_file() else None
    cases = payload.get("cases")
    gates = payload.get("portfolio_gates")
    if not (
        claimed_hash == hash_json(unhashed)
        and payload.get("schema_version") == "rosclaw_soccer.save_to_ready_successor_evidence.v1"
        and payload.get("passed") is True
        and payload.get("promotion_status") == "FROZEN_RESEARCH_DEMO"
        and payload.get("claim") == _CLAIM
        and payload.get("physics_authority") == "CPU_MUJOCO"
        and payload.get("activation_ceiling") == "SIM_ONLY"
        and payload.get("hardware_command_sent") is False
        and payload.get("commercial_use_allowed") is False
        and payload.get("pixels_used_for_scoring") is False
        and payload.get("implementation_hash") == _implementation_hash()
        and request.is_file()
        and hash_bytes(request.read_bytes()) == payload.get("request_hash")
        and isinstance(request_payload, dict)
        and payload.get("artifacts") == request_payload.get("artifacts")
        and isinstance(cases, dict)
        and set(cases) == set(_AUTHORITY_BY_LANE)
        and isinstance(gates, dict)
        and all(gates.values())
    ):
        raise ValueError("save-to-ready evidence authority contract is invalid")
    for case in cases.values():
        name = case.get("trajectory_file") if isinstance(case, dict) else None
        trajectory = source.parent / str(name)
        if not (
            isinstance(case, dict)
            and case.get("passed") is True
            and case.get("strict_replay") is True
            and isinstance(name, str)
            and Path(name).name == name
            and trajectory.is_file()
            and hash_bytes(trajectory.read_bytes()) == case.get("trajectory_hash")
            and case.get("successor", {}).get("successor_state") == "GOALKEEPER_READY"
        ):
            raise ValueError("save-to-ready lane binding changed")
    return cast(dict[str, Any], payload)


def _implementation_hash() -> str:
    package = Path(__file__).parents[1]
    return str(
        hash_json(
            {
                "save_to_ready_successor": hash_bytes(Path(__file__).read_bytes()),
                "shared_world": hash_bytes(
                    (package / "skills" / "team" / "shared_world.py").read_bytes()
                ),
                "dynamic_corner_save": hash_bytes(
                    (package / "training" / "dynamic_corner_save.py").read_bytes()
                ),
                "dynamic_takeoff_exam": hash_bytes(
                    (package / "training" / "dynamic_takeoff_exam.py").read_bytes()
                ),
                "recovery_foundation": hash_bytes(
                    (package / "training" / "recovery_foundation.py").read_bytes()
                ),
            }
        )
    )


def _git_head(checkout: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--striker-actor", type=Path, required=True)
    parser.add_argument("--goalkeeper-actor", type=Path, required=True)
    parser.add_argument("--gmt-model", type=Path, required=True)
    parser.add_argument("--gmt-skill", type=Path, required=True)
    parser.add_argument("--dive-source-checkout", type=Path, required=True)
    parser.add_argument("--dive-athlete-checkpoint", type=Path, required=True)
    parser.add_argument("--dive-athlete-exam", type=Path, required=True)
    parser.add_argument("--parent-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_save_to_ready_successor_evidence(
        asset_root=args.asset_root,
        striker_actor_path=args.striker_actor,
        goalkeeper_actor_path=args.goalkeeper_actor,
        gmt_model_path=args.gmt_model,
        gmt_skill_path=args.gmt_skill,
        dive_source_checkout=args.dive_source_checkout,
        dive_athlete_checkpoint_path=args.dive_athlete_checkpoint,
        dive_athlete_exam_path=args.dive_athlete_exam,
        parent_evidence_path=args.parent_evidence,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SaveToReadySuccessorConfig",
    "evaluate_save_to_ready_successor",
    "run_save_to_ready_successor_evidence",
    "validate_save_to_ready_successor_evidence",
]
