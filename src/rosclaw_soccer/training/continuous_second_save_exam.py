"""Strict continuous-world second-save exam for the recovered G1 goalkeeper.

The curriculum reuses the one physical ball after the qualified pass, strike
and airborne glove save.  A bounded SIM-only ball cannon creates a second
causal threat only after measured goalkeeper readiness.  It does not reset
the clock, robot, ball, qpos or qvel, and it is never represented as a second
G1 strike.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
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
from rosclaw_soccer.skills.team.shared_world import (
    G1GoalkeeperConfig,
    G1SecondThreatConfig,
    G1SharedWorldResult,
    simulate_shared_world,
)
from rosclaw_soccer.training.dynamic_corner_save import (
    DynamicCornerSaveLane,
    expanded_dynamic_corner_lanes,
)
from rosclaw_soccer.training.dynamic_takeoff_exam import evaluate_dynamic_takeoff_save
from rosclaw_soccer.training.recovery_athlete_cpu_exam import (
    validate_recovery_athlete_cpu_exam,
)
from rosclaw_soccer.training.recovery_athlete_integration_exam import (
    RecoveryAthleteIntegrationConfig,
    _lane_kwargs,
    _prefix_digest,
)
from rosclaw_soccer.training.save_to_ready_successor import (
    SaveToReadySuccessorConfig,
    _ready_window,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec

_CLAIM = "CONTINUOUS_LIVE_BALL_CANNON_SECOND_GLOVE_SAVE_AND_READY"
_KNOWN_LANES = frozenset({"left-inner", "left-outer", "right-inner", "right-outer"})


@dataclass(frozen=True)
class ContinuousSecondSaveExamConfig:
    """Fail-closed physics gates for the first continuous second-save lane."""

    lane_ids: tuple[str, ...] = ("left-inner",)
    simulation_duration_sec: float = 25.0
    threat: G1SecondThreatConfig = G1SecondThreatConfig()
    ready: SaveToReadySuccessorConfig = SaveToReadySuccessorConfig()
    minimum_second_glove_height_m: float = 1.20
    maximum_glove_surface_penetration_m: float = 0.02
    maximum_glove_surface_separation_m: float = 0.01
    minimum_post_launch_speed_gain_mps: float = 4.0
    maximum_rearm_ball_step_m: float = 0.08
    maximum_rearm_keeper_step_m: float = 0.04
    maximum_rearm_joint_step_rad: float = 0.12
    maximum_launch_ball_step_m: float = 0.08
    minimum_post_second_pelvis_height_m: float = 0.64
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    commercial_use_allowed: bool = False
    schema_version: str = "rosclaw_soccer.continuous_second_save_exam_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.simulation_duration_sec,
            self.minimum_second_glove_height_m,
            self.maximum_glove_surface_penetration_m,
            self.maximum_glove_surface_separation_m,
            self.minimum_post_launch_speed_gain_mps,
            self.maximum_rearm_ball_step_m,
            self.maximum_rearm_keeper_step_m,
            self.maximum_rearm_joint_step_rad,
            self.maximum_launch_ball_step_m,
            self.minimum_post_second_pelvis_height_m,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("continuous second-save settings must be finite")
        if not self.lane_ids or len(set(self.lane_ids)) != len(self.lane_ids):
            raise ValueError("continuous second-save lanes must be unique and non-empty")
        if not set(self.lane_ids) <= _KNOWN_LANES:
            raise ValueError("continuous second-save lane is unknown")
        if not 23.0 <= self.simulation_duration_sec <= 25.0:
            raise ValueError("continuous second-save duration is invalid")
        if self.simulation_duration_sec < self.threat.launch_time_sec + 5.0:
            raise ValueError("continuous second-save recovery window is incomplete")
        if not 1.10 <= self.minimum_second_glove_height_m <= 1.50:
            raise ValueError("continuous second-save glove-height gate is invalid")
        if not 0.005 <= self.maximum_glove_surface_penetration_m <= 0.03:
            raise ValueError("continuous second-save penetration gate is invalid")
        if not 0.002 <= self.maximum_glove_surface_separation_m <= 0.02:
            raise ValueError("continuous second-save separation gate is invalid")
        if not 2.0 <= self.minimum_post_launch_speed_gain_mps <= 8.0:
            raise ValueError("continuous second-save launch-speed gate is invalid")
        if not 0.02 <= self.maximum_rearm_ball_step_m <= 0.10:
            raise ValueError("continuous second-save rearm ball-step gate is invalid")
        if not 0.01 <= self.maximum_rearm_keeper_step_m <= 0.08:
            raise ValueError("continuous second-save rearm keeper-step gate is invalid")
        if not 0.05 <= self.maximum_rearm_joint_step_rad <= 0.20:
            raise ValueError("continuous second-save rearm joint-step gate is invalid")
        if not 0.02 <= self.maximum_launch_ball_step_m <= 0.10:
            raise ValueError("continuous second-save launch ball-step gate is invalid")
        if not 0.60 <= self.minimum_post_second_pelvis_height_m <= 0.72:
            raise ValueError("continuous second-save pelvis-height gate is invalid")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.commercial_use_allowed
        ):
            raise ValueError("continuous second-save exam must remain non-commercial SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _git_head(checkout: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_trajectory(path: Path, trajectory: dict[str, NDArray[Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **trajectory)  # type: ignore[arg-type]
    os.replace(temporary, path)


def _second_save_kwargs(
    *,
    lane: DynamicCornerSaveLane,
    assets: dict[str, Path],
    recovery_checkpoint: Path,
    recovery_exam: Path,
    config: ContinuousSecondSaveExamConfig,
) -> tuple[dict[str, Any], G1GoalkeeperConfig, G1TrainingGoalSpec]:
    kwargs, _, goal, _ = _lane_kwargs(
        lane=lane,
        asset_paths=assets,
        recovery_checkpoint_path=recovery_checkpoint,
        recovery_exam_path=recovery_exam,
        config=RecoveryAthleteIntegrationConfig(),
    )
    goalkeeper = cast(G1GoalkeeperConfig, kwargs["goalkeeper_config"])
    goalkeeper = replace(
        goalkeeper,
        successor_lateral_probe_enabled=False,
        successor_lateral_probe_command_mps=0.0,
    )
    kwargs["goalkeeper_config"] = goalkeeper
    kwargs["simulation_duration_sec"] = config.simulation_duration_sec
    kwargs["second_threat_config"] = config.threat
    return kwargs, goalkeeper, goal


def _event_index(time: NDArray[np.float64], value: float) -> int:
    return int(np.clip(np.searchsorted(time, value, side="left"), 1, time.size - 1))


def _continuity_metrics(
    *,
    result: G1SharedWorldResult,
    trajectory: dict[str, NDArray[Any]],
) -> dict[str, Any]:
    rearm_time = result.second_threat_rearm_time_sec
    launch_time = result.second_threat_launch_time_sec
    if rearm_time is None or launch_time is None:
        return {"valid": False, "reason": "second-threat event timestamps are absent"}
    time = np.asarray(trajectory["time"], dtype=np.float64)
    ball = np.asarray(trajectory["ball_pose"], dtype=np.float64)[:, :3]
    ball_velocity = np.asarray(trajectory["ball_velocity"], dtype=np.float64)[:, :3]
    pelvis = np.asarray(trajectory["goalkeeper_pelvis_pose"], dtype=np.float64)[:, :3]
    joints = np.asarray(trajectory["goalkeeper_joint_position"], dtype=np.float64)
    force = np.asarray(trajectory["second_threat_launcher_force"], dtype=np.float64)
    observed_flight = np.asarray(trajectory["goalkeeper_observed_flight_active"], dtype=np.bool_)
    flight_start = np.asarray(trajectory["goalkeeper_observed_flight_start_sec"], dtype=np.float64)
    reaction = np.asarray(trajectory["goalkeeper_reaction_active"], dtype=np.bool_)
    epochs = np.asarray(trajectory["goalkeeper_contact_epoch"], dtype=np.int64)
    if (
        time.ndim != 1
        or time.size < 100
        or ball.shape != (time.size, 3)
        or ball_velocity.shape != (time.size, 3)
        or pelvis.shape != (time.size, 3)
        or joints.shape != (time.size, 29)
        or force.shape != (time.size, 3)
        or observed_flight.shape != time.shape
        or flight_start.shape != time.shape
        or reaction.shape != time.shape
        or epochs.shape != time.shape
        or not all(
            np.all(np.isfinite(array))
            for array in (time, ball, ball_velocity, pelvis, joints, force)
        )
    ):
        return {"valid": False, "reason": "continuous second-save telemetry is invalid"}
    rearm = _event_index(time, rearm_time)
    launch = _event_index(time, launch_time)
    contact = (
        time.size - 1
        if result.goalkeeper_second_glove_contact_time_sec is None
        else _event_index(time, result.goalkeeper_second_glove_contact_time_sec)
    )
    post_launch = (time >= launch_time) & (time <= launch_time + 0.20)
    pre_speed = float(np.linalg.norm(ball_velocity[max(0, launch - 1)]))
    post_speed = float(np.max(np.linalg.norm(ball_velocity[post_launch], axis=1)))
    causal_window = slice(launch, min(contact + 1, time.size))
    causal_starts = flight_start[causal_window]
    finite_causal_starts = causal_starts[np.isfinite(causal_starts)]
    return {
        "valid": True,
        "control_dt_max_error_sec": float(np.max(np.abs(np.diff(time) - 0.02))),
        "rearm_ball_position_step_m": float(np.linalg.norm(ball[rearm] - ball[rearm - 1])),
        "rearm_keeper_position_step_m": float(np.linalg.norm(pelvis[rearm] - pelvis[rearm - 1])),
        "rearm_joint_peak_step_rad": float(np.max(np.abs(joints[rearm] - joints[rearm - 1]))),
        "launch_ball_position_step_m": float(np.linalg.norm(ball[launch] - ball[launch - 1])),
        "launcher_active_frame_count": int(np.count_nonzero(np.linalg.norm(force, axis=1) > 0.0)),
        "launcher_telemetry_peak_force_n": float(np.max(np.linalg.norm(force, axis=1))),
        "post_launch_speed_gain_mps": post_speed - pre_speed,
        "flight_epoch_clear_at_launch": not bool(observed_flight[launch]),
        "new_flight_epoch_observed": bool(np.any(observed_flight[causal_window])),
        "new_flight_start_time_sec": (
            None if finite_causal_starts.size == 0 else float(np.min(finite_causal_starts))
        ),
        "causal_reaction_observed": bool(np.any(reaction[causal_window])),
        "maximum_contact_epoch": int(np.max(epochs)),
    }


def evaluate_continuous_second_save(
    *,
    result: G1SharedWorldResult,
    trajectory: dict[str, NDArray[Any]],
    lane: DynamicCornerSaveLane,
    goal: G1TrainingGoalSpec,
    goalkeeper: G1GoalkeeperConfig,
    config: ContinuousSecondSaveExamConfig,
) -> dict[str, Any]:
    """Evaluate one uninterrupted first-save, second-save and ready chain."""

    required = {
        "time",
        "ball_pose",
        "ball_velocity",
        "goalkeeper_pelvis_pose",
        "goalkeeper_root_velocity",
        "goalkeeper_torso_quaternion",
        "goalkeeper_foot_contact",
        "goalkeeper_joint_position",
        "goalkeeper_policy_action",
        "goalkeeper_command_mps",
        "goalkeeper_observed_flight_active",
        "goalkeeper_observed_flight_start_sec",
        "goalkeeper_reaction_active",
        "goalkeeper_contact_epoch",
        "second_threat_launcher_force",
    }
    if not required <= set(trajectory):
        return {"passed": False, "reason": "continuous second-save telemetry is incomplete"}
    # The inherited exam predates multi-threat rollouts and reads the final
    # global goal flag.  Score its frozen prefix with the first-save outcome
    # already retained by G1SharedWorldResult, never with the second ball's
    # later outcome.
    first_prefix_result = replace(
        result,
        goal_crossed=not result.goalkeeper_save_observed,
        goal_plane_crossed=not result.goalkeeper_save_observed,
    )
    first_takeoff = evaluate_dynamic_takeoff_save(
        result=first_prefix_result,
        trajectory=trajectory,
        config=lane.takeoff_config,
    )
    continuity = _continuity_metrics(result=result, trajectory=trajectory)
    time = np.asarray(trajectory["time"], dtype=np.float64)
    final_ready_mask = time >= float(time[-1]) - config.ready.ready_hold_sec + 1.0e-9
    final_ready = _ready_window(
        trajectory=trajectory,
        mask=final_ready_mask,
        goal=goal,
        depth_from_goal_line_m=goalkeeper.depth_from_goal_line_m,
        config=config.ready,
    )
    second_contact = result.goalkeeper_second_glove_contact_time_sec
    post_second_height: float | None = None
    if second_contact is not None:
        post_second = time >= second_contact
        post_second_height = float(
            np.min(np.asarray(trajectory["goalkeeper_pelvis_pose"])[post_second, 2])
        )
    surface = result.goalkeeper_second_glove_contact_surface_distance_m
    gates = {
        "qualified_first_airborne_save": first_takeoff.get("passed") is True,
        "ready_rearm_observed": bool(
            result.second_threat_rearmed
            and result.second_threat_rearm_time_sec is not None
            and result.second_threat_launch_time_sec is not None
            and result.second_threat_rearm_time_sec < result.second_threat_launch_time_sec
        ),
        "bounded_physical_launcher": bool(
            result.second_threat_launch_observed
            and 0.0 < result.second_threat_peak_force_n <= config.threat.maximum_force_n
            and continuity.get("launcher_active_frame_count", 0) >= 3
            and continuity.get("post_launch_speed_gain_mps", -math.inf)
            >= config.minimum_post_launch_speed_gain_mps
        ),
        "forward_field_side_second_threat": bool(
            result.second_threat_launch_position_m is not None
            and result.second_threat_target_velocity_mps is not None
            and result.second_threat_launch_position_m[0] + goal.ball_radius_m
            < goal.plane_x_m - config.threat.target_depth_before_goal_m
            and result.second_threat_target_velocity_mps[0] > 0.0
        ),
        "new_causal_threat_epoch": bool(
            continuity.get("flight_epoch_clear_at_launch") is True
            and continuity.get("new_flight_epoch_observed") is True
            and isinstance(continuity.get("new_flight_start_time_sec"), int | float)
            and continuity["new_flight_start_time_sec"] >= config.threat.launch_time_sec - 1.0e-9
            and continuity.get("causal_reaction_observed") is True
            and continuity.get("maximum_contact_epoch") == 2
        ),
        "no_rearm_or_launch_position_jump": bool(
            continuity.get("valid") is True
            and continuity.get("control_dt_max_error_sec", math.inf) <= 1.0e-9
            and continuity.get("rearm_ball_position_step_m", math.inf)
            <= config.maximum_rearm_ball_step_m
            and continuity.get("rearm_keeper_position_step_m", math.inf)
            <= config.maximum_rearm_keeper_step_m
            and continuity.get("rearm_joint_peak_step_rad", math.inf)
            <= config.maximum_rearm_joint_step_rad
            and continuity.get("launch_ball_position_step_m", math.inf)
            <= config.maximum_launch_ball_step_m
        ),
        "ordered_second_anatomical_glove_contact": bool(
            second_contact is not None
            and result.second_threat_launch_time_sec is not None
            and result.second_threat_launch_time_sec < second_contact
            and result.goalkeeper_second_ball_contact_observed
            and result.goalkeeper_second_glove_contact_observed
            and result.goalkeeper_second_glove_contact_side in {"left", "right", "both"}
        ),
        "collision_faithful_second_glove": bool(
            surface is not None
            and math.isfinite(surface)
            and -config.maximum_glove_surface_penetration_m
            <= surface
            <= config.maximum_glove_surface_separation_m
        ),
        "high_second_glove_contact": bool(
            result.goalkeeper_second_glove_contact_height_m is not None
            and result.goalkeeper_second_glove_contact_height_m
            >= config.minimum_second_glove_height_m
        ),
        "physical_second_save": bool(
            result.goalkeeper_second_save_observed and not result.goal_crossed
        ),
        "post_second_stability": bool(
            post_second_height is not None
            and math.isfinite(post_second_height)
            and post_second_height >= config.minimum_post_second_pelvis_height_m
            and result.finite_state
            and not result.goalkeeper_joint_limit_violation
            and not result.torque_limit_violation
        ),
        "second_save_to_ready": final_ready.get("passed") is True,
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "first_takeoff_exam": first_takeoff,
        "continuity": continuity,
        "post_second_minimum_pelvis_height_m": post_second_height,
        "final_ready": final_ready,
        "result": result.to_dict(),
        "trajectory_digest": trajectory_digest(trajectory),
    }


def run_continuous_second_save_exam(
    *,
    asset_root: Path,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    dive_source_checkout: Path,
    dive_athlete_checkpoint_path: Path,
    dive_athlete_exam_path: Path,
    recovery_athlete_checkpoint_path: Path,
    recovery_athlete_exam_path: Path,
    output_dir: Path,
    source_checkout: Path,
    config: ContinuousSecondSaveExamConfig | None = None,
) -> dict[str, Any]:
    """Run parent/candidate physics and strict candidate replay per lane."""

    active = config or ContinuousSecondSaveExamConfig()
    checkout = source_checkout.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("continuous second-save evidence must use a new external directory")
    assets = {
        "striker_actor": striker_actor_path.expanduser().resolve(),
        "goalkeeper_actor": goalkeeper_actor_path.expanduser().resolve(),
        "gmt_model": gmt_model_path.expanduser().resolve(),
        "gmt_skill": gmt_skill_path.expanduser().resolve(),
        "dive_source": dive_source_checkout.expanduser().resolve(),
        "dive_athlete_checkpoint": dive_athlete_checkpoint_path.expanduser().resolve(),
        "dive_athlete_exam": dive_athlete_exam_path.expanduser().resolve(),
    }
    recovery_checkpoint = recovery_athlete_checkpoint_path.expanduser().resolve()
    recovery_exam = recovery_athlete_exam_path.expanduser().resolve()
    files = tuple(value for key, value in assets.items() if key != "dive_source") + (
        recovery_checkpoint,
        recovery_exam,
    )
    if not all(path.is_file() for path in files) or not (assets["dive_source"] / ".git").exists():
        raise FileNotFoundError("continuous second-save input artifact is missing")
    cpu_exam = validate_recovery_athlete_cpu_exam(recovery_exam)
    recovery_hash = hash_bytes(recovery_checkpoint.read_bytes())
    if cpu_exam.get("checkpoint_hash") != recovery_hash:
        raise ValueError("continuous second-save recovery checkpoint binding changed")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    destination.mkdir(parents=True)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.continuous_second_save_request.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "source_commit": _git_head(checkout),
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "recovery_checkpoint_hash": recovery_hash,
        "recovery_cpu_exam_hash": cpu_exam["report_hash"],
        "artifacts": {
            key: (_git_head(value) if key == "dive_source" else hash_bytes(value.read_bytes()))
            for key, value in assets.items()
        },
        "physics_backend": "mujoco_cpu",
        "launcher_identity": "SIM_ONLY_BOUNDED_BALL_CANNON_NOT_A_G1_STRIKE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    request["request_hash"] = hash_json(request)
    _atomic_json(destination / "request.json", request)

    lanes = {lane.lane_id: lane for lane in expanded_dynamic_corner_lanes()}
    cases: dict[str, Any] = {}
    for lane_id in active.lane_ids:
        lane = lanes[lane_id]
        kwargs, goalkeeper, goal = _second_save_kwargs(
            lane=lane,
            assets=assets,
            recovery_checkpoint=recovery_checkpoint,
            recovery_exam=recovery_exam,
            config=active,
        )
        parent_kwargs = dict(kwargs)
        parent_kwargs["second_threat_config"] = None
        parent_result, parent_trajectory = simulate_shared_world(asset_root, **parent_kwargs)
        parent_path = destination / f"{lane_id}-parent-trajectory.npz"
        _atomic_trajectory(parent_path, parent_trajectory)
        try:
            result, trajectory = simulate_shared_world(asset_root, **kwargs)
            replay_result, replay_trajectory = simulate_shared_world(asset_root, **kwargs)
        except RuntimeError as exc:
            cases[lane_id] = {
                "passed": False,
                "gates": {"simulation_completed": False},
                "evaluation": {
                    "passed": False,
                    "reason": "continuous second-save simulation rejected",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                },
                "strict_replay": False,
                "parent_trajectory_file": parent_path.name,
                "parent_trajectory_hash": hash_bytes(parent_path.read_bytes()),
                "trajectory_file": None,
                "trajectory_hash": None,
            }
            continue
        evaluation = evaluate_continuous_second_save(
            result=result,
            trajectory=trajectory,
            lane=lane,
            goal=goal,
            goalkeeper=goalkeeper,
            config=active,
        )
        replay = evaluate_continuous_second_save(
            result=replay_result,
            trajectory=replay_trajectory,
            lane=lane,
            goal=goal,
            goalkeeper=goalkeeper,
            config=active,
        )
        first_contact = result.goalkeeper_ball_contact_time_sec
        prefix_unchanged = bool(
            first_contact is not None
            and parent_result.goalkeeper_ball_contact_time_sec == first_contact
            and _prefix_digest(parent_trajectory, contact_time_sec=first_contact)
            == _prefix_digest(trajectory, contact_time_sec=first_contact)
        )
        strict_replay = bool(
            result.to_dict() == replay_result.to_dict()
            and trajectory_digest(trajectory) == trajectory_digest(replay_trajectory)
            and replay.get("passed") is True
        )
        lane_gates = {
            "first_save_prefix_unchanged": prefix_unchanged,
            "continuous_candidate_passed": evaluation.get("passed") is True,
            "strict_replay": strict_replay,
        }
        trajectory_path = destination / f"{lane_id}-continuous-trajectory.npz"
        _atomic_trajectory(trajectory_path, trajectory)
        cases[lane_id] = {
            "passed": bool(all(lane_gates.values())),
            "gates": lane_gates,
            "evaluation": evaluation,
            "strict_replay": strict_replay,
            "parent_trajectory_file": parent_path.name,
            "parent_trajectory_hash": hash_bytes(parent_path.read_bytes()),
            "trajectory_file": trajectory_path.name,
            "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
        }

    portfolio_gates = {
        "all_requested_lanes_pass": set(cases) == set(active.lane_ids)
        and all(case["passed"] for case in cases.values()),
        "at_least_one_continuous_second_save": any(case["passed"] for case in cases.values()),
    }
    passed = bool(all(portfolio_gates.values()))
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.continuous_second_save_exam.v1",
        "passed": passed,
        "promotion_status": (
            "PROMOTED_SIM_ONLY_CONTINUOUS_SECOND_SAVE_CURRICULUM"
            if passed
            else "REJECTED_DEVELOPMENT"
        ),
        "claim": _CLAIM,
        "request_hash": request["request_hash"],
        "source_commit": request["source_commit"],
        "recovery_checkpoint_hash": recovery_hash,
        "recovery_cpu_exam_hash": cpu_exam["report_hash"],
        "portfolio_gates": portfolio_gates,
        "cases": cases,
        "physics_backend": "mujoco_cpu",
        "pixels_used_for_scoring": False,
        "reset_or_teleport_used": False,
        "second_striker_claimed": False,
        "launcher_identity": "SIM_ONLY_BOUNDED_BALL_CANNON_NOT_A_G1_STRIKE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "implementation_hash": _implementation_hash(),
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(destination / "evidence.json", report)
    return report


def validate_continuous_second_save_exam(path: Path) -> dict[str, Any]:
    """Validate integrity, current implementation and all physics gates."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("continuous second-save evidence must be an object")
    unhashed = dict(payload)
    claimed = unhashed.pop("report_hash", None)
    gates = payload.get("portfolio_gates")
    cases = payload.get("cases")
    if not (
        claimed == hash_json(unhashed)
        and payload.get("schema_version") == "rosclaw_soccer.continuous_second_save_exam.v1"
        and payload.get("passed") is True
        and payload.get("promotion_status") == "PROMOTED_SIM_ONLY_CONTINUOUS_SECOND_SAVE_CURRICULUM"
        and payload.get("claim") == _CLAIM
        and payload.get("physics_backend") == "mujoco_cpu"
        and payload.get("pixels_used_for_scoring") is False
        and payload.get("reset_or_teleport_used") is False
        and payload.get("second_striker_claimed") is False
        and payload.get("launcher_identity") == "SIM_ONLY_BOUNDED_BALL_CANNON_NOT_A_G1_STRIKE"
        and payload.get("activation_ceiling") == "SIM_ONLY"
        and payload.get("hardware_command_sent") is False
        and payload.get("commercial_use_allowed") is False
        and payload.get("implementation_hash") == _implementation_hash()
        and isinstance(gates, dict)
        and gates
        and all(gates.values())
        and isinstance(cases, dict)
        and cases
        and all(
            isinstance(case, dict)
            and case.get("passed") is True
            and case.get("strict_replay") is True
            for case in cases.values()
        )
    ):
        raise ValueError("continuous second-save evidence failed closed")
    return cast(dict[str, Any], payload)


def _implementation_hash() -> str:
    package = Path(__file__).parents[1]
    files = (
        Path(__file__),
        package / "skills" / "team" / "shared_world.py",
        package / "skills" / "goalkeeper_v2" / "observations.py",
        Path(__file__).parent / "recovery_athlete_student.py",
        Path(__file__).parent / "save_to_ready_successor.py",
    )
    return str(hash_json({path.name: hash_bytes(path.read_bytes()) for path in files}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--striker-actor", type=Path, required=True)
    parser.add_argument("--goalkeeper-actor", type=Path, required=True)
    parser.add_argument("--gmt-model", type=Path, required=True)
    parser.add_argument("--gmt-skill", type=Path, required=True)
    parser.add_argument("--dive-source", type=Path, required=True)
    parser.add_argument("--dive-checkpoint", type=Path, required=True)
    parser.add_argument("--dive-exam", type=Path, required=True)
    parser.add_argument("--recovery-checkpoint", type=Path, required=True)
    parser.add_argument("--recovery-exam", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_continuous_second_save_exam(
        asset_root=args.asset_root,
        striker_actor_path=args.striker_actor,
        goalkeeper_actor_path=args.goalkeeper_actor,
        gmt_model_path=args.gmt_model,
        gmt_skill_path=args.gmt_skill,
        dive_source_checkout=args.dive_source,
        dive_athlete_checkpoint_path=args.dive_checkpoint,
        dive_athlete_exam_path=args.dive_exam,
        recovery_athlete_checkpoint_path=args.recovery_checkpoint,
        recovery_athlete_exam_path=args.recovery_exam,
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
