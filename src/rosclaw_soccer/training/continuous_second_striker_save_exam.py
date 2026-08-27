"""Strict four-G1, two-ball, two-save continuous-world qualification.

The second threat is created only by the fourth G1's anatomical foot contact
with a second physical football that exists from model compilation onward.
No ball cannon, qpos/qvel write, reset or teleport is permitted after time
zero.  The exam binds the learned contact actor, upper-corner muscle memory,
measured goalkeeper rearm, high glove collision, outward deflection and final
ready state in one uninterrupted MuJoCo clock.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.upper_corner_strike import UpperCornerStrikePolicy
from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import (
    G1GoalkeeperConfig,
    G1PhysicalSecondStrikerConfig,
    G1SharedWorldResult,
    simulate_shared_world,
)
from rosclaw_soccer.training.continuous_second_save_exam import (
    ContinuousSecondSaveExamConfig,
    _second_save_kwargs,
)
from rosclaw_soccer.training.dynamic_corner_save import (
    DynamicCornerSaveLane,
    expanded_dynamic_corner_lanes,
)
from rosclaw_soccer.training.dynamic_takeoff_exam import evaluate_dynamic_takeoff_save
from rosclaw_soccer.training.save_to_ready_successor import (
    SaveToReadySuccessorConfig,
    _ready_window,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec

_CLAIM = "CONTINUOUS_FOUR_G1_SECOND_STRIKER_HIGH_GLOVE_SAVE_AND_READY"
_KNOWN_LANES = frozenset({"left-inner", "left-outer", "right-inner", "right-outer"})


@dataclass(frozen=True)
class ContinuousSecondStrikerSaveExamConfig:
    """Fail-closed gates for a physical second-striker successor."""

    lane_ids: tuple[str, ...] = ("left-inner",)
    simulation_duration_sec: float = 25.0
    striker: G1PhysicalSecondStrikerConfig = G1PhysicalSecondStrikerConfig()
    ready: SaveToReadySuccessorConfig = SaveToReadySuccessorConfig()
    minimum_second_glove_height_m: float = 1.20
    maximum_glove_surface_penetration_m: float = 0.02
    maximum_glove_surface_separation_m: float = 0.01
    maximum_precontact_second_ball_speed_mps: float = 0.05
    minimum_post_glove_outward_speed_mps: float = 0.50
    minimum_post_second_pelvis_height_m: float = 0.64
    maximum_control_dt_error_sec: float = 1.0e-9
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    commercial_use_allowed: bool = False
    schema_version: str = "rosclaw_soccer.continuous_second_striker_save_config.v2"

    def __post_init__(self) -> None:
        values = (
            self.simulation_duration_sec,
            self.minimum_second_glove_height_m,
            self.maximum_glove_surface_penetration_m,
            self.maximum_glove_surface_separation_m,
            self.maximum_precontact_second_ball_speed_mps,
            self.minimum_post_glove_outward_speed_mps,
            self.minimum_post_second_pelvis_height_m,
            self.maximum_control_dt_error_sec,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("continuous second-striker gates must be finite")
        if not self.lane_ids or len(set(self.lane_ids)) != len(self.lane_ids):
            raise ValueError("continuous second-striker lanes must be unique and non-empty")
        if not set(self.lane_ids) <= _KNOWN_LANES:
            raise ValueError("continuous second-striker lane is unknown")
        if not 23.0 <= self.simulation_duration_sec <= 25.0:
            raise ValueError("continuous second-striker duration is invalid")
        if not 1.10 <= self.minimum_second_glove_height_m <= 1.50:
            raise ValueError("continuous second-striker glove-height gate is invalid")
        if not 0.005 <= self.maximum_glove_surface_penetration_m <= 0.03:
            raise ValueError("continuous second-striker penetration gate is invalid")
        if not 0.002 <= self.maximum_glove_surface_separation_m <= 0.02:
            raise ValueError("continuous second-striker separation gate is invalid")
        if not 0.01 <= self.maximum_precontact_second_ball_speed_mps <= 0.10:
            raise ValueError("continuous second-striker precontact speed gate is invalid")
        if not 0.10 <= self.minimum_post_glove_outward_speed_mps <= 2.0:
            raise ValueError("continuous second-striker deflection gate is invalid")
        if not 0.60 <= self.minimum_post_second_pelvis_height_m <= 0.72:
            raise ValueError("continuous second-striker recovery-height gate is invalid")
        if not 1.0e-12 <= self.maximum_control_dt_error_sec <= 1.0e-6:
            raise ValueError("continuous second-striker clock gate is invalid")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.commercial_use_allowed
        ):
            raise ValueError("continuous second-striker exam must remain SIM_ONLY")

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


def physical_second_striker_kwargs(
    *,
    lane: DynamicCornerSaveLane,
    assets: dict[str, Path],
    recovery_checkpoint: Path,
    recovery_exam: Path,
    config: ContinuousSecondStrikerSaveExamConfig,
) -> tuple[dict[str, Any], G1GoalkeeperConfig, G1TrainingGoalSpec]:
    """Assemble one no-cannon physical successor from the qualified prefix."""

    kwargs, goalkeeper, goal = _second_save_kwargs(
        lane=lane,
        assets=assets,
        recovery_checkpoint=recovery_checkpoint,
        recovery_exam=recovery_exam,
        config=ContinuousSecondSaveExamConfig(
            simulation_duration_sec=config.simulation_duration_sec
        ),
    )
    kwargs.update(
        simulation_duration_sec=config.simulation_duration_sec,
        second_threat_config=None,
        physical_second_striker_config=config.striker,
        second_striker_ballistic_contact_torque_config=(UpperCornerStrikePolicy().torque_config()),
    )
    return kwargs, goalkeeper, goal


def _event_index(time: NDArray[np.float64], value: float) -> int:
    return int(np.clip(np.searchsorted(time, value, side="left"), 1, time.size - 1))


def _continuous_metrics(
    result: G1SharedWorldResult,
    trajectory: dict[str, NDArray[Any]],
) -> dict[str, Any]:
    contact_time = result.second_striker_contact_time_sec
    glove_time = result.goalkeeper_second_glove_contact_time_sec
    time = np.asarray(trajectory["time"], dtype=np.float64)
    ball_velocity = np.asarray(trajectory["second_ball_velocity"], dtype=np.float64)[:, :3]
    actor_active = np.asarray(trajectory["second_striker_ballistic_actor_active"], dtype=np.bool_)
    actor_torque = np.asarray(trajectory["second_striker_ballistic_actor_torque"], dtype=np.float64)
    target_active = np.asarray(
        trajectory["second_striker_ballistic_contact_active"], dtype=np.bool_
    )
    torque_active = np.asarray(
        trajectory["second_striker_ballistic_contact_torque_active"], dtype=np.bool_
    )
    observed_flight = np.asarray(trajectory["goalkeeper_observed_flight_active"], dtype=np.bool_)
    flight_start = np.asarray(trajectory["goalkeeper_observed_flight_start_sec"], dtype=np.float64)
    reaction = np.asarray(trajectory["goalkeeper_reaction_active"], dtype=np.bool_)
    if (
        time.ndim != 1
        or time.size < 100
        or ball_velocity.shape != (time.size, 3)
        or actor_torque.shape != (time.size, 29)
        or any(value.shape != time.shape for value in (actor_active, target_active, torque_active))
        or any(value.shape != time.shape for value in (observed_flight, flight_start, reaction))
        or not np.all(np.isfinite(ball_velocity))
        or not np.all(np.isfinite(actor_torque))
    ):
        return {"valid": False, "reason": "continuous physical telemetry is invalid"}
    base: dict[str, Any] = {
        "valid": True,
        "save_phase_valid": False,
        "control_dt_max_error_sec": float(np.max(np.abs(np.diff(time) - 0.02))),
        "actor_active_frame_count": int(np.count_nonzero(actor_active)),
        "actor_peak_torque_nm": float(np.max(np.abs(actor_torque))),
        "contact_target_memory_active_frame_count": int(np.count_nonzero(target_active)),
        "upper_corner_torque_memory_active_frame_count": int(np.count_nonzero(torque_active)),
        "new_flight_observed": False,
        "new_flight_start_time_sec": None,
        "causal_reaction_observed": False,
    }
    if contact_time is None:
        return base | {
            "reason": "second contact timestamp is absent",
            "contact_frame": None,
            "glove_frame": None,
        }
    contact = _event_index(time, contact_time)
    precontact = time < contact_time - 1.0e-9
    base |= {
        "precontact_peak_ball_speed_mps": float(
            np.max(np.linalg.norm(ball_velocity[precontact], axis=1))
        ),
        "contact_frame": contact,
        "glove_frame": None,
    }
    if glove_time is None:
        return base | {"reason": "second glove timestamp is absent"}
    glove = _event_index(time, glove_time)
    causal = (time >= contact_time) & (time <= glove_time + 0.04)
    post_glove = (time >= glove_time) & (time <= glove_time + 0.06)
    starts = flight_start[causal]
    starts = starts[np.isfinite(starts)]
    post_velocity = ball_velocity[post_glove]
    return base | {
        "save_phase_valid": True,
        "new_flight_observed": bool(np.any(observed_flight[causal])),
        "new_flight_start_time_sec": None if starts.size == 0 else float(np.min(starts)),
        "causal_reaction_observed": bool(np.any(reaction[causal])),
        "pre_glove_forward_speed_mps": float(ball_velocity[max(0, glove - 1), 0]),
        "post_glove_minimum_forward_speed_mps": float(np.min(post_velocity[:, 0])),
        "post_glove_peak_outward_speed_mps": float(np.max(np.abs(post_velocity[:, 1]))),
        "glove_frame": glove,
    }


def evaluate_continuous_second_striker_save(
    *,
    result: G1SharedWorldResult,
    trajectory: dict[str, NDArray[Any]],
    lane: DynamicCornerSaveLane,
    goal: G1TrainingGoalSpec,
    goalkeeper: G1GoalkeeperConfig,
    config: ContinuousSecondStrikerSaveExamConfig,
) -> dict[str, Any]:
    """Score one uninterrupted physical second-striker successor."""

    required = {
        "time",
        "second_ball_pose",
        "second_ball_velocity",
        "second_striker_pelvis_pose",
        "second_striker_joint_position",
        "second_striker_ballistic_actor_active",
        "second_striker_ballistic_actor_torque",
        "second_striker_ballistic_contact_active",
        "second_striker_ballistic_contact_torque_active",
        "goalkeeper_pelvis_pose",
        "goalkeeper_root_velocity",
        "goalkeeper_torso_quaternion",
        "goalkeeper_foot_contact",
        "goalkeeper_joint_position",
        "goalkeeper_command_mps",
        "goalkeeper_observed_flight_active",
        "goalkeeper_observed_flight_start_sec",
        "goalkeeper_reaction_active",
        "goalkeeper_contact_epoch",
    }
    if not required <= set(trajectory):
        return {"passed": False, "reason": "physical successor telemetry is incomplete"}
    first = evaluate_dynamic_takeoff_save(
        result=result,
        trajectory=trajectory,
        config=lane.takeoff_config,
    )
    metrics = _continuous_metrics(result, trajectory)
    time = np.asarray(trajectory["time"], dtype=np.float64)
    final_mask = time >= float(time[-1]) - config.ready.ready_hold_sec + 1.0e-9
    final_ready = _ready_window(
        trajectory=trajectory,
        mask=final_mask,
        goal=goal,
        depth_from_goal_line_m=goalkeeper.depth_from_goal_line_m,
        config=config.ready,
    )
    surface = result.goalkeeper_second_glove_contact_surface_distance_m
    flight_start = metrics.get("new_flight_start_time_sec")
    contact_time = result.second_striker_contact_time_sec
    gates = {
        "qualified_first_airborne_save": first.get("passed") is True,
        "four_g1_two_ball_from_time_zero": bool(
            result.physical_second_striker_enabled
            and result.second_striker_ball_existed_from_time_zero
        ),
        "measured_ready_rearm_before_foot_contact": bool(
            result.second_threat_rearmed
            and result.second_threat_rearm_time_sec is not None
            and contact_time is not None
            and result.second_threat_rearm_time_sec < contact_time
        ),
        "anatomical_second_striker_contact": bool(
            result.second_striker_contact_observed
            and result.second_striker_contact_foot == config.striker.kick_foot
            and config.striker.minimum_contact_force_n
            <= result.second_striker_contact_force_peak_n
            <= config.striker.maximum_contact_force_n
            and not result.second_striker_unexpected_precontact_collision_geoms
        ),
        "stationary_second_ball_until_contact": bool(
            metrics.get("precontact_peak_ball_speed_mps", math.inf)
            <= config.maximum_precontact_second_ball_speed_mps
        ),
        "learned_multi_role_contact_stack_active": bool(
            metrics.get("actor_active_frame_count", 0) > 0
            and metrics.get("actor_peak_torque_nm", 0.0) > 0.0
            and metrics.get("contact_target_memory_active_frame_count", 0) > 0
            and metrics.get("upper_corner_torque_memory_active_frame_count", 0) > 0
        ),
        "bounded_forward_high_launch": bool(
            result.second_striker_postcontact_peak_ball_speed_mps
            - result.second_striker_precontact_peak_ball_speed_mps
            >= config.striker.minimum_post_contact_speed_gain_mps
            and result.second_striker_postcontact_peak_forward_ball_speed_mps
            >= config.striker.minimum_forward_ball_speed_mps
        ),
        "new_causal_goalkeeper_flight_epoch": bool(
            metrics.get("new_flight_observed") is True
            and isinstance(flight_start, int | float)
            and contact_time is not None
            and flight_start >= contact_time - 1.0e-9
            and metrics.get("causal_reaction_observed") is True
        ),
        "collision_faithful_high_glove_contact": bool(
            result.goalkeeper_second_glove_contact_observed
            and result.goalkeeper_second_glove_contact_height_m is not None
            and result.goalkeeper_second_glove_contact_height_m
            >= config.minimum_second_glove_height_m
            and surface is not None
            and -config.maximum_glove_surface_penetration_m
            <= surface
            <= config.maximum_glove_surface_separation_m
        ),
        "outward_physical_save": bool(
            result.goalkeeper_second_save_observed
            and not result.second_ball_goal_crossed
            and metrics.get("pre_glove_forward_speed_mps", -math.inf) > 0.0
            and metrics.get("post_glove_minimum_forward_speed_mps", math.inf) < 0.0
            and metrics.get("post_glove_peak_outward_speed_mps", 0.0)
            >= config.minimum_post_glove_outward_speed_mps
        ),
        "second_striker_remains_stable": bool(
            result.second_striker_min_pelvis_height_m is not None
            and result.second_striker_min_pelvis_height_m >= config.striker.minimum_pelvis_height_m
            and not result.second_striker_joint_limit_violation
        ),
        "whole_world_safety": bool(
            result.finite_state
            and not result.joint_limit_violation
            and not result.torque_limit_violation
            and not result.actuator_saturation
            and result.robot_robot_contact_count == 0
        ),
        "continuous_clock": bool(
            metrics.get("valid") is True
            and metrics.get("control_dt_max_error_sec", math.inf)
            <= config.maximum_control_dt_error_sec
        ),
        "final_goalkeeper_ready": final_ready.get("passed") is True,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": metrics,
        "first_takeoff_exam": first,
        "final_ready": final_ready,
        "result": result.to_dict(),
    }


def run_continuous_second_striker_save_exam(
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
    config: ContinuousSecondStrikerSaveExamConfig | None = None,
) -> dict[str, Any]:
    """Run and strictly replay the four-G1 physical successor."""

    active = config or ContinuousSecondStrikerSaveExamConfig()
    checkout = source_checkout.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("continuous second-striker evidence must use a new external directory")
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
        raise FileNotFoundError("continuous second-striker input artifact is missing")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    destination.mkdir(parents=True)
    upper_policy = UpperCornerStrikePolicy()
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.continuous_second_striker_save_request.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "source_commit": _git_head(checkout),
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "upper_corner_muscle_memory_hash": upper_policy.artifact_hash,
        "artifacts": {
            key: (_git_head(value) if key == "dive_source" else hash_bytes(value.read_bytes()))
            for key, value in assets.items()
        },
        "recovery_checkpoint_hash": hash_bytes(recovery_checkpoint.read_bytes()),
        "recovery_exam_hash": hash_bytes(recovery_exam.read_bytes()),
        "physics_backend": "mujoco_cpu",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "reset_or_teleport_used": False,
        "ball_cannon_used": False,
    }
    request["request_hash"] = hash_json(request)
    _atomic_json(destination / "request.json", request)
    lanes = {lane.lane_id: lane for lane in expanded_dynamic_corner_lanes()}
    cases: dict[str, Any] = {}
    for lane_id in active.lane_ids:
        lane = lanes[lane_id]
        kwargs, goalkeeper, goal = physical_second_striker_kwargs(
            lane=lane,
            assets=assets,
            recovery_checkpoint=recovery_checkpoint,
            recovery_exam=recovery_exam,
            config=active,
        )
        result, trajectory = simulate_shared_world(asset_root, **kwargs)
        replay_result, replay_trajectory = simulate_shared_world(asset_root, **kwargs)
        evaluation = evaluate_continuous_second_striker_save(
            result=result,
            trajectory=trajectory,
            lane=lane,
            goal=goal,
            goalkeeper=goalkeeper,
            config=active,
        )
        replay = evaluate_continuous_second_striker_save(
            result=replay_result,
            trajectory=replay_trajectory,
            lane=lane,
            goal=goal,
            goalkeeper=goalkeeper,
            config=active,
        )
        trajectory_file = f"{lane_id}-four-g1-trajectory.npz"
        trajectory_path = destination / trajectory_file
        _atomic_trajectory(trajectory_path, trajectory)
        digest = trajectory_digest(trajectory)
        replay_digest = trajectory_digest(replay_trajectory)
        strict_replay = bool(
            result.to_dict() == replay_result.to_dict()
            and digest == replay_digest
            and evaluation == replay
        )
        cases[lane_id] = {
            "passed": bool(evaluation.get("passed") and strict_replay),
            "evaluation": evaluation,
            "strict_replay": strict_replay,
            "trajectory_file": trajectory_file,
            "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
            "trajectory_digest": digest,
            "replay_trajectory_digest": replay_digest,
        }
    implementation_hash = hash_json(
        {
            "shared_world": hash_bytes(
                (Path(__file__).parents[1] / "skills/team/shared_world.py").read_bytes()
            ),
            "exam": hash_bytes(Path(__file__).read_bytes()),
            "actor_decoder": hash_bytes(
                (
                    Path(__file__).parents[1] / "growth/ballistic_contact_impulse_actor.py"
                ).read_bytes()
            ),
        }
    )
    evidence: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.continuous_second_striker_save_evidence.v1",
        "claim": _CLAIM,
        "passed": all(case["passed"] for case in cases.values()),
        "cases": cases,
        "request_hash": request["request_hash"],
        "source_commit": request["source_commit"],
        "implementation_hash": implementation_hash,
        "physics_backend": "mujoco_cpu",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "reset_or_teleport_used": False,
        "ball_cannon_used": False,
        "pixels_used_for_scoring": False,
        "promotion_status": "SIM_ONLY_EVIDENCE_NOT_HARDWARE_AUTHORIZATION",
    }
    evidence["report_hash"] = hash_json(evidence)
    evidence_path = destination / "evidence.json"
    _atomic_json(evidence_path, evidence)
    return validate_continuous_second_striker_save_exam(evidence_path)


def validate_continuous_second_striker_save_exam(path: Path) -> dict[str, Any]:
    """Validate hashes and non-promotion boundaries without running physics."""

    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("continuous second-striker evidence must be a JSON object")
    expected = str(payload.pop("report_hash", ""))
    if expected != hash_json(payload):
        raise ValueError("continuous second-striker evidence integrity mismatch")
    payload["report_hash"] = expected
    cases = payload.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise ValueError("continuous second-striker evidence has no cases")
    for case in cases.values():
        if not isinstance(case, dict) or case.get("strict_replay") is not True:
            raise ValueError("continuous second-striker strict replay is absent")
        trajectory = resolved.parent / str(case.get("trajectory_file", ""))
        if not trajectory.is_file() or case.get("trajectory_hash") != hash_bytes(
            trajectory.read_bytes()
        ):
            raise ValueError("continuous second-striker trajectory binding changed")
    if (
        payload.get("schema_version") != "rosclaw_soccer.continuous_second_striker_save_evidence.v1"
        or payload.get("claim") != _CLAIM
        or payload.get("physics_backend") != "mujoco_cpu"
        or payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("hardware_command_sent") is not False
        or payload.get("commercial_use_allowed") is not False
        or payload.get("reset_or_teleport_used") is not False
        or payload.get("ball_cannon_used") is not False
        or payload.get("pixels_used_for_scoring") is not False
    ):
        raise ValueError("continuous second-striker authority contract is invalid")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--striker-actor", type=Path, required=True)
    parser.add_argument("--goalkeeper-actor", type=Path, required=True)
    parser.add_argument("--gmt-model", type=Path, required=True)
    parser.add_argument("--gmt-skill", type=Path, required=True)
    parser.add_argument("--dive-source", type=Path, required=True)
    parser.add_argument("--dive-athlete-checkpoint", type=Path, required=True)
    parser.add_argument("--dive-athlete-exam", type=Path, required=True)
    parser.add_argument("--recovery-athlete-checkpoint", type=Path, required=True)
    parser.add_argument("--recovery-athlete-exam", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_continuous_second_striker_save_exam(
        asset_root=cast(Path, args.asset_root),
        striker_actor_path=cast(Path, args.striker_actor),
        goalkeeper_actor_path=cast(Path, args.goalkeeper_actor),
        gmt_model_path=cast(Path, args.gmt_model),
        gmt_skill_path=cast(Path, args.gmt_skill),
        dive_source_checkout=cast(Path, args.dive_source),
        dive_athlete_checkpoint_path=cast(Path, args.dive_athlete_checkpoint),
        dive_athlete_exam_path=cast(Path, args.dive_athlete_exam),
        recovery_athlete_checkpoint_path=cast(Path, args.recovery_athlete_checkpoint),
        recovery_athlete_exam_path=cast(Path, args.recovery_athlete_exam),
        output_dir=cast(Path, args.output_dir),
        source_checkout=cast(Path, args.source_checkout),
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ContinuousSecondStrikerSaveExamConfig",
    "evaluate_continuous_second_striker_save",
    "physical_second_striker_kwargs",
    "run_continuous_second_striker_save_exam",
    "validate_continuous_second_striker_save_exam",
]
