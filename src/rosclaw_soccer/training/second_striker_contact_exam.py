"""Physical fourth-G1/second-ball contact qualification.

This is the bridge between the S107 SIM-only ball cannon and a continuous
second-striker rollout.  It compiles four qualified G1 bodies and two physical
footballs from time zero, then requires the fourth G1's frozen RoboNaldo ONNX
policy to create an ordered anatomical foot contact and forward ball-speed
gain.  It does not yet claim the complete first-save/second-save chain.
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

from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.providers.g1.mujoco_primitives import load_robonaldo
from rosclaw_soccer.sim.contracts import (
    G1_HARD_TORQUE_LIMITS,
    ShotParameters,
    hash_bytes,
    hash_json,
)
from rosclaw_soccer.skills.team.shared_world import (
    G1JointGuardConfig,
    _enter_policy,
    _fill_local_state,
    _make_robot,
    _project_joint_safe_torque,
    _robot_geom_ids,
    _update_policy,
)
from rosclaw_soccer.world.field import (
    G1TrainingGoalSpec,
    build_g1_four_player_two_ball_stadium_model,
)

_STANDBY_TARGET = (
    -0.2,
    0.0,
    0.0,
    0.42,
    -0.23,
    0.0,
    -0.2,
    0.0,
    0.0,
    0.42,
    -0.23,
    0.0,
    0.0,
    0.0,
    0.0,
    0.35,
    0.18,
    0.0,
    0.87,
    0.0,
    0.0,
    0.0,
    0.35,
    -0.18,
    0.0,
    0.87,
    0.0,
    0.0,
    0.0,
)
_STANDBY_KP = (
    100,
    100,
    100,
    150,
    40,
    40,
    100,
    100,
    100,
    150,
    40,
    40,
    300,
    300,
    300,
    100,
    100,
    50,
    50,
    20,
    20,
    20,
    100,
    100,
    50,
    50,
    20,
    20,
    20,
)
_STANDBY_KD = (
    2,
    2,
    2,
    4,
    2,
    2,
    2,
    2,
    2,
    4,
    2,
    2,
    3,
    3,
    3,
    2,
    2,
    2,
    2,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    1,
    1,
    1,
)


@dataclass(frozen=True)
class SecondStrikerContactExamConfig:
    """Fail-closed bounds for the fourth G1's isolated contact exam."""

    simulation_duration_sec: float = 9.0
    policy_start_time_sec: float = 2.19
    second_striker_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    second_ball_origin_m: tuple[float, float, float] = (1.285, -0.018, 0.115)
    target_m: tuple[float, float, float] = (7.5, 0.45, 1.35)
    policy_target_m: tuple[float, float, float] = (7.5, 0.70, 0.50)
    minimum_post_contact_speed_gain_mps: float = 4.0
    minimum_forward_ball_speed_mps: float = 3.0
    minimum_contact_force_n: float = 20.0
    maximum_contact_force_n: float = 1500.0
    minimum_striker_pelvis_height_m: float = 0.60
    maximum_torque_fraction: float = 0.85
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    commercial_use_allowed: bool = False
    schema_version: str = "rosclaw_soccer.second_striker_contact_exam_config.v1"

    def __post_init__(self) -> None:
        flat = (
            self.simulation_duration_sec,
            self.policy_start_time_sec,
            *self.second_striker_origin_m,
            *self.second_ball_origin_m,
            *self.target_m,
            *self.policy_target_m,
            self.minimum_post_contact_speed_gain_mps,
            self.minimum_forward_ball_speed_mps,
            self.minimum_contact_force_n,
            self.maximum_contact_force_n,
            self.minimum_striker_pelvis_height_m,
            self.maximum_torque_fraction,
        )
        if not all(math.isfinite(value) for value in flat):
            raise ValueError("second-striker contact settings must be finite")
        if not 6.0 <= self.simulation_duration_sec <= 9.0:
            raise ValueError("second-striker contact duration is invalid")
        if not 1.5 <= self.policy_start_time_sec <= 3.0:
            raise ValueError("second-striker policy start is invalid")
        if self.second_striker_origin_m != (0.0, 0.0, 0.0):
            raise ValueError("second striker must own the canonical isolated strike pocket")
        if not 1.15 <= self.second_ball_origin_m[0] <= 1.40 or not (
            -0.25 <= self.second_ball_origin_m[1] <= 0.15
        ):
            raise ValueError("second football is outside the qualified strike pocket")
        if not 0.105 <= self.second_ball_origin_m[2] <= 0.130:
            raise ValueError("second football height is invalid")
        if not 4.0 <= self.target_m[0] <= 12.0 or abs(self.target_m[1]) > 3.4:
            raise ValueError("second-striker target is invalid")
        if not 0.115 <= self.target_m[2] <= 2.30:
            raise ValueError("second-striker target height is invalid")
        if not 4.0 <= self.policy_target_m[0] <= 12.0 or not (
            -1.2 <= self.policy_target_m[1] <= 1.2 and 0.115 <= self.policy_target_m[2] <= 1.2
        ):
            raise ValueError("second-striker policy target is invalid")
        if not 2.0 <= self.minimum_post_contact_speed_gain_mps <= 8.0:
            raise ValueError("second-striker speed-gain gate is invalid")
        if not 2.0 <= self.minimum_forward_ball_speed_mps <= 8.0:
            raise ValueError("second-striker forward-speed gate is invalid")
        if not 5.0 <= self.minimum_contact_force_n <= 100.0:
            raise ValueError("second-striker contact-force gate is invalid")
        if not 200.0 <= self.maximum_contact_force_n <= 3000.0:
            raise ValueError("second-striker maximum contact-force gate is invalid")
        if self.maximum_contact_force_n <= self.minimum_contact_force_n:
            raise ValueError("second-striker contact-force interval is empty")
        if not 0.50 <= self.minimum_striker_pelvis_height_m <= 0.72:
            raise ValueError("second-striker pelvis-height gate is invalid")
        if not 0.50 <= self.maximum_torque_fraction <= 0.90:
            raise ValueError("second-striker torque envelope is invalid")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.commercial_use_allowed
        ):
            raise ValueError("second-striker contact exam must remain non-commercial SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


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


def _git_head(checkout: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _make_probe_robot(
    *,
    model: Any,
    data: Any,
    role: str,
    prefix: str,
    origin: tuple[float, float, float],
    yaw: float,
    state_type: Any,
    output_type: Any,
    policy_type: Any,
    initial_position: NDArray[np.float64],
    initial_quaternion: NDArray[np.float64],
    initial_joints: NDArray[np.float64],
    target: tuple[float, float, float],
    active: bool,
    active_start_sec: float = 0.0,
) -> Any:
    return _make_robot(
        model=model,
        data=data,
        role=role,
        prefix=prefix,
        origin=np.asarray(origin, dtype=np.float64),
        yaw=yaw,
        state_type=state_type,
        output_type=output_type,
        policy_type=policy_type,
        parameters=ShotParameters(
            stance_offset_y=-0.06,
            pelvis_yaw_offset=0.175,
            com_shift_y=-0.065,
            swing_speed_scale=0.90,
            foot_yaw_offset=0.03025,
            recovery_step_length=0.055,
            policy_type="parameter",
        ),
        start_sec=active_start_sec if active else math.inf,
        initial_position=initial_position,
        initial_quaternion=initial_quaternion,
        initial_joints=initial_joints,
        target_local=np.asarray(target, dtype=np.float32),
        phase_hold_frames=0,
        standby_target=np.asarray(_STANDBY_TARGET, dtype=np.float64),
        standby_kp=np.asarray(_STANDBY_KP, dtype=np.float64),
        standby_kd=np.asarray(_STANDBY_KD, dtype=np.float64),
        use_locomotion_standby=True,
        recovery_controller=None,
        post_policy_frame=275 if active else None,
        post_policy_blend_frames=10 if active else 0,
        phase_sync_enabled=False,
        recovery_torque_actor=None,
        joint_guard_enabled=True,
        post_policy_neutral_velocity_enabled=False,
        post_policy_forward_velocity_mps=0.0,
        joint_guard_config=G1JointGuardConfig(),
        joint_guard_late_config=None,
        post_policy_recovery_enabled=False,
        early_arrival_parameters=None,
        motion_prior=None,
        motion_prior_position_blend=0.0,
        motion_prior_velocity_blend=0.0,
        motion_prior_strike_leg_scale=1.0,
        motion_prior_joint_scales=(1.0,) * 29,
        motion_prior_velocity_joint_scales=(1.0,) * 29,
        motion_prior_contact_policy_frame=253,
        contact_prior=None,
        contact_prior_position_blend=0.0,
        contact_prior_velocity_blend=0.0,
        contact_prior_contact_policy_frame=253,
        contact_prior_joint_scales=(1.0,) * 6,
    )


def simulate_second_striker_contact(
    *,
    asset_root: Path,
    config: SecondStrikerContactExamConfig | None = None,
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Run one deterministic fourth-G1 physical contact attempt."""

    import mujoco

    active = config or SecondStrikerContactExamConfig()
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    goal = G1TrainingGoalSpec(
        plane_x_m=active.target_m[0],
        width_m=7.32,
        height_m=2.44,
        target_y_m=active.target_m[1],
        target_z_m=active.target_m[2],
        regulation_field_enabled=True,
    )
    model = build_g1_four_player_two_ball_stadium_model(
        asset_root,
        passer_origin_m=(5.10, -3.0, 0.0),
        goalkeeper_origin_m=(goal.plane_x_m - 0.48, 0.0, 0.0),
        second_striker_origin_m=active.second_striker_origin_m,
        first_ball_origin_m=(3.895, -2.84, goal.ball_radius_m),
        second_ball_origin_m=active.second_ball_origin_m,
        spec=goal,
    )
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002
    state_type, output_type, policy_type, _ = load_robonaldo(asset_root)
    motion_path = asset_root / "policy" / "robonaldo" / "model" / "freekick_motion.npz"
    with np.load(motion_path, allow_pickle=False) as motion:
        initial_position = np.asarray(motion["body_pos_w"][0, 0], dtype=np.float64)
        initial_quaternion = np.asarray(motion["body_quat_w"][0, 0], dtype=np.float64)
    robots = (
        _make_probe_robot(
            model=model,
            data=data,
            role="source_shooter",
            prefix="",
            origin=(0.0, 3.0, 0.0),
            yaw=0.0,
            state_type=state_type,
            output_type=output_type,
            policy_type=policy_type,
            initial_position=initial_position,
            initial_quaternion=initial_quaternion,
            initial_joints=np.asarray(_STANDBY_TARGET, dtype=np.float64),
            target=active.policy_target_m,
            active=False,
        ),
        _make_probe_robot(
            model=model,
            data=data,
            role="passer",
            prefix="passer_",
            origin=(5.10, -3.0, 0.0),
            yaw=math.pi,
            state_type=state_type,
            output_type=output_type,
            policy_type=policy_type,
            initial_position=initial_position,
            initial_quaternion=initial_quaternion,
            initial_joints=np.asarray(_STANDBY_TARGET, dtype=np.float64),
            target=(-5.0, 0.0, 0.2),
            active=False,
        ),
        _make_probe_robot(
            model=model,
            data=data,
            role="goalkeeper",
            prefix="goalkeeper_",
            origin=(goal.plane_x_m - 0.48, 0.0, 0.0),
            yaw=math.pi,
            state_type=state_type,
            output_type=output_type,
            policy_type=policy_type,
            initial_position=initial_position,
            initial_quaternion=initial_quaternion,
            initial_joints=np.asarray(_STANDBY_TARGET, dtype=np.float64),
            target=(-5.0, 0.0, 0.2),
            active=False,
        ),
        _make_probe_robot(
            model=model,
            data=data,
            role="second_striker",
            prefix="second_striker_",
            origin=active.second_striker_origin_m,
            yaw=0.0,
            state_type=state_type,
            output_type=output_type,
            policy_type=policy_type,
            initial_position=initial_position,
            initial_quaternion=initial_quaternion,
            initial_joints=np.asarray(_STANDBY_TARGET, dtype=np.float64),
            target=active.target_m,
            active=True,
            active_start_sec=active.policy_start_time_sec,
        ),
    )
    second_striker = robots[-1]
    second_ball_body = int(model.body("second_ball").id)
    second_ball_geom = int(model.geom("second_ball_geom").id)
    second_ball_joint = int(model.joint("second_ball_free").id)
    second_ball_qpos = int(model.jnt_qposadr[second_ball_joint])
    second_ball_qvel = int(model.jnt_dofadr[second_ball_joint])
    second_striker_geoms = _robot_geom_ids(model, second_striker.pelvis_body)
    hard_limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    torque_limits = hard_limits * active.maximum_torque_fraction
    mujoco.mj_forward(model, data)
    for robot in robots:
        _fill_local_state(robot, data, second_ball_body, second_ball_qvel)

    trace: dict[str, list[Any]] = {
        "time": [],
        "second_ball_pose": [],
        "second_ball_velocity": [],
        "second_striker_pelvis_pose": [],
        "second_striker_joint_position": [],
        "second_striker_commanded_torque": [],
        "second_striker_executed_torque": [],
        "second_striker_policy_frame": [],
        "second_striker_foot_contact": [],
        "second_striker_contact_force_n": [],
    }
    contact_time: float | None = None
    contact_foot: str | None = None
    contact_force_peak = 0.0
    precontact_speed = 0.0
    postcontact_speed = 0.0
    postcontact_forward_speed = 0.0
    minimum_pelvis_height = math.inf
    finite = True
    torque_violation = False
    joint_violation = False
    unexpected_precontact_collision_geoms: set[str] = set()
    total_frames = int(round(active.simulation_duration_sec / 0.02))
    force_buffer = np.zeros(6, dtype=np.float64)

    for frame in range(total_frames):
        if not second_striker.entered and float(data.time) + 1.0e-12 >= second_striker.start_sec:
            _enter_policy(second_striker)
        policy_frames: dict[str, int] = {}
        for robot in robots:
            _fill_local_state(robot, data, second_ball_body, second_ball_qvel)
            policy_frames[robot.role] = _update_policy(
                robot,
                frame,
                timestamp_sec=float(data.time),
            )
        commanded: dict[str, NDArray[np.float64]] = {}
        frame_contact = False
        frame_force = 0.0
        for _ in range(10):
            for robot in robots:
                q = np.asarray(data.qpos[robot.joint_qpos], dtype=np.float64)
                dq = np.asarray(data.qvel[robot.joint_qvel], dtype=np.float64)
                raw = (robot.last_target - q) * robot.kp + (robot.target_velocity - dq) * robot.kd
                commanded[robot.role] = raw.copy()
                projected = raw
                if robot is second_striker and second_striker.contact_latched:
                    guard = second_striker.joint_guard_config
                    projected, _ = _project_joint_safe_torque(
                        joint_position=q,
                        joint_velocity=dq,
                        commanded_torque=raw,
                        joint_ranges=model.jnt_range[robot.joint_ids],
                        limited=model.jnt_limited[robot.joint_ids].astype(bool),
                        margin_rad=guard.margin_rad,
                        prediction_horizon_sec=guard.prediction_horizon_sec,
                        boundary_kp=guard.boundary_kp,
                        boundary_kd=guard.boundary_kd,
                    )
                applied = np.clip(projected, -torque_limits, torque_limits)
                if robot is second_striker:
                    torque_violation = torque_violation or bool(
                        np.any(np.abs(applied) > hard_limits + 1.0e-9)
                    )
                data.ctrl[robot.actuators] = applied
            mujoco.mj_step(model, data)
            for index in range(int(data.ncon)):
                contact = data.contact[index]
                if second_ball_geom not in {int(contact.geom1), int(contact.geom2)}:
                    continue
                other = (
                    int(contact.geom2)
                    if int(contact.geom1) == second_ball_geom
                    else int(contact.geom1)
                )
                other_name = model.geom(other).name
                anatomical_foot = other in second_striker_geoms and (
                    other_name.startswith("second_striker_left_foot")
                    or other_name.startswith("second_striker_right_foot")
                )
                if contact_time is None and other_name != "floor" and not anatomical_foot:
                    unexpected_precontact_collision_geoms.add(other_name)
                if not anatomical_foot:
                    continue
                mujoco.mj_contactForce(model, data, index, force_buffer)
                force_n = float(np.linalg.norm(force_buffer[:3]))
                frame_contact = True
                frame_force = max(frame_force, force_n)
                contact_force_peak = max(contact_force_peak, force_n)
                if contact_time is None:
                    contact_time = float(data.time)
                    contact_foot = "left" if "left_foot" in other_name else "right"
                    second_striker.contact_latched = True
                    second_striker.contact_time = contact_time
        velocity = np.asarray(data.qvel[second_ball_qvel : second_ball_qvel + 3], dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        if contact_time is None:
            precontact_speed = max(precontact_speed, speed)
        else:
            postcontact_speed = max(postcontact_speed, speed)
            postcontact_forward_speed = max(postcontact_forward_speed, float(velocity[0]))
        pelvis_height = float(data.qpos[second_striker.qpos_base + 2])
        minimum_pelvis_height = min(minimum_pelvis_height, pelvis_height)
        ranges = model.jnt_range[second_striker.joint_ids]
        limited = model.jnt_limited[second_striker.joint_ids].astype(bool)
        q = np.asarray(data.qpos[second_striker.joint_qpos], dtype=np.float64)
        joint_violation = joint_violation or bool(
            np.any(q[limited] < ranges[limited, 0] - 1.0e-5)
            or np.any(q[limited] > ranges[limited, 1] + 1.0e-5)
        )
        finite = finite and all(
            np.all(np.isfinite(value)) for value in (data.qpos, data.qvel, data.ctrl, velocity)
        )
        trace["time"].append(float(data.time))
        trace["second_ball_pose"].append(data.qpos[second_ball_qpos : second_ball_qpos + 7].copy())
        trace["second_ball_velocity"].append(
            data.qvel[second_ball_qvel : second_ball_qvel + 6].copy()
        )
        trace["second_striker_pelvis_pose"].append(
            data.qpos[second_striker.qpos_base : second_striker.qpos_base + 7].copy()
        )
        trace["second_striker_joint_position"].append(q.copy())
        trace["second_striker_commanded_torque"].append(commanded["second_striker"].copy())
        trace["second_striker_executed_torque"].append(
            data.actuator_force[second_striker.actuators].copy()
        )
        trace["second_striker_policy_frame"].append(policy_frames["second_striker"])
        trace["second_striker_foot_contact"].append(frame_contact)
        trace["second_striker_contact_force_n"].append(frame_force)
        if not finite:
            break

    trajectory = {name: np.asarray(values) for name, values in trace.items()}
    gates = {
        "four_g1_two_ball_world": model.nu == 116 and model.nq >= 4 * 36 + 14,
        "ordered_anatomical_foot_contact": contact_time is not None
        and contact_foot in {"left", "right"},
        "no_unexpected_precontact_collision": not unexpected_precontact_collision_geoms,
        "physical_contact_force": active.minimum_contact_force_n
        <= contact_force_peak
        <= active.maximum_contact_force_n,
        "post_contact_speed_gain": postcontact_speed - precontact_speed
        >= active.minimum_post_contact_speed_gain_mps,
        "forward_goalward_ball": postcontact_forward_speed >= active.minimum_forward_ball_speed_mps,
        "finite_safe_striker": finite
        and not torque_violation
        and not joint_violation
        and minimum_pelvis_height >= active.minimum_striker_pelvis_height_m,
    }
    result: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.second_striker_contact_result.v1",
        "passed": all(gates.values()),
        "gates": gates,
        "contact_time_sec": contact_time,
        "contact_foot": contact_foot,
        "contact_force_peak_n": contact_force_peak,
        "precontact_peak_ball_speed_mps": precontact_speed,
        "postcontact_peak_ball_speed_mps": postcontact_speed,
        "postcontact_peak_forward_ball_speed_mps": postcontact_forward_speed,
        "minimum_striker_pelvis_height_m": minimum_pelvis_height,
        "finite_state": finite,
        "torque_limit_violation": torque_violation,
        "joint_limit_violation": joint_violation,
        "unexpected_precontact_collision_geoms": sorted(unexpected_precontact_collision_geoms),
        "second_ball_existed_from_time_zero": True,
        "reset_or_teleport_used": False,
        "frozen_robonaldo_onnx_policy_used": True,
        "complete_second_save_claimed": False,
        "trajectory_digest": trajectory_digest(trajectory),
    }
    return result, trajectory


def run_second_striker_contact_exam(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
    config: SecondStrikerContactExamConfig | None = None,
) -> dict[str, Any]:
    """Write an integrity-bound development or promoted contact receipt."""

    active = config or SecondStrikerContactExamConfig()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("second-striker contact evidence must use a new external directory")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    destination.mkdir(parents=True)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.second_striker_contact_request.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "source_commit": _git_head(checkout),
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "physics_backend": "mujoco_cpu",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    request["request_hash"] = hash_json(request)
    _atomic_json(destination / "request.json", request)
    result, trajectory = simulate_second_striker_contact(
        asset_root=asset_root,
        config=active,
    )
    replay_result, replay_trajectory = simulate_second_striker_contact(
        asset_root=asset_root,
        config=active,
    )
    strict_replay = bool(
        result == replay_result
        and trajectory_digest(trajectory) == trajectory_digest(replay_trajectory)
    )
    trajectory_path = destination / "trajectory.npz"
    _atomic_trajectory(trajectory_path, trajectory)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.second_striker_contact_exam.v1",
        "passed": bool(result["passed"] and strict_replay),
        "promotion_status": (
            "PROMOTED_SIM_ONLY_SECOND_STRIKER_CONTACT"
            if result["passed"] and strict_replay
            else "REJECTED_DEVELOPMENT"
        ),
        "claim": "FOURTH_G1_PHYSICAL_SECOND_BALL_FOOT_CONTACT_ONLY",
        "request_hash": request["request_hash"],
        "source_commit": request["source_commit"],
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "result": result,
        "strict_replay": strict_replay,
        "trajectory_file": trajectory_path.name,
        "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
        "implementation_hash": _implementation_hash(),
        "physics_backend": "mujoco_cpu",
        "pixels_used_for_scoring": False,
        "reset_or_teleport_used": False,
        "complete_second_save_claimed": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(destination / "evidence.json", report)
    return report


def validate_second_striker_contact_exam(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("second-striker contact evidence must be an object")
    unhashed = dict(payload)
    claimed = unhashed.pop("report_hash", None)
    result = payload.get("result")
    if not (
        claimed == hash_json(unhashed)
        and payload.get("schema_version") == "rosclaw_soccer.second_striker_contact_exam.v1"
        and payload.get("passed") is True
        and payload.get("promotion_status") == "PROMOTED_SIM_ONLY_SECOND_STRIKER_CONTACT"
        and payload.get("claim") == "FOURTH_G1_PHYSICAL_SECOND_BALL_FOOT_CONTACT_ONLY"
        and payload.get("implementation_hash") == _implementation_hash()
        and payload.get("physics_backend") == "mujoco_cpu"
        and payload.get("pixels_used_for_scoring") is False
        and payload.get("reset_or_teleport_used") is False
        and payload.get("complete_second_save_claimed") is False
        and payload.get("activation_ceiling") == "SIM_ONLY"
        and payload.get("hardware_command_sent") is False
        and payload.get("commercial_use_allowed") is False
        and payload.get("strict_replay") is True
        and isinstance(result, dict)
        and result.get("passed") is True
        and isinstance(result.get("gates"), dict)
        and all(result["gates"].values())
    ):
        raise ValueError("second-striker contact evidence failed closed")
    return cast(dict[str, Any], payload)


def _implementation_hash() -> str:
    root = Path(__file__).parents[1]
    files = (
        Path(__file__),
        root / "world" / "field.py",
        root / "skills" / "team" / "shared_world.py",
    )
    return str(hash_json({path.name: hash_bytes(path.read_bytes()) for path in files}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, default=Path.cwd())
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_second_striker_contact_exam(
        asset_root=args.asset_root,
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "SecondStrikerContactExamConfig",
    "run_second_striker_contact_exam",
    "simulate_second_striker_contact",
    "validate_second_striker_contact_exam",
]
