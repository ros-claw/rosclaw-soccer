"""Three-G1 support controller for a striker-owned free-kick duel.

The striker remains under the complete free-kick controller.  This module owns
only the two attached agents: a causally reacting goalkeeper and an off-ball
team-mate.  All three bodies and the football advance in one MuJoCo model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.mujoco_primitives import load_robonaldo, roll_pitch
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, ShotParameters
from rosclaw_soccer.skills.team.shared_world import (
    G1JointGuardConfig,
    _fill_local_state,
    _make_robot,
    _normalized_locomotion_command,
    _normalized_zero_locomotion_command,
    _project_joint_safe_torque,
    _robot_geom_ids,
    _update_policy,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec

_CONTROL_DT = 0.02
_MOTION_REL = Path("policy/robonaldo/model/freekick_motion.npz")


@dataclass(frozen=True)
class G1FrontDuelConfig:
    """Request-bound layout and causal goalkeeper response contract."""

    teammate_origin_m: tuple[float, float, float] = (2.0, -2.45, 0.0)
    goalkeeper_depth_from_goal_line_m: float = 0.48
    goalkeeper_reaction_delay_sec: float = 0.06
    goalkeeper_lateral_position_gain: float = 1.80
    goalkeeper_maximum_lateral_speed_mps: float = 0.40
    goalkeeper_ready_shuffle_speed_mps: float = 0.08
    goalkeeper_arm_spread_rad: float = 0.38
    goalkeeper_maximum_waist_lean_rad: float = 0.12
    goalkeeper_block_blend_frames: int = 10
    goalkeeper_block_hold_frames: int = 42
    goalkeeper_block_shoulder_pitch_rad: float = 0.32
    goalkeeper_block_shoulder_roll_rad: float = 0.24
    goalkeeper_block_elbow_flex_rad: float = 0.12
    torque_authority_projection_ratio: float = 0.99
    torque_authority_projection_max_fraction: float = 0.01
    schema_version: str = "rosclaw_soccer.g1_front_duel_config.v2"

    def __post_init__(self) -> None:
        values = (
            *self.teammate_origin_m,
            self.goalkeeper_depth_from_goal_line_m,
            self.goalkeeper_reaction_delay_sec,
            self.goalkeeper_lateral_position_gain,
            self.goalkeeper_maximum_lateral_speed_mps,
            self.goalkeeper_ready_shuffle_speed_mps,
            self.goalkeeper_arm_spread_rad,
            self.goalkeeper_maximum_waist_lean_rad,
            self.goalkeeper_block_shoulder_pitch_rad,
            self.goalkeeper_block_shoulder_roll_rad,
            self.goalkeeper_block_elbow_flex_rad,
            self.torque_authority_projection_ratio,
            self.torque_authority_projection_max_fraction,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("front-duel configuration must be finite")
        if abs(self.teammate_origin_m[1]) < 1.50:
            raise ValueError("front-duel team-mate must remain outside the striker lane")
        if not 0.25 <= self.goalkeeper_depth_from_goal_line_m <= 0.80:
            raise ValueError("front-duel goalkeeper depth must be in [0.25, 0.80] m")
        if not 0.04 <= self.goalkeeper_reaction_delay_sec <= 0.20:
            raise ValueError("front-duel goalkeeper reaction delay must be in [0.04, 0.20] s")
        if not 0.5 <= self.goalkeeper_lateral_position_gain <= 2.5:
            raise ValueError("front-duel goalkeeper position gain must be in [0.5, 2.5]")
        if not 0.20 <= self.goalkeeper_maximum_lateral_speed_mps <= 0.40:
            raise ValueError("front-duel goalkeeper speed must be in [0.20, 0.40] m/s")
        if not 0.0 <= self.goalkeeper_ready_shuffle_speed_mps <= 0.20:
            raise ValueError("front-duel ready shuffle must be in [0, 0.20] m/s")
        if not 0.10 <= self.goalkeeper_arm_spread_rad <= 0.45:
            raise ValueError("front-duel arm spread must be in [0.10, 0.45] rad")
        if not 0.0 <= self.goalkeeper_maximum_waist_lean_rad <= 0.15:
            raise ValueError("front-duel waist lean must be in [0, 0.15] rad")
        if not 1 <= self.goalkeeper_block_blend_frames <= 40:
            raise ValueError("front-duel block blend must contain 1..40 frames")
        if not 1 <= self.goalkeeper_block_hold_frames <= 100:
            raise ValueError("front-duel block hold must contain 1..100 frames")
        if not 0.90 <= self.torque_authority_projection_ratio <= 0.99:
            raise ValueError("front-duel torque authority ratio must be in [0.90, 0.99]")
        if not 0.001 <= self.torque_authority_projection_max_fraction <= 0.05:
            raise ValueError("front-duel torque projection fraction must be in [0.001, 0.05]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class G1FrontDuelSummary:
    """Machine-readable safety and interaction result for the attached agents."""

    finite_state: bool
    teammate_minimum_pelvis_height_m: float
    goalkeeper_minimum_pelvis_height_m: float
    goalkeeper_peak_tilt_rad: float
    goalkeeper_peak_lateral_speed_mps: float
    goalkeeper_reaction_frames: int
    goalkeeper_ball_contact_observed: bool
    goalkeeper_ball_contact_time_sec: float | None
    robot_robot_contact_count: int
    actuator_saturation: bool
    actuator_saturation_steps: int
    actuator_saturation_fraction: float
    actuator_peak_demand_ratio: float
    joint_limit_violation: bool
    torque_authority_projection_steps: int = 0
    torque_authority_projection_fraction: float = 0.0
    torque_authority_projection_peak_correction_nm: float = 0.0
    torque_authority_preprojection_peak_demand_ratio: float = 0.0
    torque_authority_projection_qualified: bool = True
    three_agents_share_physics_world: bool = True
    all_agents_have_independent_controllers: bool = True
    schema_version: str = "rosclaw_soccer.g1_front_duel_summary.v2"

    @property
    def passed(self) -> bool:
        return bool(
            self.finite_state
            and self.teammate_minimum_pelvis_height_m >= 0.55
            and self.goalkeeper_minimum_pelvis_height_m >= 0.55
            and self.goalkeeper_peak_tilt_rad <= 0.75
            and self.goalkeeper_reaction_frames > 0
            and self.robot_robot_contact_count == 0
            and not self.actuator_saturation
            and self.torque_authority_projection_qualified
            and not self.joint_limit_violation
            and self.three_agents_share_physics_world
            and self.all_agents_have_independent_controllers
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def _standby_pose() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    target = np.asarray(
        (
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
        ),
        dtype=np.float64,
    )
    kp = np.asarray(
        (100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 300, 300, 300)
        + (100, 100, 50, 50, 20, 20, 20)
        + (100, 100, 50, 50, 20, 20, 20),
        dtype=np.float64,
    )
    kd = np.asarray(
        (2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 3, 3, 3)
        + (2, 2, 2, 2, 1, 1, 1)
        + (2, 2, 2, 2, 1, 1, 1),
        dtype=np.float64,
    )
    return target, kp, kd


class G1FrontDuelController:
    """Own the two auxiliary G1s while leaving striker authority untouched."""

    def __init__(
        self,
        *,
        model: Any,
        data: Any,
        asset_root: Path,
        goal: G1TrainingGoalSpec,
        config: G1FrontDuelConfig,
        ball_body: int,
        ball_qpos: int,
        ball_qvel: int,
        ball_geom: int,
    ) -> None:
        import mujoco

        self.model = model
        self.goal = goal
        self.config = config
        self.ball_body = ball_body
        self.ball_qpos = ball_qpos
        self.ball_qvel = ball_qvel
        self.ball_geom = ball_geom
        state_type, output_type, policy_type, _ = load_robonaldo(asset_root)
        with np.load(asset_root / _MOTION_REL) as motion:
            pelvis_height = float(np.asarray(motion["body_pos_w"])[0, 0, 2])
        standby_target, standby_kp, standby_kd = _standby_pose()
        common: dict[str, Any] = {
            "model": model,
            "data": data,
            "state_type": state_type,
            "output_type": output_type,
            "policy_type": policy_type,
            "parameters": ShotParameters(policy_type="parameter"),
            "start_sec": math.inf,
            "initial_position": np.asarray((0.0, 0.0, pelvis_height), dtype=np.float64),
            "initial_quaternion": np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
            "initial_joints": standby_target,
            "phase_hold_frames": 0,
            "standby_target": standby_target,
            "standby_kp": standby_kp,
            "standby_kd": standby_kd,
            "recovery_controller": None,
            "post_policy_frame": None,
            "post_policy_blend_frames": 0,
            "phase_sync_enabled": False,
            "recovery_torque_actor": None,
            "joint_guard_enabled": True,
            "post_policy_neutral_velocity_enabled": False,
            "post_policy_forward_velocity_mps": 0.0,
            "joint_guard_config": G1JointGuardConfig(
                margin_rad=0.08,
                prediction_horizon_sec=0.16,
                boundary_kp=140.0,
                boundary_kd=12.0,
            ),
            "joint_guard_late_config": None,
            "post_policy_recovery_enabled": False,
            "early_arrival_parameters": None,
            "motion_prior": None,
            "motion_prior_position_blend": 0.0,
            "motion_prior_velocity_blend": 0.0,
            "motion_prior_strike_leg_scale": 1.0,
            "motion_prior_joint_scales": (1.0,) * 29,
            "motion_prior_velocity_joint_scales": (1.0,) * 29,
            "motion_prior_contact_policy_frame": 253,
            "contact_prior": None,
            "contact_prior_position_blend": 0.0,
            "contact_prior_velocity_blend": 0.0,
            "contact_prior_contact_policy_frame": 253,
            "contact_prior_joint_scales": (1.0,) * 6,
        }
        teammate_origin = np.asarray(config.teammate_origin_m, dtype=np.float64)
        goalkeeper_origin = np.asarray(
            (goal.plane_x_m - config.goalkeeper_depth_from_goal_line_m, 0.0, 0.0),
            dtype=np.float64,
        )
        self.teammate = _make_robot(
            role="teammate",
            prefix="passer_",
            origin=teammate_origin,
            yaw=0.0,
            target_local=np.asarray((goal.plane_x_m, 0.0, 0.2), dtype=np.float32),
            use_locomotion_standby=True,
            **common,
        )
        self.goalkeeper = _make_robot(
            role="goalkeeper",
            prefix="goalkeeper_",
            origin=goalkeeper_origin,
            yaw=math.pi,
            target_local=np.asarray((-goal.plane_x_m, 0.0, 0.2), dtype=np.float32),
            # The qualified locomotion policy owns the complete support chain.
            # The first duel contract adapts only its causal lateral command;
            # arm reach remains frozen until a separate stability exam passes.
            use_locomotion_standby=True,
            **common,
        )
        self._goalkeeper_geoms = _robot_geom_ids(model, self.goalkeeper.pelvis_body)
        self._teammate_geoms = _robot_geom_ids(model, self.teammate.pelvis_body)
        self._shooter_geoms = _robot_geom_ids(
            model,
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")),
        )
        self._hard_limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
        self._finite = True
        self._actuator_saturation = False
        self._actuator_saturation_steps = 0
        self._torque_steps = 0
        self._actuator_peak_demand_ratio = 0.0
        self._torque_authority_projection_steps = 0
        self._torque_authority_projection_peak_correction = 0.0
        self._torque_authority_preprojection_peak_demand_ratio = 0.0
        self._joint_limit_violation = False
        self._teammate_min_height = pelvis_height
        self._goalkeeper_min_height = pelvis_height
        self._goalkeeper_peak_tilt = 0.0
        self._goalkeeper_peak_lateral_speed = 0.0
        self._goalkeeper_reaction_frames = 0
        self._goalkeeper_contact_time: float | None = None
        self._robot_robot_contacts = 0
        self._target_y = 0.0
        self._target_z = goal.ball_radius_m
        self._reaction_active = False
        self._block_frame = -1

    @property
    def teammate_origin_m(self) -> tuple[float, float, float]:
        return (
            float(self.teammate.origin[0]),
            float(self.teammate.origin[1]),
            float(self.teammate.origin[2]),
        )

    @property
    def goalkeeper_origin_m(self) -> tuple[float, float, float]:
        return (
            float(self.goalkeeper.origin[0]),
            float(self.goalkeeper.origin[1]),
            float(self.goalkeeper.origin[2]),
        )

    def update(
        self,
        data: Any,
        *,
        simulation_frame: int,
        striker_contact_time: float | None,
    ) -> None:
        for robot in (self.teammate, self.goalkeeper):
            _fill_local_state(robot, data, self.ball_body, self.ball_qvel)
        if self.teammate.standby_policy is None:
            raise RuntimeError("front-duel team-mate locomotion policy is unavailable")
        self.teammate.state.vel_cmd = _normalized_zero_locomotion_command(
            self.teammate.standby_policy
        )
        ball = np.asarray(data.qpos[self.ball_qpos : self.ball_qpos + 3], dtype=np.float64)
        velocity = np.asarray(data.qvel[self.ball_qvel : self.ball_qvel + 3], dtype=np.float64)
        timestamp = float(data.time)
        self._reaction_active = bool(
            striker_contact_time is not None
            and timestamp >= striker_contact_time + self.config.goalkeeper_reaction_delay_sec
            and velocity[0] > 0.10
        )
        if self._reaction_active:
            horizon = max(
                0.0,
                (float(data.qpos[self.goalkeeper.qpos_base]) - float(ball[0]))
                / max(float(velocity[0]), 1e-9),
            )
            mouth_limit = self.goal.width_m / 2.0 - self.goal.ball_radius_m
            self._target_y = float(
                np.clip(ball[1] + velocity[1] * horizon, -mouth_limit, mouth_limit)
            )
            self._target_z = float(
                np.clip(
                    ball[2] + velocity[2] * horizon - 0.5 * 9.81 * horizon * horizon,
                    self.goal.ball_radius_m,
                    self.goal.height_m - self.goal.ball_radius_m,
                )
            )
            current_y = float(data.qpos[self.goalkeeper.qpos_base + 1])
            world_velocity_y = float(
                np.clip(
                    self.config.goalkeeper_lateral_position_gain * (self._target_y - current_y),
                    -self.config.goalkeeper_maximum_lateral_speed_mps,
                    self.config.goalkeeper_maximum_lateral_speed_mps,
                )
            )
            self._block_frame += 1
            self._goalkeeper_reaction_frames += 1
        else:
            world_velocity_y = self.config.goalkeeper_ready_shuffle_speed_mps * math.sin(
                2.0 * math.pi * timestamp / 2.4
            )
            self._target_y = 0.12 * math.sin(2.0 * math.pi * timestamp / 2.4)
            self._target_z = self.goal.ball_radius_m
            self._block_frame = -1
        if self.goalkeeper.standby_policy is not None:
            self.goalkeeper.state.vel_cmd = _normalized_locomotion_command(
                self.goalkeeper.standby_policy,
                np.asarray((0.0, -world_velocity_y, 0.0), dtype=np.float64),
            )
        _update_policy(self.teammate, simulation_frame, timestamp_sec=timestamp)
        _update_policy(self.goalkeeper, simulation_frame, timestamp_sec=timestamp)

    def apply_torque(self, data: Any) -> None:
        saturated_step = False
        projection_active_step = False
        for robot in (self.teammate, self.goalkeeper):
            q = np.asarray(data.qpos[robot.joint_qpos], dtype=np.float64)
            dq = np.asarray(data.qvel[robot.joint_qvel], dtype=np.float64)
            raw = (robot.last_target - q) * robot.kp + (robot.target_velocity - dq) * robot.kd
            ranges = np.asarray(self.model.jnt_range[robot.joint_ids], dtype=np.float64)
            limited = np.asarray(self.model.jnt_limited[robot.joint_ids], dtype=bool)
            projected, _ = _project_joint_safe_torque(
                joint_position=q,
                joint_velocity=dq,
                commanded_torque=raw,
                joint_ranges=ranges,
                limited=limited,
                margin_rad=robot.joint_guard_config.margin_rad,
                prediction_horizon_sec=robot.joint_guard_config.prediction_horizon_sec,
                boundary_kp=robot.joint_guard_config.boundary_kp,
                boundary_kd=robot.joint_guard_config.boundary_kd,
            )
            preprojection_ratio = float(np.max(np.abs(projected) / self._hard_limits))
            self._torque_authority_preprojection_peak_demand_ratio = max(
                self._torque_authority_preprojection_peak_demand_ratio,
                preprojection_ratio,
            )
            authority_bound = self._hard_limits * self.config.torque_authority_projection_ratio
            authority_projected = np.clip(projected, -authority_bound, authority_bound)
            correction = authority_projected - projected
            projection_active = bool(np.any(np.abs(correction) > 1e-12))
            projection_active_step = projection_active_step or projection_active
            self._torque_authority_projection_peak_correction = max(
                self._torque_authority_projection_peak_correction,
                float(np.max(np.abs(correction))),
            )
            demand_ratio = float(np.max(np.abs(authority_projected) / self._hard_limits))
            self._actuator_peak_demand_ratio = max(
                self._actuator_peak_demand_ratio,
                demand_ratio,
            )
            saturated_step = saturated_step or demand_ratio >= 0.999
            data.ctrl[robot.actuators] = authority_projected
        self._torque_steps += 1
        self._torque_authority_projection_steps += int(projection_active_step)
        self._actuator_saturation_steps += int(saturated_step)
        self._actuator_saturation = self._actuator_saturation or saturated_step

    def observe_physics(self, data: Any) -> None:
        self._finite = self._finite and bool(
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(data.ctrl))
        )
        self._teammate_min_height = min(
            self._teammate_min_height,
            float(data.qpos[self.teammate.qpos_base + 2]),
        )
        self._goalkeeper_min_height = min(
            self._goalkeeper_min_height,
            float(data.qpos[self.goalkeeper.qpos_base + 2]),
        )
        roll, pitch = roll_pitch(data.xquat[self.goalkeeper.torso_body])
        self._goalkeeper_peak_tilt = max(self._goalkeeper_peak_tilt, abs(roll), abs(pitch))
        self._goalkeeper_peak_lateral_speed = max(
            self._goalkeeper_peak_lateral_speed,
            abs(float(data.qvel[self.goalkeeper.qvel_base + 1])),
        )
        for robot in (self.teammate, self.goalkeeper):
            positions = np.asarray(data.qpos[robot.joint_qpos], dtype=np.float64)
            ranges = np.asarray(self.model.jnt_range[robot.joint_ids], dtype=np.float64)
            limited = np.asarray(self.model.jnt_limited[robot.joint_ids], dtype=bool)
            self._joint_limit_violation = self._joint_limit_violation or bool(
                np.any(positions[limited] < ranges[limited, 0] - 1e-5)
                or np.any(positions[limited] > ranges[limited, 1] + 1e-5)
            )
        ball_contact = False
        robot_contact = False
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.ball_geom in pair and pair & self._goalkeeper_geoms:
                ball_contact = True
            if (pair & self._goalkeeper_geoms and pair & self._shooter_geoms) or (
                pair & self._teammate_geoms
                and pair & (self._shooter_geoms | self._goalkeeper_geoms)
            ):
                robot_contact = True
        if ball_contact and self._goalkeeper_contact_time is None:
            self._goalkeeper_contact_time = float(data.time)
        self._robot_robot_contacts += int(robot_contact)

    def append_trace(self, trace: dict[str, list[Any]], data: Any) -> None:
        trace["teammate_pelvis_pose"].append(
            data.qpos[self.teammate.qpos_base : self.teammate.qpos_base + 7].copy()
        )
        trace["teammate_joint_position"].append(data.qpos[self.teammate.joint_qpos].copy())
        trace["goalkeeper_pelvis_pose"].append(
            data.qpos[self.goalkeeper.qpos_base : self.goalkeeper.qpos_base + 7].copy()
        )
        trace["goalkeeper_joint_position"].append(data.qpos[self.goalkeeper.joint_qpos].copy())
        trace["goalkeeper_target_y_m"].append(self._target_y)
        trace["goalkeeper_target_z_m"].append(self._target_z)
        trace["goalkeeper_reaction_active"].append(self._reaction_active)
        trace["goalkeeper_ball_contact"].append(
            self._goalkeeper_contact_time is not None
            and abs(float(data.time) - self._goalkeeper_contact_time) <= _CONTROL_DT + 1e-9
        )

    @staticmethod
    def add_trace_keys(trace: dict[str, list[Any]]) -> None:
        for key in (
            "teammate_pelvis_pose",
            "teammate_joint_position",
            "goalkeeper_pelvis_pose",
            "goalkeeper_joint_position",
            "goalkeeper_target_y_m",
            "goalkeeper_target_z_m",
            "goalkeeper_reaction_active",
            "goalkeeper_ball_contact",
        ):
            trace[key] = []

    def summary(self) -> G1FrontDuelSummary:
        return G1FrontDuelSummary(
            finite_state=self._finite,
            teammate_minimum_pelvis_height_m=self._teammate_min_height,
            goalkeeper_minimum_pelvis_height_m=self._goalkeeper_min_height,
            goalkeeper_peak_tilt_rad=self._goalkeeper_peak_tilt,
            goalkeeper_peak_lateral_speed_mps=self._goalkeeper_peak_lateral_speed,
            goalkeeper_reaction_frames=self._goalkeeper_reaction_frames,
            goalkeeper_ball_contact_observed=self._goalkeeper_contact_time is not None,
            goalkeeper_ball_contact_time_sec=self._goalkeeper_contact_time,
            robot_robot_contact_count=self._robot_robot_contacts,
            actuator_saturation=self._actuator_saturation,
            actuator_saturation_steps=self._actuator_saturation_steps,
            actuator_saturation_fraction=(
                self._actuator_saturation_steps / max(1, self._torque_steps)
            ),
            actuator_peak_demand_ratio=self._actuator_peak_demand_ratio,
            joint_limit_violation=self._joint_limit_violation,
            torque_authority_projection_steps=self._torque_authority_projection_steps,
            torque_authority_projection_fraction=(
                self._torque_authority_projection_steps / max(1, self._torque_steps)
            ),
            torque_authority_projection_peak_correction_nm=(
                self._torque_authority_projection_peak_correction
            ),
            torque_authority_preprojection_peak_demand_ratio=(
                self._torque_authority_preprojection_peak_demand_ratio
            ),
            torque_authority_projection_qualified=bool(
                self._torque_authority_projection_steps / max(1, self._torque_steps)
                <= self.config.torque_authority_projection_max_fraction
            ),
        )


__all__ = [
    "G1FrontDuelConfig",
    "G1FrontDuelController",
    "G1FrontDuelSummary",
]
