"""Independent CPU MuJoCo exam for an MJWarp-trained goalkeeper candidate."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.training.goalkeeper_agility import shape_goalkeeper_action_numpy
from rosclaw_soccer.training.goalkeeper_combat_mjwarp import (
    COMBAT_ARM_RESIDUAL_LIMIT,
    COMBAT_RECOVERY_CAPTURE_HORIZON_SEC,
    COMBAT_RECOVERY_CENTER_DEADBAND_M,
    COMBAT_RECOVERY_LATERAL_GATE_LIMIT,
    COMBAT_SECOND_SHOT_ARM_RESIDUAL_LIMIT,
    COMBAT_SIGNED_LATERAL_GATE_LIMIT,
    COMBAT_WAIST_RESIDUAL_LIMIT,
)
from rosclaw_soccer.training.goalkeeper_mjwarp import (
    _LOCO_DEFAULT,
    _LOCO_KD,
    _LOCO_KP,
    _LOCO_TO_MOTOR,
    _RESIDUAL_LIMITS_RAD,
    _RESIDUAL_MOTOR_INDICES,
    GoalkeeperMJWarpConfig,
    goalkeeper_world_config,
)
from rosclaw_soccer.training.goalkeeper_mobility_option import (
    MOBILE_UPPER_BODY_KD,
    MOBILE_UPPER_BODY_KP,
    GoalkeeperMobilityOptionConfig,
    guard_lateral_velocity_numpy,
    project_recovery_command_numpy,
)
from rosclaw_soccer.training.goalkeeper_multistep import (
    GoalkeeperMultiStepAccumulator,
    GoalkeeperMultiStepConfig,
    GoalkeeperStepBatch,
)


@dataclass(frozen=True)
class GoalkeeperCPUExamConfig:
    episode_count: int = 64
    first_seed: int = 91_001
    minimum_first_save_rate: float = 0.0
    minimum_second_attempt_save_rate: float = 0.0
    minimum_second_attempt_save_improvement: float = 0.02
    minimum_second_save_improvement: float = 0.02
    maximum_save_rate_regression: float = 0.0
    maximum_applied_actor_action_step: float = 0.06
    minimum_pelvis_height_m: float = 0.65
    maximum_root_speed_mps: float = 1.50
    maximum_root_angular_speed_rad_s: float = 3.50
    maximum_p95_root_angular_speed_rad_s: float = 3.20
    minimum_lateral_speed_improvement_mps: float = 0.10
    minimum_hand_displacement_improvement_m: float = 0.01
    maximum_p95_hand_speed_mps: float = 5.0
    maximum_mean_second_release_lateral_error_m: float = 0.55
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_cpu_exam_config.v8"

    def __post_init__(self) -> None:
        if not 8 <= self.episode_count <= 2048:
            raise ValueError("goalkeeper CPU exam episode count must be in [8, 2048]")
        if not 0 <= self.first_seed < 2**31 - self.episode_count:
            raise ValueError("goalkeeper CPU exam seed range is invalid")
        if not 0.0 <= self.minimum_first_save_rate <= 1.0:
            raise ValueError("goalkeeper CPU exam first-save floor is invalid")
        if not 0.0 <= self.minimum_second_attempt_save_rate <= 1.0:
            raise ValueError("goalkeeper CPU exam second-attempt floor is invalid")
        if not 0.0 <= self.minimum_second_attempt_save_improvement <= 0.25:
            raise ValueError("goalkeeper CPU exam second-attempt improvement is invalid")
        if not 0.0 <= self.minimum_second_save_improvement <= 0.25:
            raise ValueError("goalkeeper CPU exam second-save improvement is invalid")
        if not 0.0 <= self.maximum_save_rate_regression <= 0.05:
            raise ValueError("goalkeeper CPU exam save-rate non-inferiority margin is invalid")
        if not 0.01 <= self.maximum_applied_actor_action_step <= 0.20:
            raise ValueError("goalkeeper CPU exam action-step ceiling is invalid")
        if not 0.55 <= self.minimum_pelvis_height_m <= 0.78:
            raise ValueError("goalkeeper CPU exam pelvis-height floor is invalid")
        if not 0.5 <= self.maximum_root_speed_mps <= 3.0:
            raise ValueError("goalkeeper CPU exam root-speed ceiling is invalid")
        if not 0.5 <= self.maximum_root_angular_speed_rad_s <= 8.0:
            raise ValueError("goalkeeper CPU exam root-angular-speed ceiling is invalid")
        if not 0.5 <= self.maximum_p95_root_angular_speed_rad_s <= 6.0:
            raise ValueError("goalkeeper CPU exam p95 root-angular-speed ceiling is invalid")
        if self.maximum_p95_root_angular_speed_rad_s > self.maximum_root_angular_speed_rad_s:
            raise ValueError("goalkeeper CPU exam p95 ceiling cannot exceed the maximum ceiling")
        if not 0.0 <= self.minimum_lateral_speed_improvement_mps <= 0.50:
            raise ValueError("goalkeeper CPU exam lateral-speed improvement is invalid")
        if not 0.0 <= self.minimum_hand_displacement_improvement_m <= 0.40:
            raise ValueError("goalkeeper CPU exam hand-displacement improvement is invalid")
        if not 0.5 <= self.maximum_p95_hand_speed_mps <= 10.0:
            raise ValueError("goalkeeper CPU exam hand-speed ceiling is invalid")
        if not 0.05 <= self.maximum_mean_second_release_lateral_error_m <= 1.0:
            raise ValueError("goalkeeper CPU exam recenter-error ceiling is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("goalkeeper CPU exam is SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass
class _EpisodeResult:
    seed: int
    reward: float
    first_save: bool
    first_hand_save: bool
    recovered: bool
    second_attempt_save: bool
    second_attempt_hand_save: bool
    second_save: bool
    second_hand_save: bool
    failed: bool
    finite_state: bool
    joint_limit_violation: bool
    torque_limit_violation: bool
    maximum_torque_fraction: float
    maximum_requested_actor_action_step: float
    maximum_applied_actor_action_step: float
    minimum_pelvis_height_m: float
    maximum_root_speed_mps: float
    maximum_root_angular_speed_rad_s: float
    maximum_lateral_displacement_m: float
    maximum_lateral_speed_mps: float
    maximum_hand_displacement_m: float
    maximum_hand_speed_mps: float
    second_release_lateral_error_m: float
    joint_guard_active_fraction: float
    bimanual_reach_fraction: float
    maximum_applied_teacher_blend: float = 0.0
    maximum_applied_runtime_reach_blend: float = 0.0
    trajectory: dict[str, np.ndarray] | None = None


def _require_declared_difficulty_world(config: GoalkeeperMJWarpConfig) -> None:
    """Reject a difficulty label that is not bound to its complete physics preset."""

    declared_world = goalkeeper_world_config(
        difficulty_profile=config.difficulty_profile,
        environment_count=config.environment_count,
        second_shot_probability=config.second_shot_probability,
        shot_intent_cue_enabled=config.shot_intent_cue_enabled,
        hard_shot_fraction=config.hard_shot_fraction,
        hard_shot_height_mode=config.hard_shot_height_mode,
        hard_shot_flight_time_range_sec=config.hard_shot_flight_time_range_sec,
    )
    if config.config_hash != declared_world.config_hash:
        raise ValueError(
            "goalkeeper CPU exam world does not match its declared difficulty profile; "
            "use goalkeeper_world_config()"
        )


@dataclass(frozen=True)
class _CombatTeacherRuntime:
    teacher: Any
    report: dict[str, Any]
    maximum_blend: float
    default_qpos: np.ndarray
    joint_group_scale: np.ndarray
    mobility_option_enabled: bool = False
    mobility_option_config: GoalkeeperMobilityOptionConfig = GoalkeeperMobilityOptionConfig()
    intercept_conditioning_enabled: bool = False
    runtime_reach_atlas: Any | None = None
    runtime_reach_blend: float = 0.0


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _gravity_projection(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = (float(value) for value in quaternion)
    return np.asarray(
        (
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ),
        dtype=np.float64,
    )


def _rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a world vector into a MuJoCo free-joint frame (wxyz)."""

    qw, qx, qy, qz = (float(value) for value in quaternion)
    vx, vy, vz = (float(value) for value in vector)
    return np.asarray(
        (
            (1.0 - 2.0 * (qy * qy + qz * qz)) * vx
            + 2.0 * (qx * qy + qz * qw) * vy
            + 2.0 * (qx * qz - qy * qw) * vz,
            2.0 * (qx * qy - qz * qw) * vx
            + (1.0 - 2.0 * (qx * qx + qz * qz)) * vy
            + 2.0 * (qy * qz + qx * qw) * vz,
            2.0 * (qx * qz + qy * qw) * vx
            + 2.0 * (qy * qz - qx * qw) * vy
            + (1.0 - 2.0 * (qx * qx + qy * qy)) * vz,
        ),
        dtype=np.float64,
    )


def _causal_intercept(
    data: Any,
    *,
    shot_index: int,
    config: GoalkeeperMJWarpConfig,
    mobility_option_enabled: bool = False,
) -> np.ndarray:
    ball = np.asarray(data.qpos[36:39], dtype=np.float64)
    velocity = np.asarray(data.qvel[35:38], dtype=np.float64)
    if shot_index <= 0 or velocity[0] <= 0.10:
        ready_y = (
            float(
                np.clip(
                    data.qpos[1] + 0.28 * data.qvel[1],
                    config.target_y_range_m[0],
                    config.target_y_range_m[1],
                )
            )
            if mobility_option_enabled
            else 0.0
        )
        return np.asarray((config.keeper_x_m - 0.08, ready_y, 0.82), dtype=np.float64)
    time_to_line = float(
        np.clip((config.keeper_x_m - 0.08 - ball[0]) / max(velocity[0], 0.10), 0.0, 1.2)
    )
    return np.asarray(
        (
            config.keeper_x_m - 0.08,
            np.clip(
                ball[1] + velocity[1] * time_to_line,
                config.target_y_range_m[0],
                config.target_y_range_m[1],
            ),
            np.clip(
                ball[2] + velocity[2] * time_to_line - 0.5 * 9.81 * time_to_line**2,
                config.target_z_range_m[0],
                config.target_z_range_m[1],
            ),
        ),
        dtype=np.float64,
    )


def _actor_observation(
    data: Any,
    *,
    target: np.ndarray,
    joint_target: np.ndarray,
    previous_action: np.ndarray,
    shot_index: int,
    step_index: int,
    config: GoalkeeperMJWarpConfig,
    intent_cue: np.ndarray | None = None,
) -> np.ndarray:
    gravity = _gravity_projection(np.asarray(data.qpos[3:7], dtype=np.float64))
    phase = np.zeros(3, dtype=np.float64)
    phase[shot_index] = 1.0
    cue = np.empty(0, dtype=np.float64)
    if config.shot_intent_cue_enabled:
        if intent_cue is None or intent_cue.shape != (3,):
            raise ValueError("CPU goalkeeper shot-intent cue is missing")
        cue = np.asarray(
            (
                (intent_cue[0] - data.qpos[1]) * 0.5,
                (intent_cue[1] - data.qpos[2]) * 0.5,
                intent_cue[2],
            ),
            dtype=np.float64,
        )
    observation = np.concatenate(
        (
            (np.asarray(data.qpos[36:39]) - np.asarray(data.qpos[:3])) * 0.4,
            np.asarray(data.qvel[35:38]) * 0.2,
            (target - np.asarray(data.qpos[:3])) * 0.5,
            gravity,
            np.asarray(data.qvel[:3]) * 0.5,
            np.asarray(data.qvel[3:6]) * 0.25,
            np.asarray(data.qpos[19:36]) - joint_target[12:29],
            np.asarray(data.qvel[18:35]) * 0.05,
            previous_action,
            cue,
            phase,
            np.asarray((step_index * config.control_dt_sec / config.episode_duration_sec,)),
        )
    )
    expected_size = 77 if config.shot_intent_cue_enabled else 74
    if observation.shape != (expected_size,) or not np.all(np.isfinite(observation)):
        raise RuntimeError("CPU goalkeeper actor observation contract changed")
    return np.asarray(np.clip(observation, -10.0, 10.0), dtype=np.float64)


def _robot_ball_contacts(model: Any, data: Any) -> tuple[bool, bool]:
    """Return any-body and anatomically hand-only ball contacts."""

    ball_body = int(model.body("ball").id)
    hand_geoms = {
        int(model.geom(name).id)
        for name in (
            "left_hand_collision",
            "right_hand_collision",
            "left_goalkeeper_glove",
            "right_goalkeeper_glove",
        )
    }
    robot_contact = False
    hand_contact = False
    for index in range(data.ncon):
        contact = data.contact[index]
        geom_one = int(contact.geom1)
        geom_two = int(contact.geom2)
        if contact.dist > 0.002:
            continue
        body_one = int(model.geom_bodyid[geom_one])
        body_two = int(model.geom_bodyid[geom_two])
        ball_one = body_one == ball_body
        ball_two = body_two == ball_body
        robot_one = 1 <= body_one < ball_body
        robot_two = 1 <= body_two < ball_body
        if (ball_one and robot_two) or (ball_two and robot_one):
            robot_contact = True
            hand_contact |= bool(
                (ball_one and geom_two in hand_geoms) or (ball_two and geom_one in hand_geoms)
            )
    return robot_contact, hand_contact


def _sample_shot(rng: np.random.Generator, config: GoalkeeperMJWarpConfig) -> dict[str, Any]:
    target_y = float(rng.uniform(*config.target_y_range_m))
    target_z = float(rng.uniform(*config.target_z_range_m))
    hard = config.hard_shot_fraction > 0.0 and rng.random() < config.hard_shot_fraction
    if hard:
        far_min = min(0.72, config.target_y_range_m[1] - 0.02)
        target_y = float(rng.choice((-1.0, 1.0)) * rng.uniform(far_min, config.target_y_range_m[1]))
        if config.hard_shot_height_mode == "balanced":
            height_band = int(rng.integers(0, 3))
        else:
            height_band = {"low": 0, "mid": 1, "high": 2}[config.hard_shot_height_mode]
        if height_band == 0:
            height_range = (
                config.target_z_range_m[0],
                min(0.60, config.target_z_range_m[1] - 0.04),
            )
        elif height_band == 1:
            height_range = (
                max(0.60, config.target_z_range_m[0] + 0.02),
                min(1.10, config.target_z_range_m[1] - 0.02),
            )
        else:
            height_range = (
                min(1.10, config.target_z_range_m[1] - 0.02),
                config.target_z_range_m[1],
            )
        target_z = float(rng.uniform(*height_range))
    flight_range = (
        config.hard_shot_flight_time_range_sec
        if hard and config.hard_shot_flight_time_range_sec is not None
        else config.flight_time_range_sec
    )
    return {
        "target": np.asarray(
            (
                config.keeper_x_m - 0.08,
                target_y,
                target_z,
            ),
            dtype=np.float64,
        ),
        "flight": float(rng.uniform(*flight_range)),
        "start_x": float(rng.uniform(*config.ball_start_x_range_m)),
        "start_z": float(rng.uniform(*config.ball_start_z_range_m)),
    }


def _sample_intent_cue(
    rng: np.random.Generator,
    shot: dict[str, Any],
    config: GoalkeeperMJWarpConfig,
) -> np.ndarray:
    target = np.asarray(shot["target"], dtype=np.float64)
    visible = float(rng.random() >= config.shot_intent_cue_dropout_probability)
    return np.asarray(
        (
            np.clip(
                target[1] + rng.normal(0.0, config.shot_intent_cue_lateral_noise_m),
                *config.target_y_range_m,
            )
            * visible,
            np.clip(
                target[2] + rng.normal(0.0, config.shot_intent_cue_height_noise_m),
                *config.target_z_range_m,
            )
            * visible,
            visible,
        ),
        dtype=np.float64,
    )


def _park_ball(data: Any) -> None:
    data.qpos[36:43] = (-20.0, 0.0, 0.115, 1.0, 0.0, 0.0, 0.0)
    data.qvel[35:41] = 0.0


def _launch_ball(data: Any, shot: dict[str, Any]) -> None:
    target = np.asarray(shot["target"], dtype=np.float64)
    flight = float(shot["flight"])
    start_x = float(shot["start_x"])
    start_z = float(shot["start_z"])
    data.qpos[36:43] = (
        start_x,
        0.18 * target[1],
        start_z,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    data.qvel[35:41] = (
        (target[0] - start_x) / flight,
        (target[1] - data.qpos[37]) / flight,
        (target[2] - start_z + 0.5 * 9.81 * flight**2) / flight,
        0.0,
        0.0,
        0.0,
    )


def _run_episode(
    *,
    model: Any,
    data: Any,
    locomotion: Any,
    policy: Any,
    seed: int,
    world_config: GoalkeeperMJWarpConfig,
    combat_teacher: _CombatTeacherRuntime | None = None,
    record_trajectory: bool = False,
) -> _EpisodeResult:
    import mujoco
    import torch

    rng = np.random.default_rng(seed)
    first_shot = _sample_shot(rng, world_config)
    second_shot = _sample_shot(rng, world_config)
    first_intent_cue = np.zeros(3, dtype=np.float64)
    second_intent_cue = np.zeros(3, dtype=np.float64)
    if world_config.shot_intent_cue_enabled:
        first_intent_cue = _sample_intent_cue(rng, first_shot, world_config)
        second_intent_cue = _sample_intent_cue(rng, second_shot, world_config)
    second_enabled = bool(rng.random() < world_config.second_shot_probability)
    loco_to_motor = np.asarray(_LOCO_TO_MOTOR, dtype=np.int64)
    loco_default = np.asarray(_LOCO_DEFAULT, dtype=np.float64)
    kp = np.zeros(29, dtype=np.float64)
    kd = np.zeros(29, dtype=np.float64)
    kp[loco_to_motor] = np.asarray(_LOCO_KP, dtype=np.float64)
    kd[loco_to_motor] = np.asarray(_LOCO_KD, dtype=np.float64)
    mobility_enabled = bool(combat_teacher and combat_teacher.mobility_option_enabled)
    mobility = (
        combat_teacher.mobility_option_config
        if combat_teacher is not None
        else GoalkeeperMobilityOptionConfig()
    )
    if mobility_enabled:
        kp[12:] = np.asarray(MOBILE_UPPER_BODY_KP, dtype=np.float64)
        kd[12:] = np.asarray(MOBILE_UPPER_BODY_KD, dtype=np.float64)
    limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    residual_indices = np.asarray(_RESIDUAL_MOTOR_INDICES, dtype=np.int64)
    residual_limits = np.asarray(_RESIDUAL_LIMITS_RAD, dtype=np.float64)
    target = np.zeros(29, dtype=np.float64)
    target[loco_to_motor] = loco_default
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = (world_config.keeper_x_m, 0.0, 0.793, 0.0, 0.0, 0.0, 1.0)
    data.qpos[7:36] = target
    _park_ball(data)
    mujoco.mj_forward(model, data)
    previous_action = np.zeros(18, dtype=np.float64)
    teacher_history = np.zeros((10, 96), dtype=np.float32)
    previous_teacher_action = np.zeros(29, dtype=np.float32)
    previous_teacher_target_delta = np.zeros(29, dtype=np.float64)
    previous_joint_velocity = np.asarray(data.qvel[6:35], dtype=np.float64).copy()
    loco_action = np.zeros(29, dtype=np.float64)
    hidden = torch.zeros((1, 1, 256), dtype=torch.float32)
    cell = torch.zeros_like(hidden)
    task = GoalkeeperMultiStepAccumulator(
        1,
        GoalkeeperMultiStepConfig(
            control_dt_sec=world_config.control_dt_sec,
            episode_duration_sec=world_config.episode_duration_sec,
            first_shot_release_sec=world_config.first_shot_release_sec,
            second_shot_release_sec=world_config.second_shot_release_sec,
            joint_acceleration_penalty_scale=3.0e-7,
            contact_bonus=5.0,
            hand_contact_bonus=world_config.hand_contact_bonus,
            true_save_bonus=25.0,
            hand_save_bonus=world_config.hand_save_bonus,
            second_hand_save_bonus=world_config.second_hand_save_bonus,
            second_save_bonus=40.0,
            recovery_bonus=15.0,
            reach_reward_scale=world_config.reach_reward_scale,
            bimanual_reach_reward_scale=world_config.bimanual_reach_reward_scale,
            second_shot_reach_reward_multiplier=(world_config.second_shot_reach_reward_multiplier),
            root_linear_speed_penalty_scale=world_config.root_linear_speed_penalty_scale,
            root_angular_speed_penalty_scale=world_config.root_angular_speed_penalty_scale,
            action_magnitude_penalty_scale=world_config.action_magnitude_penalty_scale,
            unsafe_penalty=50.0,
        ),
    )
    first_release = round(world_config.first_shot_release_sec / world_config.control_dt_sec)
    first_end = round(world_config.first_shot_end_sec / world_config.control_dt_sec)
    second_release = round(world_config.second_shot_release_sec / world_config.control_dt_sec)
    second_end = round(world_config.second_shot_end_sec / world_config.control_dt_sec)
    cumulative_reward = 0.0
    maximum_requested_action_step = 0.0
    maximum_applied_action_step = 0.0
    maximum_torque_fraction = 0.0
    minimum_pelvis_height = math.inf
    maximum_root_speed = 0.0
    maximum_root_angular_speed = 0.0
    maximum_lateral_displacement = 0.0
    maximum_lateral_speed = 0.0
    left_hand_geom = int(model.geom("left_hand_collision").id)
    right_hand_geom = int(model.geom("right_hand_collision").id)
    ready_left_hand_relative = np.asarray(
        data.geom_xpos[left_hand_geom] - data.qpos[:3], dtype=np.float64
    ).copy()
    ready_right_hand_relative = np.asarray(
        data.geom_xpos[right_hand_geom] - data.qpos[:3], dtype=np.float64
    ).copy()
    previous_left_hand_relative = ready_left_hand_relative.copy()
    previous_right_hand_relative = ready_right_hand_relative.copy()
    maximum_hand_displacement = 0.0
    maximum_hand_speed = 0.0
    second_release_lateral_error = 0.0
    joint_limit_violation = False
    torque_limit_violation = False
    joint_guard_active_steps = 0
    bimanual_reach_steps = 0
    active_flight_steps = 0
    maximum_applied_teacher_blend = 0.0
    maximum_applied_runtime_reach_blend = 0.0
    mobility_teacher_gate = 0.0
    teacher_recovery_gate = 0.0
    teacher_recovery_age_steps = -1
    previous_teacher_shot_active = False
    shot_index = 0
    recorded_qpos: list[np.ndarray] = []
    recorded_qvel: list[np.ndarray] = []
    recorded_action: list[np.ndarray] = []
    recorded_shot_index: list[int] = []
    recorded_left_hand: list[np.ndarray] = []
    recorded_right_hand: list[np.ndarray] = []
    recorded_time: list[float] = []

    for step_index in range(world_config.episode_steps):
        teacher_recovery_active = False
        if step_index < first_release:
            _park_ball(data)
            mujoco.mj_forward(model, data)
        elif step_index == first_release:
            shot_index = 1
            _launch_ball(data, first_shot)
            mujoco.mj_forward(model, data)
        elif step_index == first_end:
            shot_index = 0
            _park_ball(data)
            mujoco.mj_forward(model, data)
        elif step_index == second_release:
            second_release_lateral_error = abs(float(data.qpos[1]))
            shot_index = 2 if second_enabled else 0
            _park_ball(data)
            if second_enabled:
                _launch_ball(data, second_shot)
            mujoco.mj_forward(model, data)
        elif step_index == second_end:
            shot_index = 0
            _park_ball(data)
            mujoco.mj_forward(model, data)

        intent_cue = np.zeros(3, dtype=np.float64)
        if world_config.shot_intent_cue_enabled:
            if step_index < first_release:
                intent_cue = first_intent_cue
            elif first_end <= step_index < second_release:
                intent_cue = second_intent_cue
        intercept = _causal_intercept(
            data,
            shot_index=shot_index,
            config=world_config,
            mobility_option_enabled=mobility_enabled,
        )
        if world_config.shot_intent_cue_enabled and intent_cue[2] > 0.5 and shot_index == 0:
            intercept[1:3] = intent_cue[:2]
        observation = _actor_observation(
            data,
            target=intercept,
            joint_target=target,
            previous_action=previous_action,
            shot_index=shot_index,
            step_index=step_index,
            config=world_config,
            intent_cue=intent_cue,
        )
        if policy is None:
            requested_action = np.zeros(18, dtype=np.float64)
        else:
            with torch.no_grad():
                policy_observation = observation
                try:
                    policy_observation_size = int(policy.trunk[0].in_features)
                except (AttributeError, IndexError, TypeError):
                    policy_observation_size = int(observation.shape[0])
                if observation.shape[0] == 77 and policy_observation_size == 74:
                    policy_observation = np.concatenate((observation[:70], observation[73:]))
                elif policy_observation_size != observation.shape[0]:
                    raise ValueError("CPU goalkeeper policy observation contract mismatch")
                mean, _, _ = policy(
                    torch.from_numpy(policy_observation.astype(np.float32))[None, :]
                )
                requested_action = np.tanh(mean[0].numpy()).astype(np.float64)
        if combat_teacher is None:
            requested_action, _ = shape_goalkeeper_action_numpy(
                requested_action=requested_action[None, :],
                root_lateral_position_m=np.asarray((data.qpos[1],), dtype=np.float64),
                root_lateral_velocity_mps=np.asarray((data.qvel[1],), dtype=np.float64),
                root_angular_velocity_rad_s=np.asarray(data.qvel[3:6], dtype=np.float64)[None, :],
                shot_active=np.asarray((shot_index > 0,), dtype=np.bool_),
                config=world_config.agility,
            )
            requested_action = requested_action[0]
        else:
            angular_speed = float(np.linalg.norm(data.qvel[3:6]))
            onset = world_config.agility.angular_guard_onset_rad_s
            ceiling = world_config.agility.angular_guard_ceiling_rad_s
            authority = float(
                np.clip(
                    (ceiling - angular_speed) / max(ceiling - onset, 1.0e-6),
                    world_config.agility.minimum_upper_body_scale,
                    1.0,
                )
            )
            shot_active = shot_index > 0
            recovery_active = (not shot_active) and step_index >= first_release
            anticipation_active = (
                (not shot_active)
                and step_index < first_release
                and world_config.shot_intent_cue_enabled
            )
            predictive_ready = bool(
                (not shot_active)
                and world_config.shot_intent_cue_enabled
                and mobility.anticipatory_arm_reach_enabled
                and intent_cue[2] > 0.5
            )
            requested_action *= authority
            if mobility_enabled:
                requested_action[2:4] *= mobility.effective_waist_plasticity_scale
                requested_action[4:] *= mobility.effective_arm_plasticity_scale
                if mobility.counter_rotation_enabled:
                    guard_fraction = float(
                        np.clip(
                            (angular_speed - onset) / max(ceiling - onset, 1.0e-6),
                            0.0,
                            1.0,
                        )
                    )
                    gain = world_config.agility.counter_rotation_gain
                    requested_action[2] -= gain * guard_fraction * float(data.qvel[3])
                    requested_action[3] -= gain * guard_fraction * float(data.qvel[4])
            lateral_limit = (
                mobility.lateral_command_limit
                if mobility_enabled
                else COMBAT_SIGNED_LATERAL_GATE_LIMIT
            )
            waist_limit = (
                mobility.waist_residual_limit if mobility_enabled else COMBAT_WAIST_RESIDUAL_LIMIT
            )
            requested_action[0] = np.clip(
                requested_action[0],
                -lateral_limit,
                lateral_limit,
            )
            waist_start = 2 if mobility_enabled else 1
            requested_action[waist_start:4] = np.clip(
                requested_action[waist_start:4],
                -waist_limit,
                waist_limit,
            )
            arm_limit = (
                mobility.second_arm_residual_limit
                if mobility_enabled and shot_index == 2
                else (
                    mobility.first_arm_residual_limit
                    if mobility_enabled
                    else (
                        COMBAT_SECOND_SHOT_ARM_RESIDUAL_LIMIT
                        if shot_index == 2
                        else COMBAT_ARM_RESIDUAL_LIMIT
                    )
                )
            )
            requested_action[4:] = np.clip(
                requested_action[4:],
                -arm_limit,
                arm_limit,
            )
            if mobility_enabled:
                ended_shot = previous_teacher_shot_active and not shot_active
                if mobility.teacher_recovery_latch_enabled and ended_shot:
                    teacher_recovery_age_steps = 0
                    teacher_recovery_gate = mobility_teacher_gate
                hold_steps = round(mobility.teacher_recovery_hold_sec / world_config.control_dt_sec)
                decay_steps = round(
                    mobility.teacher_recovery_decay_sec / world_config.control_dt_sec
                )
                total_recovery_steps = hold_steps + decay_steps
                teacher_recovery_active = bool(
                    mobility.teacher_recovery_latch_enabled
                    and 0 <= teacher_recovery_age_steps < total_recovery_steps
                    and not shot_active
                    and not predictive_ready
                )
                if teacher_recovery_active:
                    decay_fraction = float(
                        np.clip(
                            (total_recovery_steps - teacher_recovery_age_steps)
                            / max(decay_steps, 1),
                            0.0,
                            1.0,
                        )
                    )
                    recovery_gate_request = teacher_recovery_gate * decay_fraction
                else:
                    recovery_gate_request = 0.0
                live_gate_request = (
                    float(np.clip(requested_action[1], 0.0, 1.0))
                    if shot_active
                    or (predictive_ready and mobility.predictive_teacher_warmstart_enabled)
                    else 0.0
                )
                if shot_active or (
                    predictive_ready and mobility.predictive_teacher_warmstart_enabled
                ):
                    live_gate_request = max(
                        live_gate_request,
                        mobility.predictive_teacher_gate_floor,
                    )
                desired_gate = (
                    recovery_gate_request if teacher_recovery_active else live_gate_request
                )
                gate_delta = float(
                    np.clip(
                        desired_gate - mobility_teacher_gate,
                        -mobility.teacher_gate_step,
                        mobility.teacher_gate_step,
                    )
                )
                mobility_teacher_gate += mobility.teacher_gate_filter_fraction * gate_delta
                if teacher_recovery_active:
                    teacher_recovery_age_steps += 1
                if (
                    shot_active
                    or predictive_ready
                    or teacher_recovery_age_steps >= total_recovery_steps
                ):
                    teacher_recovery_age_steps = -1
                    teacher_recovery_gate = 0.0
                previous_teacher_shot_active = shot_active
                requested_action[1] = 0.0
                if recovery_active:
                    requested_action[0] = project_recovery_command_numpy(
                        requested=float(requested_action[0]),
                        root_lateral_position_m=float(data.qpos[1]),
                        root_lateral_velocity_mps=float(data.qvel[1]),
                        config=mobility,
                        predictive_threat=predictive_ready,
                    )
            elif recovery_active:
                capture_error = float(data.qpos[1])
                capture_error += COMBAT_RECOVERY_CAPTURE_HORIZON_SEC * float(data.qvel[1])
                recovery_direction = (
                    float(np.sign(capture_error))
                    if abs(capture_error) >= COMBAT_RECOVERY_CENTER_DEADBAND_M
                    else 0.0
                )
                requested_action[0] = recovery_direction * min(
                    abs(float(requested_action[0])),
                    COMBAT_RECOVERY_LATERAL_GATE_LIMIT,
                )
            elif not shot_active and not anticipation_active:
                requested_action[0] = 0.0
            if not shot_active:
                # The noisy public shooter cue may pre-shape the arms.  The
                # waist stays unavailable; a separately bound switch may warm
                # the frozen whole-body teacher from the same cue.
                requested_action[1:4] = 0.0
                if not predictive_ready:
                    requested_action[4:] = 0.0
            if mobility_enabled:
                requested_action[0] = guard_lateral_velocity_numpy(
                    requested=float(requested_action[0]),
                    root_lateral_velocity_mps=float(data.qvel[1]),
                    config=mobility,
                )
        if shot_index == 2:
            requested_action[4:] *= world_config.second_shot_arm_authority_scale
        maximum_requested_action_step = max(
            maximum_requested_action_step,
            float(np.max(np.abs(requested_action - previous_action))),
        )
        action_step_limit = np.full(18, world_config.maximum_action_step, dtype=np.float64)
        action_step_limit[4:] = world_config.maximum_arm_action_step
        action_filter = np.full(18, world_config.action_filter_fraction, dtype=np.float64)
        action_filter[4:] = world_config.arm_action_filter_fraction
        action_delta = np.clip(
            requested_action - previous_action,
            -action_step_limit,
            action_step_limit,
        )
        action = previous_action + action_filter * action_delta
        maximum_applied_action_step = max(
            maximum_applied_action_step,
            float(np.max(np.abs(action - previous_action))),
        )
        gravity = _gravity_projection(np.asarray(data.qpos[3:7], dtype=np.float64))
        loco_observation = np.zeros(96, dtype=np.float32)
        loco_observation[:3] = data.qvel[3:6]
        loco_observation[3:6] = gravity
        loco_observation[7] = action[0] * world_config.maximum_lateral_command_mps
        loco_observation[9:38] = data.qpos[7:36][loco_to_motor] - loco_default
        loco_observation[38:67] = data.qvel[6:35][loco_to_motor]
        loco_observation[67:96] = loco_action
        with torch.no_grad():
            encoded = locomotion.normalizer.forward(
                torch.from_numpy(np.clip(loco_observation, -100.0, 100.0))[None, :]
            )
            sequence, (hidden, cell) = locomotion.rnn.forward__0(
                encoded.unsqueeze(0), (hidden, cell)
            )
            loco_action = np.clip(
                locomotion.actor.forward(sequence.squeeze(0))[0].numpy(), -100, 100
            )
        target.fill(0.0)
        target[loco_to_motor] = 0.25 * loco_action + loco_default
        if combat_teacher is not None:
            quaternion = np.asarray(data.qpos[3:7], dtype=np.float64)
            torso_position = np.asarray(data.xpos[int(model.body("torso_link").id)])
            ball_relative = _rotate_inverse(
                quaternion,
                (
                    intercept
                    if combat_teacher.intercept_conditioning_enabled
                    else np.asarray(data.qpos[36:39], dtype=np.float64)
                )
                - torso_position,
            )
            if (
                shot_index <= 0
                and predictive_ready
                and mobility.predictive_teacher_warmstart_enabled
            ):
                proxy_world = np.asarray(
                    (
                        world_config.keeper_x_m
                        - (0.08 if combat_teacher.intercept_conditioning_enabled else 1.0),
                        intent_cue[0],
                        intent_cue[1],
                    ),
                    dtype=np.float64,
                )
                ball_relative = _rotate_inverse(
                    quaternion,
                    proxy_world - torso_position,
                )
            elif shot_index <= 0:
                ball_relative.fill(0.0)
            teacher_observation = np.concatenate(
                (
                    ball_relative,
                    _rotate_inverse(quaternion, np.asarray(data.qvel[3:6], dtype=np.float64))
                    * 0.25,
                    gravity,
                    np.asarray(data.qpos[7:36], dtype=np.float64) - combat_teacher.default_qpos,
                    np.asarray(data.qvel[6:35], dtype=np.float64) * 0.05,
                    previous_teacher_action,
                )
            )
            if teacher_observation.shape != (96,) or not np.all(np.isfinite(teacher_observation)):
                raise RuntimeError("CPU goalkeeper combat teacher observation changed")
            teacher_history[:-1] = teacher_history[1:]
            teacher_history[-1] = np.clip(teacher_observation, -100.0, 100.0)
            with torch.no_grad():
                teacher_action = combat_teacher.teacher(
                    torch.from_numpy(teacher_history.reshape(1, -1))
                )[0].numpy()
            previous_teacher_action = np.asarray(
                np.clip(teacher_action, -4.0, 4.0), dtype=np.float32
            )
            teacher_target = combat_teacher.default_qpos + 0.25 * previous_teacher_action
            raw_blend = mobility_teacher_gate if mobility_enabled else abs(action[0])
            teacher_visible = bool(shot_index > 0 or predictive_ready or teacher_recovery_active)
            blend = (
                float(np.clip(raw_blend, 0.0, 1.0)) * combat_teacher.maximum_blend
                if teacher_visible
                else 0.0
            )
            joint_blend = np.clip(
                blend * combat_teacher.joint_group_scale,
                0.0,
                1.0 if mobility_enabled else 0.50,
            )
            stable_target = target.copy()
            target += joint_blend * (teacher_target - target)
            if combat_teacher.runtime_reach_atlas is not None:
                from rosclaw_soccer.training.goalkeeper_reach import (
                    task_space_reach_from_target_numpy,
                )

                normalized_reach = task_space_reach_from_target_numpy(
                    target_relative=ball_relative[None, :],
                    model=combat_teacher.runtime_reach_atlas,
                )[0]
                reach_ready = np.zeros(29, dtype=np.float64)
                reach_ready[loco_to_motor] = loco_default
                reach_limits = np.asarray(
                    tuple(combat_teacher.runtime_reach_atlas.effective_arm_limits_rad) * 2,
                    dtype=np.float64,
                )
                reach_target = reach_ready[15:29] + normalized_reach * reach_limits
                reach_visible = bool(shot_index > 0 or predictive_ready)
                reach_blend = (
                    combat_teacher.runtime_reach_blend * mobility_teacher_gate
                    if reach_visible
                    else 0.0
                )
                target[15:29] += reach_blend * (reach_target - target[15:29])
                maximum_applied_runtime_reach_blend = max(
                    maximum_applied_runtime_reach_blend, reach_blend
                )
            if mobility_enabled:
                raw_teacher_delta = target - stable_target
                filtered_groups: list[np.ndarray] = []
                group_contracts = mobility.teacher_target_filter_contracts
                for start, end, step_limit, filter_fraction in group_contracts:
                    previous = previous_teacher_target_delta[start:end]
                    delta_step = np.clip(
                        raw_teacher_delta[start:end] - previous,
                        -step_limit,
                        step_limit,
                    )
                    filtered_groups.append(previous + filter_fraction * delta_step)
                previous_teacher_target_delta[:] = np.concatenate(filtered_groups)
                target[:] = stable_target + previous_teacher_target_delta
            target[:] = np.clip(target, model.jnt_range[1:30, 0], model.jnt_range[1:30, 1])
            maximum_applied_teacher_blend = max(
                maximum_applied_teacher_blend, float(np.max(joint_blend))
            )
        residual = action[1:] * residual_limits * world_config.residual_scale
        residual[3:] *= world_config.arm_residual_scale_multiplier
        target[residual_indices] += residual
        target[residual_indices] = np.clip(
            target[residual_indices],
            model.jnt_range[13:30, 0],
            model.jnt_range[13:30, 1],
        )
        pre_ball_velocity = np.asarray(data.qvel[35:38], dtype=np.float64).copy()
        ball_contact = False
        hand_contact = False
        for _ in range(world_config.physics_substeps):
            position_torque = kp * (target - data.qpos[7:36])
            if mobility_enabled:
                from rosclaw_soccer.training.goalkeeper_mobility_option import (
                    substep_upper_body_authority_numpy,
                )

                substep_authority = substep_upper_body_authority_numpy(
                    root_angular_velocity_rad_s=np.asarray(data.qvel[3:6]),
                    config=mobility,
                )
                position_torque[12:] *= substep_authority
            torque = position_torque - kd * data.qvel[6:35]
            from rosclaw_soccer.training.joint_guard import project_joint_safe_torque_numpy

            torque, guard_active = project_joint_safe_torque_numpy(
                joint_position=np.asarray(data.qpos[7:36]),
                joint_velocity=np.asarray(data.qvel[6:35]),
                commanded_torque=torque,
                joint_ranges=np.asarray(model.jnt_range[1:30]),
                limited=model.jnt_limited[1:30].astype(bool),
                margin_rad=0.05,
                prediction_horizon_sec=0.08,
                boundary_kp=80.0,
                boundary_kd=6.0,
            )
            joint_guard_active_steps += int(guard_active)
            clipped_torque = np.clip(torque, -limits, limits)
            torque_limit_violation |= bool(np.any(np.abs(clipped_torque) > limits + 1e-9))
            maximum_torque_fraction = max(
                maximum_torque_fraction, float(np.max(np.abs(clipped_torque) / limits))
            )
            data.ctrl[:] = clipped_torque
            mujoco.mj_step(model, data)
            substep_contact, substep_hand_contact = _robot_ball_contacts(model, data)
            ball_contact |= substep_contact
            hand_contact |= substep_hand_contact
        post_ball_velocity = np.asarray(data.qvel[35:38], dtype=np.float64)
        true_save = bool(
            ball_contact
            and pre_ball_velocity[0] > 0.25
            and post_ball_velocity[0] < 0.65 * pre_ball_velocity[0]
        )
        quaternion = np.asarray(data.qpos[3:7], dtype=np.float64)
        upright = 2.0 * (quaternion[0] ** 2 + quaternion[3] ** 2) - 1.0
        joint_velocity = np.asarray(data.qvel[6:35], dtype=np.float64).copy()
        joint_acceleration = (
            joint_velocity - previous_joint_velocity
        ) / world_config.control_dt_sec
        selected_target = (
            np.asarray(second_shot["target"])
            if shot_index == 2
            else np.asarray(first_shot["target"])
        )
        if shot_index == 0:
            ready_y = (
                float(
                    np.clip(
                        data.qpos[1] + 0.28 * data.qvel[1],
                        world_config.target_y_range_m[0],
                        world_config.target_y_range_m[1],
                    )
                )
                if mobility_enabled
                else 0.0
            )
            selected_target = np.asarray((data.qpos[0] - 0.03, ready_y, 0.82), dtype=np.float64)
        left_hand_position = np.asarray(data.geom_xpos[left_hand_geom], dtype=np.float64)
        right_hand_position = np.asarray(data.geom_xpos[right_hand_geom], dtype=np.float64)
        left_hand_relative = left_hand_position - np.asarray(data.qpos[:3])
        right_hand_relative = right_hand_position - np.asarray(data.qpos[:3])
        maximum_hand_displacement = max(
            maximum_hand_displacement,
            float(np.linalg.norm(left_hand_relative - ready_left_hand_relative)),
            float(np.linalg.norm(right_hand_relative - ready_right_hand_relative)),
        )
        maximum_hand_speed = max(
            maximum_hand_speed,
            float(
                np.linalg.norm(left_hand_relative - previous_left_hand_relative)
                / world_config.control_dt_sec
            ),
            float(
                np.linalg.norm(right_hand_relative - previous_right_hand_relative)
                / world_config.control_dt_sec
            ),
        )
        previous_left_hand_relative = left_hand_relative.copy()
        previous_right_hand_relative = right_hand_relative.copy()
        if shot_index > 0:
            active_flight_steps += 1
            hand_midpoint = 0.5 * (left_hand_position + right_hand_position)
            bimanual_distance = max(
                float(np.linalg.norm(left_hand_position - selected_target)),
                float(np.linalg.norm(right_hand_position - selected_target)),
            )
            if (
                abs(float(selected_target[1] - hand_midpoint[1])) <= 0.42
                and float(selected_target[2]) >= 0.62
                and bimanual_distance <= 0.42
            ):
                bimanual_reach_steps += 1
        event_shot_index = shot_index
        result = task.step(
            GoalkeeperStepBatch(
                time_sec=np.asarray(((step_index + 1) * world_config.control_dt_sec,)),
                ball_position_m=np.asarray(data.qpos[36:39], dtype=np.float64)[None, :],
                ball_velocity_mps=post_ball_velocity[None, :],
                intercept_target_m=selected_target[None, :],
                left_hand_position_m=left_hand_position[None, :],
                right_hand_position_m=right_hand_position[None, :],
                pelvis_height_m=np.asarray((data.qpos[2],)),
                root_linear_velocity_mps=np.asarray(data.qvel[:3], dtype=np.float64)[None, :],
                root_angular_velocity_rad_s=np.asarray(data.qvel[3:6], dtype=np.float64)[None, :],
                upright_projection=np.asarray((upright,)),
                action=action[None, :],
                previous_action=previous_action[None, :],
                joint_acceleration_rad_s2=joint_acceleration[None, :],
                applied_torque_nm=np.asarray(data.ctrl, dtype=np.float64)[None, :],
                ball_contact=np.asarray((ball_contact,), dtype=np.bool_),
                hand_contact=np.asarray((hand_contact,), dtype=np.bool_),
                true_save=np.asarray((true_save,), dtype=np.bool_),
                shot_index=np.asarray((shot_index,), dtype=np.int64),
            )
        )
        if true_save:
            shot_index = 0
        cumulative_reward += float(result.total[0])
        previous_action = action.copy()
        previous_joint_velocity = joint_velocity
        minimum_pelvis_height = min(minimum_pelvis_height, float(data.qpos[2]))
        maximum_root_speed = max(maximum_root_speed, float(np.linalg.norm(data.qvel[:3])))
        maximum_root_angular_speed = max(
            maximum_root_angular_speed, float(np.linalg.norm(data.qvel[3:6]))
        )
        maximum_lateral_displacement = max(maximum_lateral_displacement, abs(float(data.qpos[1])))
        maximum_lateral_speed = max(maximum_lateral_speed, abs(float(data.qvel[1])))
        joint_positions = np.asarray(data.qpos[7:36])
        joint_limit_violation |= bool(
            np.any(joint_positions < model.jnt_range[1:30, 0] - 1e-5)
            or np.any(joint_positions > model.jnt_range[1:30, 1] + 1e-5)
        )
        if record_trajectory:
            recorded_qpos.append(np.asarray(data.qpos, dtype=np.float64).copy())
            recorded_qvel.append(np.asarray(data.qvel, dtype=np.float64).copy())
            recorded_action.append(np.asarray(action, dtype=np.float64).copy())
            recorded_shot_index.append(event_shot_index)
            recorded_left_hand.append(left_hand_position.copy())
            recorded_right_hand.append(right_hand_position.copy())
            recorded_time.append((step_index + 1) * world_config.control_dt_sec)

    return _EpisodeResult(
        seed=seed,
        reward=cumulative_reward,
        first_save=bool(task.first_save[0]),
        first_hand_save=bool(task.first_hand_save[0]),
        recovered=bool(task.recovered_after_first[0]),
        second_attempt_save=bool(task.second_attempt_save[0]),
        second_attempt_hand_save=bool(task.second_attempt_hand_save[0]),
        second_save=bool(task.second_save[0]),
        second_hand_save=bool(task.second_hand_save[0]),
        failed=bool(task.phase[0] == 7),
        finite_state=bool(
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(data.ctrl))
        ),
        joint_limit_violation=joint_limit_violation,
        torque_limit_violation=torque_limit_violation,
        maximum_torque_fraction=maximum_torque_fraction,
        maximum_requested_actor_action_step=maximum_requested_action_step,
        maximum_applied_actor_action_step=maximum_applied_action_step,
        minimum_pelvis_height_m=minimum_pelvis_height,
        maximum_root_speed_mps=maximum_root_speed,
        maximum_root_angular_speed_rad_s=maximum_root_angular_speed,
        maximum_lateral_displacement_m=maximum_lateral_displacement,
        maximum_lateral_speed_mps=maximum_lateral_speed,
        maximum_hand_displacement_m=maximum_hand_displacement,
        maximum_hand_speed_mps=maximum_hand_speed,
        second_release_lateral_error_m=second_release_lateral_error,
        joint_guard_active_fraction=joint_guard_active_steps
        / (world_config.episode_steps * world_config.physics_substeps),
        bimanual_reach_fraction=bimanual_reach_steps / max(1, active_flight_steps),
        maximum_applied_teacher_blend=maximum_applied_teacher_blend,
        maximum_applied_runtime_reach_blend=(maximum_applied_runtime_reach_blend),
        trajectory=(
            {
                "time": np.asarray(recorded_time, dtype=np.float64),
                "qpos": np.asarray(recorded_qpos, dtype=np.float64),
                "qvel": np.asarray(recorded_qvel, dtype=np.float64),
                "action": np.asarray(recorded_action, dtype=np.float64),
                "shot_index": np.asarray(recorded_shot_index, dtype=np.int64),
                "left_hand": np.asarray(recorded_left_hand, dtype=np.float64),
                "right_hand": np.asarray(recorded_right_hand, dtype=np.float64),
                "first_target": np.asarray(first_shot["target"], dtype=np.float64),
                "second_target": np.asarray(second_shot["target"], dtype=np.float64),
                "first_intent_visible": np.asarray((first_intent_cue[2] > 0.5,), dtype=np.bool_),
                "second_intent_visible": np.asarray((second_intent_cue[2] > 0.5,), dtype=np.bool_),
                "second_enabled": np.asarray((second_enabled,), dtype=np.bool_),
                "first_arrival_sec": np.asarray(
                    (world_config.first_shot_release_sec + float(first_shot["flight"]),),
                    dtype=np.float64,
                ),
                "second_arrival_sec": np.asarray(
                    (world_config.second_shot_release_sec + float(second_shot["flight"]),),
                    dtype=np.float64,
                ),
            }
            if record_trajectory
            else None
        ),
    )


def _summarize(name: str, episodes: list[_EpisodeResult]) -> dict[str, Any]:
    count = len(episodes)
    return {
        "policy": name,
        "episodes": count,
        "mean_episode_reward": float(np.mean([item.reward for item in episodes])),
        "first_save_rate": sum(item.first_save for item in episodes) / count,
        "first_hand_save_rate": sum(item.first_hand_save for item in episodes) / count,
        "recovery_rate": sum(item.recovered for item in episodes) / count,
        "second_attempt_save_rate": sum(item.second_attempt_save for item in episodes) / count,
        "second_attempt_hand_save_rate": sum(item.second_attempt_hand_save for item in episodes)
        / count,
        "second_save_rate": sum(item.second_save for item in episodes) / count,
        "second_hand_save_rate": sum(item.second_hand_save for item in episodes) / count,
        "failed_rate": sum(item.failed for item in episodes) / count,
        "failed_seeds": [item.seed for item in episodes if item.failed],
        "challenge_failures": _challenge_failures(episodes),
        "finite_state": all(item.finite_state for item in episodes),
        "joint_limit_violation_rate": sum(item.joint_limit_violation for item in episodes) / count,
        "torque_limit_violation_rate": sum(item.torque_limit_violation for item in episodes)
        / count,
        "maximum_torque_fraction": max(item.maximum_torque_fraction for item in episodes),
        "maximum_requested_actor_action_step": max(
            item.maximum_requested_actor_action_step for item in episodes
        ),
        "maximum_applied_actor_action_step": max(
            item.maximum_applied_actor_action_step for item in episodes
        ),
        "minimum_pelvis_height_m": min(item.minimum_pelvis_height_m for item in episodes),
        "maximum_root_speed_mps": max(item.maximum_root_speed_mps for item in episodes),
        "maximum_root_angular_speed_rad_s": max(
            item.maximum_root_angular_speed_rad_s for item in episodes
        ),
        "p95_root_speed_mps": float(
            np.quantile([item.maximum_root_speed_mps for item in episodes], 0.95)
        ),
        "p95_root_angular_speed_rad_s": float(
            np.quantile([item.maximum_root_angular_speed_rad_s for item in episodes], 0.95)
        ),
        "mean_maximum_lateral_displacement_m": float(
            np.mean([item.maximum_lateral_displacement_m for item in episodes])
        ),
        "mean_maximum_lateral_speed_mps": float(
            np.mean([item.maximum_lateral_speed_mps for item in episodes])
        ),
        "mean_maximum_hand_displacement_m": float(
            np.mean([item.maximum_hand_displacement_m for item in episodes])
        ),
        "mean_maximum_hand_speed_mps": float(
            np.mean([item.maximum_hand_speed_mps for item in episodes])
        ),
        "p95_maximum_hand_speed_mps": float(
            np.quantile([item.maximum_hand_speed_mps for item in episodes], 0.95)
        ),
        "mean_second_release_lateral_error_m": float(
            np.mean([item.second_release_lateral_error_m for item in episodes])
        ),
        "mean_joint_guard_active_fraction": float(
            np.mean([item.joint_guard_active_fraction for item in episodes])
        ),
        "mean_bimanual_reach_fraction": float(
            np.mean([item.bimanual_reach_fraction for item in episodes])
        ),
        "maximum_applied_teacher_blend": max(
            item.maximum_applied_teacher_blend for item in episodes
        ),
        "maximum_applied_runtime_reach_blend": max(
            item.maximum_applied_runtime_reach_blend for item in episodes
        ),
        "phase_action_audit": _phase_action_audit(episodes),
        "shot_strata_audit": _shot_strata_audit(episodes),
    }


def _challenge_failures(episodes: list[_EpisodeResult]) -> list[dict[str, Any]] | None:
    """Return replayable misses, distinct from physical safety failures."""

    if not episodes or any(item.trajectory is None for item in episodes):
        return None
    failures: list[dict[str, Any]] = []
    lateral_edges = (0.36, 0.72)
    height_edges = (0.60, 1.10)
    lateral_names = ("central", "wide", "far_corner")
    height_names = ("low", "mid", "high")
    for episode in episodes:
        trajectory = episode.trajectory
        if trajectory is None:
            continue
        for phase, saved in ((1, episode.first_save), (2, episode.second_attempt_save)):
            if saved or (phase == 2 and not bool(np.asarray(trajectory["second_enabled"])[0])):
                continue
            target = np.asarray(
                trajectory["first_target" if phase == 1 else "second_target"],
                dtype=np.float64,
            )
            lateral_index = int(np.digitize(abs(float(target[1])), lateral_edges, right=True))
            height_index = int(np.digitize(float(target[2]), height_edges, right=True))
            cue_visible = bool(
                np.asarray(
                    trajectory["first_intent_visible" if phase == 1 else "second_intent_visible"]
                )[0]
            )
            failures.append(
                {
                    "seed": episode.seed,
                    "shot": phase,
                    "stratum": f"{lateral_names[lateral_index]}_{height_names[height_index]}",
                    "target_xyz_m": [float(value) for value in target],
                    "intent_cue_visible": cue_visible,
                    **_miss_diagnostics(trajectory, target=target, phase=phase),
                }
            )
    return failures


def _miss_diagnostics(
    trajectory: dict[str, np.ndarray],
    *,
    target: np.ndarray,
    phase: int,
) -> dict[str, float]:
    """Explain whether a miss came from timing, lateral motion, or hand reach."""

    time = np.asarray(trajectory["time"], dtype=np.float64)
    qpos = np.asarray(trajectory["qpos"], dtype=np.float64)
    action = np.asarray(trajectory["action"], dtype=np.float64)
    shot_index = np.asarray(trajectory["shot_index"], dtype=np.int64)
    left = np.asarray(trajectory["left_hand"], dtype=np.float64)
    right = np.asarray(trajectory["right_hand"], dtype=np.float64)
    arrival = float(
        np.asarray(trajectory["first_arrival_sec" if phase == 1 else "second_arrival_sec"])[0]
    )
    arrival_index = int(np.argmin(np.abs(time - arrival)))
    phase_mask = shot_index == phase
    if np.any(phase_mask):
        minimum_hand_distance = float(
            min(
                np.min(np.linalg.norm(left[phase_mask] - target, axis=1)),
                np.min(np.linalg.norm(right[phase_mask] - target, axis=1)),
            )
        )
        mean_lateral_action = float(np.mean(action[phase_mask, 0]))
    else:
        minimum_hand_distance = math.inf
        mean_lateral_action = 0.0
    root_lateral = float(qpos[arrival_index, 1])
    hand_distance_at_arrival = float(
        min(
            np.linalg.norm(left[arrival_index] - target),
            np.linalg.norm(right[arrival_index] - target),
        )
    )
    return {
        "arrival_time_sec": arrival,
        "root_lateral_at_arrival_m": root_lateral,
        "root_lateral_error_at_arrival_m": float(target[1] - root_lateral),
        "minimum_hand_distance_m": minimum_hand_distance,
        "hand_distance_at_arrival_m": hand_distance_at_arrival,
        "mean_lateral_action": mean_lateral_action,
        # The pinned locomotion prior uses a deployment-frame command whose
        # lateral sign is opposite MuJoCo world y.  Keep both contracts
        # explicit so failure mining does not mistake correct motion for a
        # wrong-way command.
        "command_sign_contract_alignment": float(
            -np.sign(mean_lateral_action) * np.sign(float(target[1]))
        ),
        "root_motion_target_alignment": float(np.sign(root_lateral) * np.sign(float(target[1]))),
    }


def _shot_strata_audit(episodes: list[_EpisodeResult]) -> dict[str, Any] | None:
    """Expose where saves fail instead of hiding difficulty in one mean."""

    if not episodes or any(item.trajectory is None for item in episodes):
        return None
    lateral_edges = (0.36, 0.72)
    height_edges = (0.60, 1.10)
    lateral_names = ("central", "wide", "far_corner")
    height_names = ("low", "mid", "high")
    payload: dict[str, Any] = {
        "lateral_absolute_edges_m": lateral_edges,
        "height_edges_m": height_edges,
        "phases": {},
    }
    for phase, phase_name in ((1, "first_shot"), (2, "second_shot")):
        cells: dict[str, dict[str, int | float]] = {}
        cue_visible_attempts = 0
        cue_visible_saves = 0
        cue_hidden_attempts = 0
        cue_hidden_saves = 0
        for episode in episodes:
            trajectory = episode.trajectory
            if trajectory is None:
                continue
            if phase == 2 and not bool(np.asarray(trajectory["second_enabled"])[0]):
                continue
            target = np.asarray(
                trajectory["first_target" if phase == 1 else "second_target"],
                dtype=np.float64,
            )
            saved = episode.first_save if phase == 1 else episode.second_attempt_save
            hand_saved = episode.first_hand_save if phase == 1 else episode.second_attempt_hand_save
            lateral_index = int(np.digitize(abs(float(target[1])), lateral_edges, right=True))
            height_index = int(np.digitize(float(target[2]), height_edges, right=True))
            key = f"{lateral_names[lateral_index]}_{height_names[height_index]}"
            cell = cells.setdefault(
                key,
                {"attempts": 0, "saves": 0, "hand_saves": 0, "save_rate": 0.0},
            )
            cell["attempts"] = int(cell["attempts"]) + 1
            cell["saves"] = int(cell["saves"]) + int(saved)
            cell["hand_saves"] = int(cell["hand_saves"]) + int(hand_saved)
            visible = bool(
                np.asarray(
                    trajectory["first_intent_visible" if phase == 1 else "second_intent_visible"]
                )[0]
            )
            if visible:
                cue_visible_attempts += 1
                cue_visible_saves += int(saved)
            else:
                cue_hidden_attempts += 1
                cue_hidden_saves += int(saved)
        for cell in cells.values():
            cell["save_rate"] = float(int(cell["saves"]) / int(cell["attempts"]))
        payload["phases"][phase_name] = {
            "cells": cells,
            "visible_intent": {
                "attempts": cue_visible_attempts,
                "save_rate": cue_visible_saves / max(1, cue_visible_attempts),
            },
            "hidden_intent": {
                "attempts": cue_hidden_attempts,
                "save_rate": cue_hidden_saves / max(1, cue_hidden_attempts),
            },
        }
    return payload


def _phase_action_audit(episodes: list[_EpisodeResult]) -> dict[str, Any] | None:
    """Summarize what the actor actually applied in each causal shot phase."""

    if not episodes or any(item.trajectory is None for item in episodes):
        return None
    phase_payload: dict[str, dict[str, float | int]] = {}
    phase_means: dict[int, np.ndarray] = {}
    for phase, name in ((1, "first_shot"), (2, "second_shot")):
        action_batches: list[np.ndarray] = []
        angular_batches: list[np.ndarray] = []
        minimum_hand_distances: list[float] = []
        for episode in episodes:
            trajectory = episode.trajectory
            if trajectory is None:
                continue
            phase_mask = np.asarray(trajectory["shot_index"]) == phase
            if not np.any(phase_mask):
                continue
            actions = np.asarray(trajectory["action"], dtype=np.float64)[phase_mask]
            qvel = np.asarray(trajectory["qvel"], dtype=np.float64)[phase_mask]
            left = np.asarray(trajectory["left_hand"], dtype=np.float64)[phase_mask]
            right = np.asarray(trajectory["right_hand"], dtype=np.float64)[phase_mask]
            target_key = "first_target" if phase == 1 else "second_target"
            target = np.asarray(trajectory[target_key], dtype=np.float64)
            action_batches.append(actions)
            angular_batches.append(np.linalg.norm(qvel[:, 3:6], axis=1))
            minimum_hand_distances.append(
                float(
                    min(
                        np.min(np.linalg.norm(left - target, axis=1)),
                        np.min(np.linalg.norm(right - target, axis=1)),
                    )
                )
            )
        if not action_batches:
            continue
        actions = np.concatenate(action_batches, axis=0)
        angular = np.concatenate(angular_batches, axis=0)
        arm_norm = np.linalg.norm(actions[:, 4:], axis=1)
        arm_envelope_threshold = (
            COMBAT_ARM_RESIDUAL_LIMIT - 0.005
            if phase == 1
            else COMBAT_SECOND_SHOT_ARM_RESIDUAL_LIMIT - 0.005
        )
        phase_means[phase] = np.mean(actions, axis=0)
        phase_payload[name] = {
            "control_samples": int(actions.shape[0]),
            "mean_absolute_lateral_action": float(np.mean(np.abs(actions[:, 0]))),
            "p95_absolute_lateral_action": float(np.quantile(np.abs(actions[:, 0]), 0.95)),
            "mean_arm_action_l2": float(np.mean(arm_norm)),
            "p95_arm_action_l2": float(np.quantile(arm_norm, 0.95)),
            "arm_envelope_fraction": float(
                np.mean(np.abs(actions[:, 4:]) >= arm_envelope_threshold)
            ),
            "mean_root_angular_speed_rad_s": float(np.mean(angular)),
            "mean_minimum_hand_distance_m": float(np.mean(minimum_hand_distances)),
        }
    separation = None
    if 1 in phase_means and 2 in phase_means:
        separation = float(np.linalg.norm(phase_means[2] - phase_means[1]))
    return {
        "phases": phase_payload,
        "first_second_mean_action_l2_separation": separation,
        "source": "RECORDED_APPLIED_ACTIONS_AND_CPU_MUJOCO_STATE",
    }


def run_goalkeeper_cpu_exam(
    *,
    asset_root: Path,
    locomotion_policy_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    config: GoalkeeperCPUExamConfig | None = None,
    world_config: GoalkeeperMJWarpConfig | None = None,
    parent_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Run paired CPU physics rollouts and make a fail-closed promotion decision."""

    import mujoco
    import torch
    from torch import nn

    from rosclaw_soccer.training.goalkeeper_physics_ppo import (
        _build_actor_critic,
        _load_actor_critic_state,
    )
    from rosclaw_soccer.world.field import build_g1_stadium_model, g1_stadium_scene_hash

    active = config or GoalkeeperCPUExamConfig()
    root = asset_root.expanduser().resolve()
    active_world = world_config or GoalkeeperMJWarpConfig(environment_count=1)
    checkpoint_file = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    training_config = checkpoint.get("training_config", {})
    if not isinstance(training_config, dict):
        raise ValueError("goalkeeper CPU exam checkpoint training config is invalid")
    combat_teacher: _CombatTeacherRuntime | None = None
    combat_checkout = training_config.get("combat_teacher_checkout")
    combat_checkpoint = training_config.get("combat_teacher_checkpoint")
    if (combat_checkout is None) != (combat_checkpoint is None):
        raise ValueError("goalkeeper CPU exam combat teacher provenance is incomplete")
    if combat_checkout is not None:
        from rosclaw_soccer.training.goalkeeper_combat_teacher import (
            OFFICIAL_GOALKEEPER_BLEND_GROUP_SCALE,
            OFFICIAL_GOALKEEPER_DEFAULT_QPOS,
            load_official_goalkeeper_teacher,
        )

        teacher, teacher_report = load_official_goalkeeper_teacher(
            checkout=Path(str(combat_checkout)),
            checkpoint=Path(str(combat_checkpoint)),
            device=torch.device("cpu"),
        )
        mobility_option_enabled = bool(training_config.get("mobility_option_enabled", False))
        mobility_residual_plasticity_scale = float(
            training_config.get("mobility_residual_plasticity_scale", 0.0)
        )
        raw_waist_plasticity = training_config.get("mobility_waist_residual_plasticity_scale")
        raw_arm_plasticity = training_config.get("mobility_arm_residual_plasticity_scale")
        mobility_config = GoalkeeperMobilityOptionConfig(
            lateral_command_limit=float(
                training_config.get("mobility_lateral_command_limit", 0.75)
            ),
            recovery_command_limit=float(
                training_config.get("mobility_recovery_command_limit", 0.55)
            ),
            residual_plasticity_scale=mobility_residual_plasticity_scale,
            waist_residual_plasticity_scale=(
                None if raw_waist_plasticity is None else float(raw_waist_plasticity)
            ),
            arm_residual_plasticity_scale=(
                None if raw_arm_plasticity is None else float(raw_arm_plasticity)
            ),
            teacher_lower_body_scale=float(
                training_config.get("mobility_teacher_lower_body_scale", 0.25)
            ),
            teacher_waist_scale=float(training_config.get("mobility_teacher_waist_scale", 0.80)),
            teacher_arm_scale=float(training_config.get("mobility_teacher_arm_scale", 1.00)),
            predictive_teacher_gate_floor=float(
                training_config.get("mobility_predictive_teacher_gate_floor", 0.0)
            ),
            teacher_lower_body_target_step_rad=float(
                training_config.get("mobility_teacher_lower_body_target_step_rad", 0.08)
            ),
            teacher_lower_body_target_filter_fraction=float(
                training_config.get(
                    "mobility_teacher_lower_body_target_filter_fraction", 0.35
                )
            ),
            teacher_waist_target_step_rad=float(
                training_config.get("mobility_teacher_waist_target_step_rad", 0.05)
            ),
            teacher_waist_target_filter_fraction=float(
                training_config.get("mobility_teacher_waist_target_filter_fraction", 0.25)
            ),
            teacher_arm_target_step_rad=float(
                training_config.get("mobility_teacher_arm_target_step_rad", 0.045)
            ),
            teacher_arm_target_filter_fraction=float(
                training_config.get("mobility_teacher_arm_target_filter_fraction", 0.15)
            ),
            counter_rotation_enabled=bool(
                training_config.get("mobility_counter_rotation_enabled", False)
            ),
            anticipatory_arm_reach_enabled=bool(
                training_config.get("mobility_anticipatory_arm_reach_enabled", False)
            ),
            predictive_teacher_warmstart_enabled=bool(
                training_config.get("mobility_predictive_teacher_warmstart_enabled", False)
            ),
            teacher_recovery_latch_enabled=bool(
                training_config.get("mobility_teacher_recovery_latch_enabled", False)
            ),
            teacher_recovery_hold_sec=float(
                training_config.get("mobility_teacher_recovery_hold_sec", 0.24)
            ),
            teacher_recovery_decay_sec=float(
                training_config.get("mobility_teacher_recovery_decay_sec", 0.60)
            ),
            lateral_velocity_guard_enabled=bool(
                training_config.get("mobility_lateral_velocity_guard_enabled", False)
            ),
            substep_upper_body_guard_enabled=bool(
                training_config.get("mobility_substep_upper_body_guard_enabled", False)
            ),
            substep_upper_body_guard_onset_rad_s=float(
                training_config.get("mobility_substep_upper_body_guard_onset_rad_s", 1.80)
            ),
            substep_upper_body_guard_ceiling_rad_s=float(
                training_config.get("mobility_substep_upper_body_guard_ceiling_rad_s", 2.80)
            ),
            substep_upper_body_minimum_position_scale=float(
                training_config.get("mobility_substep_upper_body_minimum_position_scale", 0.05)
            ),
        )
        maximum_blend = float(training_config.get("maximum_combat_teacher_blend", 0.0))
        maximum_allowed_blend = 1.0 if mobility_option_enabled else 0.50
        if not math.isfinite(maximum_blend) or not 0.05 <= maximum_blend <= maximum_allowed_blend:
            raise ValueError("goalkeeper CPU exam combat teacher blend is invalid")
        runtime_reach_atlas = None
        runtime_reach_blend = float(training_config.get("runtime_task_space_reach_blend", 0.0))
        if bool(training_config.get("runtime_task_space_reach_enabled", False)):
            if (
                not mobility_option_enabled
                or not bool(training_config.get("task_space_reach_atlas_enabled", False))
                or not 0.05 <= runtime_reach_blend <= 0.85
            ):
                raise ValueError("goalkeeper CPU runtime reach boundary is invalid")
            from rosclaw_soccer.training.goalkeeper_reach import (
                GoalkeeperReachConfig,
                build_g1_task_space_reach_atlas,
            )

            runtime_reach_atlas = build_g1_task_space_reach_atlas(
                root,
                config=GoalkeeperReachConfig(
                    damping=0.12,
                    reach_gain=0.95,
                    maximum_position_error_m=0.75,
                    support_arm_scale=0.60,
                    central_support_scale=0.95,
                    residual_scale=min(
                        1.0,
                        active_world.residual_scale * active_world.arm_residual_scale_multiplier,
                    ),
                    arm_authority_scale=active_world.agility.arm_authority_scale,
                ),
            )
        combat_teacher = _CombatTeacherRuntime(
            teacher=teacher,
            report=teacher_report,
            maximum_blend=maximum_blend,
            default_qpos=np.asarray(OFFICIAL_GOALKEEPER_DEFAULT_QPOS, dtype=np.float64),
            joint_group_scale=np.asarray(
                mobility_config.teacher_group_scale
                if mobility_option_enabled
                else OFFICIAL_GOALKEEPER_BLEND_GROUP_SCALE,
                dtype=np.float64,
            ),
            mobility_option_enabled=mobility_option_enabled,
            mobility_option_config=mobility_config,
            intercept_conditioning_enabled=bool(
                training_config.get("combat_teacher_intercept_conditioning_enabled", False)
            ),
            runtime_reach_atlas=runtime_reach_atlas,
            runtime_reach_blend=runtime_reach_blend,
        )
    candidate = _build_actor_critic(
        torch,
        nn,
        int(checkpoint["observation_size"]),
        int(checkpoint["action_size"]),
        int(checkpoint["hidden_size"]),
    )
    _load_actor_critic_state(candidate, checkpoint["state_dict"])
    candidate.eval()
    parent = None
    parent_file = None
    if parent_checkpoint_path is not None:
        parent_file = parent_checkpoint_path.expanduser().resolve()
        parent_checkpoint = torch.load(parent_file, map_location="cpu", weights_only=True)
        parent_contract = (
            int(parent_checkpoint.get("observation_size", -1)),
            int(parent_checkpoint.get("action_size", -1)),
            int(parent_checkpoint.get("hidden_size", -1)),
        )
        candidate_contract = (
            int(checkpoint["observation_size"]),
            int(checkpoint["action_size"]),
            int(checkpoint["hidden_size"]),
        )
        observation_compatible = parent_contract[0] == candidate_contract[0] or (
            parent_contract[0] == 74 and candidate_contract[0] == 77
        )
        if not observation_compatible or parent_contract[1:] != candidate_contract[1:]:
            raise ValueError("goalkeeper CPU exam parent/candidate contract mismatch")
        parent = _build_actor_critic(torch, nn, *parent_contract)
        _load_actor_critic_state(parent, parent_checkpoint["state_dict"])
        parent.eval()
    locomotion = torch.jit.load(
        str(locomotion_policy_path.expanduser().resolve()), map_location="cpu"
    )
    locomotion.eval()
    model = build_g1_stadium_model(root)
    data = mujoco.MjData(model)
    if bool(training_config.get("shot_intent_cue_enabled", False)) != bool(
        active_world.shot_intent_cue_enabled
    ):
        raise ValueError("goalkeeper CPU exam shot-intent cue/world mismatch")
    if active_world.environment_count != 1:
        raise ValueError("goalkeeper CPU exam requires exactly one physics world")
    _require_declared_difficulty_world(active_world)
    seeds = tuple(active.first_seed + index for index in range(active.episode_count))

    policies = (
        (
            "frozen_locomotion_baseline" if parent is None else "parent_champion",
            parent,
        ),
        ("candidate", candidate),
    )
    summaries: dict[str, dict[str, Any]] = {}
    for name, policy in policies:
        results = [
            _run_episode(
                model=model,
                data=data,
                locomotion=locomotion,
                policy=policy,
                seed=seed,
                world_config=active_world,
                combat_teacher=combat_teacher,
                record_trajectory=True,
            )
            for seed in seeds
        ]
        summaries[name] = _summarize(name, results)
    baseline = summaries["frozen_locomotion_baseline" if parent is None else "parent_champion"]
    candidate_summary = summaries["candidate"]
    reasons: list[str] = []
    if not candidate_summary["finite_state"]:
        reasons.append("candidate_nonfinite_state")
    if candidate_summary["failed_rate"] > 0.0:
        reasons.append("candidate_fall_observed")
    if candidate_summary["minimum_pelvis_height_m"] < active.minimum_pelvis_height_m:
        reasons.append("candidate_pelvis_height_below_naturalness_floor")
    if candidate_summary["maximum_root_speed_mps"] > active.maximum_root_speed_mps:
        reasons.append("candidate_root_speed_above_naturalness_ceiling")
    if (
        candidate_summary["maximum_root_angular_speed_rad_s"]
        > active.maximum_root_angular_speed_rad_s
    ):
        reasons.append("candidate_root_angular_speed_above_naturalness_ceiling")
    if (
        candidate_summary["p95_root_angular_speed_rad_s"]
        > active.maximum_p95_root_angular_speed_rad_s
    ):
        reasons.append("candidate_p95_root_angular_speed_above_naturalness_ceiling")
    if candidate_summary["joint_limit_violation_rate"] > 0.0:
        reasons.append("candidate_joint_limit_violation")
    if candidate_summary["torque_limit_violation_rate"] > 0.0:
        reasons.append("candidate_unclipped_torque_demand")
    if (
        candidate_summary["maximum_applied_actor_action_step"]
        > active.maximum_applied_actor_action_step
    ):
        reasons.append("candidate_action_step_above_ceiling")
    rate_floor = active.maximum_save_rate_regression
    if candidate_summary["first_save_rate"] < baseline["first_save_rate"] - rate_floor:
        reasons.append("candidate_first_save_regression")
    if candidate_summary["first_save_rate"] < active.minimum_first_save_rate:
        reasons.append("candidate_first_save_rate_below_absolute_floor")
    if candidate_summary["first_hand_save_rate"] < baseline["first_hand_save_rate"] - rate_floor:
        reasons.append("candidate_first_hand_save_regression")
    if candidate_summary["recovery_rate"] < baseline["recovery_rate"] - rate_floor:
        reasons.append("candidate_recovery_regression")
    if candidate_summary["second_attempt_save_rate"] < (
        baseline["second_attempt_save_rate"]
        + active.minimum_second_attempt_save_improvement
        - rate_floor
    ):
        reasons.append("candidate_second_attempt_save_improvement_below_floor")
    if candidate_summary["second_attempt_save_rate"] < active.minimum_second_attempt_save_rate:
        reasons.append("candidate_second_attempt_save_rate_below_absolute_floor")
    if candidate_summary["second_attempt_hand_save_rate"] < (
        baseline["second_attempt_hand_save_rate"] - rate_floor
    ):
        reasons.append("candidate_second_attempt_hand_save_regression")
    if candidate_summary["second_save_rate"] < (
        baseline["second_save_rate"] + active.minimum_second_save_improvement - rate_floor
    ):
        reasons.append("candidate_second_save_improvement_below_floor")
    if candidate_summary["second_hand_save_rate"] < baseline["second_hand_save_rate"] - rate_floor:
        reasons.append("candidate_second_hand_save_regression")
    if candidate_summary["mean_episode_reward"] <= baseline["mean_episode_reward"]:
        reasons.append("candidate_reward_not_improved")
    if active.minimum_lateral_speed_improvement_mps > 0.0 and candidate_summary[
        "mean_maximum_lateral_speed_mps"
    ] < (baseline["mean_maximum_lateral_speed_mps"] + active.minimum_lateral_speed_improvement_mps):
        reasons.append("candidate_lateral_agility_improvement_below_floor")
    if active.minimum_hand_displacement_improvement_m > 0.0 and candidate_summary[
        "mean_maximum_hand_displacement_m"
    ] < (
        baseline["mean_maximum_hand_displacement_m"]
        + active.minimum_hand_displacement_improvement_m
    ):
        reasons.append("candidate_hand_displacement_improvement_below_floor")
    if candidate_summary["p95_maximum_hand_speed_mps"] > active.maximum_p95_hand_speed_mps:
        reasons.append("candidate_hand_speed_above_naturalness_ceiling")
    if (
        candidate_summary["mean_second_release_lateral_error_m"]
        > active.maximum_mean_second_release_lateral_error_m
    ):
        reasons.append("candidate_recenter_error_above_ceiling")
    passed = not reasons
    report = {
        "schema_version": "rosclaw_soccer.goalkeeper_cpu_exam.v9",
        "exam_config": asdict(active),
        "exam_config_hash": active.config_hash,
        "physics_backend": "mujoco_cpu",
        "world_config": asdict(active_world),
        "world_config_hash": active_world.config_hash,
        "physics_scene_hash": g1_stadium_scene_hash(root),
        "checkpoint_hash": hash_bytes(checkpoint_file.read_bytes()),
        "seeds": list(seeds),
        "paired_rollouts": True,
        "external_combat_teacher": (None if combat_teacher is None else combat_teacher.report),
        "lower_body_authority": (
            "FROZEN_QUALIFIED_LOCOMOTION_PRIOR"
            if combat_teacher is None
            else "BOUNDED_FROZEN_GOALKEEPER_TEACHER_BLEND"
        ),
        "baseline_kind": (
            "FROZEN_LOCOMOTION_ZERO_RESIDUAL" if parent is None else "PARENT_CHAMPION_POLICY"
        ),
        "parent_checkpoint_hash": (
            None if parent_file is None else hash_bytes(parent_file.read_bytes())
        ),
        "baseline": baseline,
        "candidate": candidate_summary,
        "passed": passed,
        "reasons": reasons,
        "promotion_status": "PROMOTED_SIM_ONLY" if passed else "REJECTED_BY_CPU_MUJOCO_EXAM",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "implementation_hashes": {
            "cpu_exam": hash_bytes(Path(__file__).read_bytes()),
            "mjwarp_environment": hash_bytes(
                Path(__file__).with_name("goalkeeper_mjwarp.py").read_bytes()
            ),
            "combat_mjwarp_environment": (
                None
                if combat_teacher is None
                else hash_bytes(
                    Path(__file__).with_name("goalkeeper_combat_mjwarp.py").read_bytes()
                )
            ),
            "combat_teacher": (
                None
                if combat_teacher is None
                else hash_bytes(
                    Path(__file__).with_name("goalkeeper_combat_teacher.py").read_bytes()
                )
            ),
            "joint_guard": hash_bytes(Path(__file__).with_name("joint_guard.py").read_bytes()),
            "multistep_reward": hash_bytes(
                Path(__file__).with_name("goalkeeper_multistep.py").read_bytes()
            ),
            "agility_shaper": hash_bytes(
                Path(__file__).with_name("goalkeeper_agility.py").read_bytes()
            ),
        },
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(output_path.expanduser().resolve(), report)
    return report


__all__ = ["GoalkeeperCPUExamConfig", "run_goalkeeper_cpu_exam"]
