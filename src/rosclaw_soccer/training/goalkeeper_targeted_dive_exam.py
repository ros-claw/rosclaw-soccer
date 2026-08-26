"""Independent CPU MuJoCo exam for a target-conditioned dive option.

The neural option is evaluated on fresh far-corner low/mid/high shots.  The
ball and robot are advanced only by CPU MuJoCo at 500 Hz; contacts and the
environment-owned dive monitor decide success and safety.  A good imitation
fit is necessary but never sufficient for promotion.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.training.goalkeeper_dive_option import GoalkeeperDiveOptionConfig
from rosclaw_soccer.training.goalkeeper_mjwarp import (
    _LOCO_DEFAULT,
    _LOCO_KD,
    _LOCO_KP,
    _LOCO_TO_MOTOR,
)
from rosclaw_soccer.training.goalkeeper_targeted_dive import (
    GoalkeeperTorchDiveMonitor,
    decode_goalkeeper_targeted_dive,
    load_goalkeeper_targeted_dive,
    targeted_dive_features_torch,
)
from rosclaw_soccer.world.field import build_g1_stadium_model, g1_stadium_scene_hash


@dataclass(frozen=True)
class GoalkeeperTargetedDiveExamConfig:
    shots_per_stratum: int = 20
    random_seed: int = 422_001
    control_dt_sec: float = 0.02
    physics_substeps: int = 10
    shot_release_sec: float = 0.70
    prediction_lead_sec: float = 0.30
    episode_duration_sec: float = 4.20
    blend_duration_sec: float = 0.50
    dive_duration_sec: float = 1.40
    recovery_duration_sec: float = 0.60
    intercept_phase_at_arrival: float | None = None
    residual_authority: float = 0.20
    negative_direction_residual_scale: float = 0.25
    far_lateral_range_m: tuple[float, float] = (0.74, 1.06)
    low_height_range_m: tuple[float, float] = (0.16, 0.58)
    mid_height_range_m: tuple[float, float] = (0.64, 1.06)
    high_height_range_m: tuple[float, float] = (1.14, 1.52)
    flight_time_range_sec: tuple[float, float] = (0.40, 0.54)
    ball_start_x_range_m: tuple[float, float] = (0.45, 1.45)
    ball_start_z_range_m: tuple[float, float] = (0.22, 0.78)
    minimum_target_save_rate: float = 0.80
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_targeted_dive_exam_config.v1"

    def __post_init__(self) -> None:
        if not 8 <= self.shots_per_stratum <= 1_000:
            raise ValueError("targeted dive exam stratum size is invalid")
        if not math.isclose(
            self.control_dt_sec / self.physics_substeps,
            0.002,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("targeted dive exam physics step must be 0.002 s")
        if not 0.0 < self.shot_release_sec < self.episode_duration_sec:
            raise ValueError("targeted dive exam timing is invalid")
        if not 0.0 <= self.prediction_lead_sec <= 0.50:
            raise ValueError("targeted dive prediction lead is invalid")
        if self.prediction_lead_sec >= self.shot_release_sec:
            raise ValueError("targeted dive prediction cue precedes the episode")
        if not 0.10 <= self.blend_duration_sec <= 0.60:
            raise ValueError("targeted dive blend timing is invalid")
        if not 0.80 <= self.dive_duration_sec <= 1.40:
            raise ValueError("targeted dive action timing is invalid")
        if self.intercept_phase_at_arrival is not None and not (
            0.45 <= self.intercept_phase_at_arrival <= 0.85
        ):
            raise ValueError("targeted dive intercept phase is invalid")
        if not 0.30 <= self.recovery_duration_sec <= 0.80:
            raise ValueError("targeted dive recovery timing is invalid")
        if not 0.0 <= self.residual_authority <= 1.0:
            raise ValueError("targeted dive residual authority is invalid")
        if not 0.0 <= self.negative_direction_residual_scale <= 1.0:
            raise ValueError("targeted dive direction authority is invalid")
        if (
            self.blend_duration_sec + self.dive_duration_sec + self.recovery_duration_sec
            >= self.episode_duration_sec
        ):
            raise ValueError("targeted dive episode is too short for recovery")
        ranges = (
            self.far_lateral_range_m,
            self.low_height_range_m,
            self.mid_height_range_m,
            self.high_height_range_m,
            self.flight_time_range_sec,
            self.ball_start_x_range_m,
            self.ball_start_z_range_m,
        )
        if any(
            not all(math.isfinite(value) for value in limits) or limits[0] >= limits[1]
            for limits in ranges
        ):
            raise ValueError("targeted dive exam range is invalid")
        if not 0.5 <= self.minimum_target_save_rate <= 1.0:
            raise ValueError("targeted dive exam target rate is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("targeted dive exam is SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _sample_cases(config: GoalkeeperTargetedDiveExamConfig) -> list[dict[str, Any]]:
    rng = np.random.default_rng(config.random_seed)
    cases: list[dict[str, Any]] = []
    for stratum, height_range in (
        ("far_corner_low", config.low_height_range_m),
        ("far_corner_mid", config.mid_height_range_m),
        ("far_corner_high", config.high_height_range_m),
    ):
        for _ in range(config.shots_per_stratum):
            direction = float(rng.choice((-1.0, 1.0)))
            cases.append(
                {
                    "seed": config.random_seed + len(cases),
                    "stratum": stratum,
                    "target_y_m": direction * float(rng.uniform(*config.far_lateral_range_m)),
                    "target_z_m": float(rng.uniform(*height_range)),
                    "flight_sec": float(rng.uniform(*config.flight_time_range_sec)),
                    "start_x_m": float(rng.uniform(*config.ball_start_x_range_m)),
                    "start_z_m": float(rng.uniform(*config.ball_start_z_range_m)),
                }
            )
    rng.shuffle(cases)
    return cases


def _ready_control() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ready = np.zeros(29, dtype=np.float64)
    order = np.asarray(_LOCO_TO_MOTOR, dtype=np.int64)
    ready[order] = np.asarray(_LOCO_DEFAULT, dtype=np.float64)
    kp = np.zeros(29, dtype=np.float64)
    kd = np.zeros(29, dtype=np.float64)
    kp[order] = np.asarray(_LOCO_KP, dtype=np.float64)
    kd[order] = np.asarray(_LOCO_KD, dtype=np.float64)
    kp[12:] = np.asarray((150.0, 150.0, 150.0) + ((150.0,) * 4 + (20.0,) * 3) * 2)
    kd[12:] = np.asarray((2.0, 2.0, 2.0) + ((2.0,) * 4 + (0.5,) * 3) * 2)
    return ready, kp, kd, np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)


def _contacts(model: Any, data: Any) -> tuple[bool, bool, bool, bool, float]:
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
    landing_geoms = hand_geoms | {
        int(model.geom("left_elbow_yaw_collision").id),
        int(model.geom("right_elbow_yaw_collision").id),
    }
    forbidden_geoms = {
        int(model.geom("pelvis_collision").id),
        int(model.geom("torso_collision1").id),
        int(model.geom("torso_collision2").id),
        int(model.geom("torso_collision3").id),
    }
    foot_left = set(range(73, 80))
    foot_right = set(range(88, 95))
    robot_contact = False
    hand_contact = False
    landing = False
    forbidden = False
    left_support = False
    right_support = False
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.dist > 0.002:
            continue
        one = int(contact.geom1)
        two = int(contact.geom2)
        body_one = int(model.geom_bodyid[one])
        body_two = int(model.geom_bodyid[two])
        ball_one = body_one == ball_body
        ball_two = body_two == ball_body
        robot_one = 1 <= body_one < ball_body
        robot_two = 1 <= body_two < ball_body
        if (ball_one and robot_two) or (ball_two and robot_one):
            robot_contact = True
            hand_contact |= bool(
                (ball_one and two in hand_geoms) or (ball_two and one in hand_geoms)
            )
        ground_pair = body_one == 0 or body_two == 0
        robot_geom = two if body_one == 0 else one
        if ground_pair:
            landing |= robot_geom in landing_geoms
            forbidden |= robot_geom in forbidden_geoms
            left_support |= robot_geom in foot_left
            right_support |= robot_geom in foot_right
    support_side = float(int(right_support) - int(left_support))
    return robot_contact, hand_contact, landing, forbidden, support_side


def _launch(data: Any, case: dict[str, Any]) -> None:
    target_x = 4.44
    target_y = float(case["target_y_m"])
    target_z = float(case["target_z_m"])
    flight = float(case["flight_sec"])
    start_x = float(case["start_x_m"])
    start_z = float(case["start_z_m"])
    data.qpos[36:43] = (start_x, 0.18 * target_y, start_z, 1.0, 0.0, 0.0, 0.0)
    data.qvel[35:41] = 0.0
    data.qvel[35] = (target_x - start_x) / flight
    data.qvel[36] = (target_y - data.qpos[37]) / flight
    data.qvel[37] = (target_z - start_z + 0.5 * 9.81 * flight * flight) / flight


def _scheduled_phase(
    *, raw_phase: float, arrival_raw_phase: float, intercept_phase: float | None
) -> float:
    """Warp phase continuously while preserving both option endpoints."""

    raw = float(np.clip(raw_phase, 0.0, 1.0))
    if intercept_phase is None:
        return raw
    arrival = float(np.clip(arrival_raw_phase, 0.05, 0.95))
    if raw <= arrival:
        return float(intercept_phase * raw / arrival)
    return float(intercept_phase + (1.0 - intercept_phase) * (raw - arrival) / (1.0 - arrival))


def _rollout_case(
    *,
    model: Any,
    decoder: Any | None,
    checkpoint: dict[str, Any] | None,
    locomotion: Any,
    case: dict[str, Any],
    config: GoalkeeperTargetedDiveExamConfig,
) -> dict[str, Any]:
    import mujoco
    import torch

    data = mujoco.MjData(model)
    ready, kp, kd, limits = _ready_control()
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = (4.52, 0.0, 0.793, 0.0, 0.0, 0.0, 1.0)
    data.qpos[7:36] = ready
    data.qpos[36:43] = (-20.0, 0.0, 0.115, 1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    monitor = GoalkeeperTorchDiveMonitor(
        torch=torch,
        environment_count=1,
        device=torch.device("cpu"),
        config=GoalkeeperDiveOptionConfig(
            maximum_option_duration_sec=3.60,
            recovery_hold_sec=0.20,
        ),
    )
    release_step = int(round(config.shot_release_sec / config.control_dt_sec))
    cue_step = release_step - int(round(config.prediction_lead_sec / config.control_dt_sec))
    steps = int(round(config.episode_duration_sec / config.control_dt_sec))
    blend_steps = int(round(config.blend_duration_sec / config.control_dt_sec))
    dive_steps = int(round(config.dive_duration_sec / config.control_dt_sec))
    arrival_sec = config.shot_release_sec + float(case["flight_sec"])
    cue_sec = cue_step * config.control_dt_sec
    arrival_raw_phase = (
        arrival_sec - cue_sec - config.blend_duration_sec
    ) / config.dive_duration_sec
    direction = -1.0 if float(case["target_y_m"]) < 0.0 else 1.0
    residual_authority = config.residual_authority * (
        config.negative_direction_residual_scale if direction < 0.0 else 1.0
    )
    robot_contact = False
    hand_contact = False
    contact_before_plane = False
    landing = False
    forbidden = False
    forbidden_latched = False
    prelaunch_violation = False
    support_side = 0.0
    unsafe_latched = False
    minimum_pelvis = math.inf
    maximum_angular = 0.0
    maximum_torque_fraction = 0.0
    minimum_hand_distance = math.inf
    minimum_hand_distance_time_sec = math.inf
    minimum_hand_error_xyz = np.full(3, math.inf, dtype=np.float64)
    hand_distance_at_arrival = math.inf
    hand_error_at_arrival_xyz = np.full(3, math.inf, dtype=np.float64)
    arrival_sample_error_sec = math.inf
    first_robot_contact_time_sec: float | None = None
    joint_lower = np.asarray(model.jnt_range[1:30, 0], dtype=np.float64)
    joint_upper = np.asarray(model.jnt_range[1:30, 1], dtype=np.float64)
    terminal_target = ready.copy()
    handoff_step = cue_step + blend_steps + dive_steps
    loco_order = np.asarray(_LOCO_TO_MOTOR, dtype=np.int64)
    loco_default = np.asarray(_LOCO_DEFAULT, dtype=np.float64)
    loco_action = np.zeros(29, dtype=np.float64)
    loco_hidden = torch.zeros((1, 1, 256), dtype=torch.float32)
    loco_cell = torch.zeros_like(loco_hidden)
    for step in range(steps):
        time_sec = step * config.control_dt_sec
        if step == release_step:
            _launch(data, case)
            mujoco.mj_forward(model, data)
        qw, qx, qy, qz = (float(data.qpos[index]) for index in range(3, 7))
        upright = 2.0 * (qw * qw + qz * qz) - 1.0
        linear_speed = float(np.linalg.norm(data.qvel[:3]))
        angular_speed = float(np.linalg.norm(data.qvel[3:6]))
        gravity = np.asarray(
            (
                2.0 * (-qz * qx + qw * qy),
                -2.0 * (qz * qy + qw * qx),
                1.0 - 2.0 * (qw * qw + qz * qz),
            ),
            dtype=np.float64,
        )
        loco_observation = np.zeros(96, dtype=np.float64)
        loco_observation[:3] = data.qvel[3:6]
        loco_observation[3:6] = gravity
        loco_observation[9:38] = data.qpos[7:36][loco_order] - loco_default
        loco_observation[38:67] = data.qvel[6:35][loco_order]
        loco_observation[67:96] = loco_action
        option_step = step - cue_step
        option_visible = bool(decoder is not None and step >= cue_step)
        if decoder is not None and cue_step <= step <= handoff_step:
            loco_action.fill(0.0)
            loco_hidden = torch.zeros((1, 1, 256), dtype=torch.float32)
            loco_cell = torch.zeros_like(loco_hidden)
        with torch.inference_mode():
            encoded = locomotion.normalizer.forward(
                torch.as_tensor(loco_observation, dtype=torch.float32).unsqueeze(0)
            )
            sequence, (loco_hidden, loco_cell) = locomotion.rnn.forward__0(
                encoded.unsqueeze(0), (loco_hidden, loco_cell)
            )
            loco_action = np.asarray(
                torch.clamp(locomotion.actor.forward(sequence.squeeze(0)), -100.0, 100.0)[0],
                dtype=np.float64,
            )
        loco_target = np.zeros(29, dtype=np.float64)
        loco_target[loco_order] = 0.25 * loco_action + loco_default
        monitor_result = monitor.step(
            option_request=torch.tensor((decoder is not None and step == cue_step,)),
            threat_id=torch.tensor((1,), dtype=torch.long),
            threat_visible=torch.tensor((option_visible,)),
            lateral_intercept_error_m=torch.tensor(
                (float(case["target_y_m"]) - float(data.qpos[1]),)
            ),
            pelvis_height_m=torch.tensor((float(data.qpos[2]),)),
            upright_projection=torch.tensor((upright,)),
            root_linear_speed_mps=torch.tensor((linear_speed,)),
            root_angular_speed_rad_s=torch.tensor((angular_speed,)),
            permitted_landing_contact=torch.tensor((landing,)),
            forbidden_body_contact=torch.tensor((forbidden,)),
        )
        unsafe_latched |= bool(monitor_result["unsafe"][0])
        if decoder is None or checkpoint is None:
            target = loco_target
        elif option_step < 0:
            # A target-independent ready stance avoids leaking the shot while
            # ensuring the imitation option starts from its declared manifold.
            target = ready
        elif option_step < blend_steps:
            phase = 0.0
            features = targeted_dive_features_torch(
                torch=torch,
                direction=torch.tensor((direction,)),
                phase=torch.tensor((phase,)),
                target_lateral_m=torch.tensor((float(case["target_y_m"]),)),
                target_height_m=torch.tensor((float(case["target_z_m"]),)),
                time_to_arrival_sec=torch.tensor((max(0.0, arrival_sec - time_sec),)),
                root_lateral_m=torch.tensor((float(data.qpos[1]),)),
                root_lateral_speed_mps=torch.tensor((float(data.qvel[1]),)),
                pelvis_height_m=torch.tensor((float(data.qpos[2]),)),
                upright_projection=torch.tensor((upright,)),
                support_side=torch.tensor((support_side,)),
                root_angular_speed_rad_s=torch.tensor((angular_speed,)),
            )
            option_target = np.asarray(
                decode_goalkeeper_targeted_dive(
                    model=decoder,
                    checkpoint=checkpoint,
                    features=features,
                    residual_authority=residual_authority,
                )[0],
                dtype=np.float64,
            )
            blend = (option_step + 1) / blend_steps
            target = ready + blend * (option_target - ready)
            terminal_target = option_target
        elif option_step < blend_steps + dive_steps:
            raw_phase = (option_step - blend_steps) / max(dive_steps - 1, 1)
            phase = _scheduled_phase(
                raw_phase=raw_phase,
                arrival_raw_phase=arrival_raw_phase,
                intercept_phase=config.intercept_phase_at_arrival,
            )
            features = targeted_dive_features_torch(
                torch=torch,
                direction=torch.tensor((direction,)),
                phase=torch.tensor((phase,)),
                target_lateral_m=torch.tensor((float(case["target_y_m"]),)),
                target_height_m=torch.tensor((float(case["target_z_m"]),)),
                time_to_arrival_sec=torch.tensor((max(0.0, arrival_sec - time_sec),)),
                root_lateral_m=torch.tensor((float(data.qpos[1]),)),
                root_lateral_speed_mps=torch.tensor((float(data.qvel[1]),)),
                pelvis_height_m=torch.tensor((float(data.qpos[2]),)),
                upright_projection=torch.tensor((upright,)),
                support_side=torch.tensor((support_side,)),
                root_angular_speed_rad_s=torch.tensor((angular_speed,)),
            )
            terminal_target = np.asarray(
                decode_goalkeeper_targeted_dive(
                    model=decoder,
                    checkpoint=checkpoint,
                    features=features,
                    residual_authority=residual_authority,
                )[0],
                dtype=np.float64,
            )
            target = terminal_target
        else:
            target = loco_target
        target = np.clip(target, joint_lower, joint_upper)
        for substep in range(config.physics_substeps):
            requested = kp * (target - data.qpos[7:36]) - kd * data.qvel[6:35]
            maximum_torque_fraction = max(
                maximum_torque_fraction,
                float(np.max(np.abs(requested) / limits)),
            )
            data.ctrl[:] = np.clip(requested, -limits, limits)
            mujoco.mj_step(model, data)
            robot_now, hand_now, landing, forbidden, support_side = _contacts(model, data)
            forbidden_latched |= forbidden
            robot_contact |= robot_now
            hand_contact |= hand_now
            contact_before_plane |= bool(robot_now and float(data.qpos[36]) <= 4.60)
            physics_time_sec = time_sec + (substep + 1) * (
                config.control_dt_sec / config.physics_substeps
            )
            if robot_now and first_robot_contact_time_sec is None:
                first_robot_contact_time_sec = physics_time_sec
            target_point = np.asarray(
                (4.44, float(case["target_y_m"]), float(case["target_z_m"])),
                dtype=np.float64,
            )
            left = np.asarray(
                data.geom_xpos[int(model.geom("left_hand_collision").id)],
                dtype=np.float64,
            )
            right = np.asarray(
                data.geom_xpos[int(model.geom("right_hand_collision").id)],
                dtype=np.float64,
            )
            left_error = left - target_point
            right_error = right - target_point
            if np.linalg.norm(left_error) <= np.linalg.norm(right_error):
                selected_error = left_error
            else:
                selected_error = right_error
            selected_distance = float(np.linalg.norm(selected_error))
            if selected_distance < minimum_hand_distance:
                minimum_hand_distance = selected_distance
                minimum_hand_distance_time_sec = physics_time_sec
                minimum_hand_error_xyz = selected_error.copy()
            sample_error = abs(physics_time_sec - arrival_sec)
            if sample_error < arrival_sample_error_sec:
                arrival_sample_error_sec = sample_error
                hand_distance_at_arrival = selected_distance
                hand_error_at_arrival_xyz = selected_error.copy()
            minimum_pelvis = min(minimum_pelvis, float(data.qpos[2]))
            maximum_angular = max(maximum_angular, float(np.linalg.norm(data.qvel[3:6])))
    finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
    recovered = bool(decoder is None or int(monitor.completed_dives[0]) > 0)
    if decoder is None:
        safe = bool(
            finite and minimum_pelvis >= 0.55 and maximum_angular <= 3.50 and not forbidden_latched
        )
    else:
        safe = bool(
            finite
            and not unsafe_latched
            and not prelaunch_violation
            and not forbidden_latched
            and recovered
        )
    saved = bool(contact_before_plane and safe)
    timing_error_sec = minimum_hand_distance_time_sec - arrival_sec
    if saved:
        failure_category = "SAVED"
    elif not safe:
        failure_category = "SAFETY_OR_RECOVERY"
    elif robot_contact:
        failure_category = "LATE_CONTACT"
    elif minimum_hand_distance <= 0.18 and abs(timing_error_sec) > 0.06:
        failure_category = "TIMING_MISMATCH"
    elif abs(float(hand_error_at_arrival_xyz[2])) > 1.15 * abs(float(hand_error_at_arrival_xyz[1])):
        failure_category = "HEIGHT_GAP"
    elif abs(float(hand_error_at_arrival_xyz[1])) > 1.15 * abs(float(hand_error_at_arrival_xyz[2])):
        failure_category = "LATERAL_GAP"
    else:
        failure_category = "CONTACT_GEOMETRY"
    return {
        **case,
        "saved": saved,
        "robot_contact": robot_contact,
        "hand_contact": hand_contact,
        "safe": safe,
        "prelaunch_violation": prelaunch_violation,
        "forbidden_contact": forbidden_latched,
        "recovered": recovered,
        "minimum_pelvis_height_m": minimum_pelvis,
        "maximum_root_angular_speed_rad_s": maximum_angular,
        "maximum_requested_torque_fraction": maximum_torque_fraction,
        "minimum_hand_distance_m": minimum_hand_distance,
        "minimum_hand_distance_time_sec": minimum_hand_distance_time_sec,
        "hand_distance_at_arrival_m": hand_distance_at_arrival,
        "hand_target_error_at_arrival_xyz_m": hand_error_at_arrival_xyz.tolist(),
        "minimum_hand_target_error_xyz_m": minimum_hand_error_xyz.tolist(),
        "minimum_hand_timing_error_sec": timing_error_sec,
        "prediction_cue_sec": cue_sec,
        "available_response_time_sec": arrival_sec - cue_sec,
        "first_robot_contact_time_sec": first_robot_contact_time_sec,
        "failure_category": failure_category,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strata: dict[str, Any] = {}
    for name in ("far_corner_low", "far_corner_mid", "far_corner_high"):
        selected = [row for row in rows if row["stratum"] == name]
        strata[name] = {
            "attempts": len(selected),
            "saves": sum(bool(row["saved"]) for row in selected),
            "save_rate": sum(bool(row["saved"]) for row in selected) / len(selected),
            "hand_save_rate": sum(bool(row["saved"] and row["hand_contact"]) for row in selected)
            / len(selected),
            "safe_rate": sum(bool(row["safe"]) for row in selected) / len(selected),
            "mean_minimum_hand_distance_m": float(
                np.mean([row["minimum_hand_distance_m"] for row in selected])
            ),
            "mean_hand_distance_at_arrival_m": float(
                np.mean([row["hand_distance_at_arrival_m"] for row in selected])
            ),
            "mean_absolute_timing_error_sec": float(
                np.mean([abs(row["minimum_hand_timing_error_sec"]) for row in selected])
            ),
        }
    directions: dict[str, Any] = {}
    for name, sign in (("left", -1.0), ("right", 1.0)):
        selected = [row for row in rows if math.copysign(1.0, row["target_y_m"]) == sign]
        directions[name] = {
            "attempts": len(selected),
            "saves": sum(bool(row["saved"]) for row in selected),
            "save_rate": sum(bool(row["saved"]) for row in selected) / len(selected),
            "safe_rate": sum(bool(row["safe"]) for row in selected) / len(selected),
            "mean_hand_distance_at_arrival_m": float(
                np.mean([row["hand_distance_at_arrival_m"] for row in selected])
            ),
        }
    failure_categories = {
        name: sum(row["failure_category"] == name for row in rows)
        for name in (
            "SAFETY_OR_RECOVERY",
            "LATE_CONTACT",
            "TIMING_MISMATCH",
            "HEIGHT_GAP",
            "LATERAL_GAP",
            "CONTACT_GEOMETRY",
        )
    }
    summary: dict[str, Any] = {
        "attempts": len(rows),
        "save_rate": sum(bool(row["saved"]) for row in rows) / len(rows),
        "raw_contact_rate": sum(bool(row["robot_contact"]) for row in rows) / len(rows),
        "raw_hand_contact_rate": sum(bool(row["hand_contact"]) for row in rows) / len(rows),
        "safe_rate": sum(bool(row["safe"]) for row in rows) / len(rows),
        "recovery_rate": sum(bool(row["recovered"]) for row in rows) / len(rows),
        "strata": strata,
        "directions": directions,
        "failure_categories": failure_categories,
        "minimum_pelvis_height_m": min(float(row["minimum_pelvis_height_m"]) for row in rows),
        "maximum_root_angular_speed_rad_s": max(
            float(row["maximum_root_angular_speed_rad_s"]) for row in rows
        ),
        "failures": [
            {
                "seed": row["seed"],
                "stratum": row["stratum"],
                "target_y_m": row["target_y_m"],
                "target_z_m": row["target_z_m"],
                "safe": row["safe"],
                "recovered": row["recovered"],
                "minimum_hand_distance_m": row["minimum_hand_distance_m"],
                "hand_distance_at_arrival_m": row["hand_distance_at_arrival_m"],
                "minimum_hand_timing_error_sec": row["minimum_hand_timing_error_sec"],
                "hand_target_error_at_arrival_xyz_m": row["hand_target_error_at_arrival_xyz_m"],
                "flight_sec": row["flight_sec"],
                "failure_category": row["failure_category"],
                **(
                    {
                        "maximum_applied_option_gate": row["maximum_applied_option_gate"],
                        "maximum_applied_runtime_reach_blend": row[
                            "maximum_applied_runtime_reach_blend"
                        ],
                        "maximum_applied_overhead_reach_blend": row.get(
                            "maximum_applied_overhead_reach_blend", 0.0
                        ),
                    }
                    if "maximum_applied_option_gate" in row
                    else {}
                ),
            }
            for row in rows
            if not row["saved"]
        ],
    }
    if rows and "maximum_applied_option_gate" in rows[0]:
        summary["mean_maximum_applied_option_gate"] = float(
            np.mean([row["maximum_applied_option_gate"] for row in rows])
        )
        summary["mean_maximum_applied_runtime_reach_blend"] = float(
            np.mean([row["maximum_applied_runtime_reach_blend"] for row in rows])
        )
        summary["mean_maximum_applied_overhead_reach_blend"] = float(
            np.mean([row.get("maximum_applied_overhead_reach_blend", 0.0) for row in rows])
        )
    return summary


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def run_goalkeeper_targeted_dive_exam(
    *,
    asset_root: Path,
    locomotion_policy_path: Path,
    checkpoint_path: Path,
    training_report_path: Path,
    output_path: Path,
    config: GoalkeeperTargetedDiveExamConfig | None = None,
) -> dict[str, Any]:
    """Run paired ready-pose and target-conditioned trials on fresh cases."""

    import torch

    active = config or GoalkeeperTargetedDiveExamConfig()
    decoder, checkpoint = load_goalkeeper_targeted_dive(
        checkpoint_path=checkpoint_path, device=torch.device("cpu")
    )
    training_report = json.loads(
        training_report_path.expanduser().resolve().read_text(encoding="utf-8")
    )
    model = build_g1_stadium_model(asset_root.expanduser().resolve())
    locomotion_file = locomotion_policy_path.expanduser().resolve()
    locomotion = torch.jit.load(  # type: ignore[no-untyped-call]
        str(locomotion_file), map_location="cpu"
    )
    locomotion.eval()
    cases = _sample_cases(active)
    baseline_rows = [
        _rollout_case(
            model=model,
            decoder=None,
            checkpoint=None,
            locomotion=locomotion,
            case=case,
            config=active,
        )
        for case in cases
    ]
    candidate_rows = [
        _rollout_case(
            model=model,
            decoder=decoder,
            checkpoint=checkpoint,
            locomotion=locomotion,
            case=case,
            config=active,
        )
        for case in cases
    ]
    baseline = _summary(baseline_rows)
    candidate = _summary(candidate_rows)
    strata_pass = all(
        float(item["save_rate"]) >= active.minimum_target_save_rate
        for item in candidate["strata"].values()
    )
    fit_passed = bool(training_report.get("fit_gate_passed", False))
    passed = bool(
        fit_passed
        and strata_pass
        and candidate["safe_rate"] == 1.0
        and candidate["recovery_rate"] == 1.0
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.goalkeeper_targeted_dive_exam.v1",
        "physics_backend": "mujoco_cpu",
        "physics_scene_hash": g1_stadium_scene_hash(asset_root.expanduser().resolve()),
        "config": asdict(active),
        "config_hash": active.config_hash,
        "challenge": "FIRST_SHOT_FAR_CORNER_LOW_MID_HIGH_ONLY",
        "target_information_contract": "BOUNDED_PRE_KICK_PREDICTION_CUE",
        "pre_cue_controller": "TARGET_INDEPENDENT_READY_STANCE",
        "paired_seed_count": len(cases),
        "training_fit_gate_passed": fit_passed,
        "training_report_hash": training_report.get("report_hash"),
        "locomotion_policy_path": locomotion_file.name,
        "locomotion_policy_hash": hash_bytes(locomotion_file.read_bytes()),
        "baseline": baseline,
        "candidate": candidate,
        "failure_replay_manifest": {
            "case_count": len(candidate["failures"]),
            "failure_categories": candidate["failure_categories"],
            "manifest_hash": hash_json(candidate["failures"]),
            "eligible_for_next_sim_training": True,
        },
        "minimum_target_save_rate": active.minimum_target_save_rate,
        "passed_80_percent_each_stratum": strata_pass,
        "passed": passed,
        "promotion_status": (
            "OPTION_PENDING_COMBAT_POLICY_INTEGRATION"
            if passed
            else "REJECTED_BY_TARGETED_DIVE_CPU_EXAM"
        ),
        "video_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--locomotion-policy", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots-per-stratum", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=422_001)
    parser.add_argument("--prediction-lead", type=float, default=0.30)
    parser.add_argument("--blend-duration", type=float, default=0.50)
    parser.add_argument("--dive-duration", type=float, default=1.40)
    parser.add_argument("--intercept-phase", type=float, default=None)
    parser.add_argument("--residual-authority", type=float, default=0.20)
    parser.add_argument("--negative-direction-scale", type=float, default=0.25)
    args = parser.parse_args()
    report = run_goalkeeper_targeted_dive_exam(
        asset_root=args.asset_root,
        locomotion_policy_path=args.locomotion_policy,
        checkpoint_path=args.checkpoint,
        training_report_path=args.training_report,
        output_path=args.output,
        config=GoalkeeperTargetedDiveExamConfig(
            shots_per_stratum=args.shots_per_stratum,
            random_seed=args.random_seed,
            prediction_lead_sec=args.prediction_lead,
            blend_duration_sec=args.blend_duration,
            dive_duration_sec=args.dive_duration,
            intercept_phase_at_arrival=args.intercept_phase,
            residual_authority=args.residual_authority,
            negative_direction_residual_scale=args.negative_direction_scale,
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GoalkeeperTargetedDiveExamConfig",
    "run_goalkeeper_targeted_dive_exam",
]
