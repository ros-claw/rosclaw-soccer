"""Independent CPU MuJoCo exam for a residual-RL targeted dive actor."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.goalkeeper_mjwarp import _LOCO_DEFAULT, _LOCO_TO_MOTOR
from rosclaw_soccer.training.goalkeeper_physics_ppo import (
    _build_actor_critic,
    _load_actor_critic_state,
)
from rosclaw_soccer.training.goalkeeper_targeted_dive import (
    decode_goalkeeper_targeted_dive,
    load_goalkeeper_targeted_dive,
    targeted_dive_features_torch,
)
from rosclaw_soccer.training.goalkeeper_targeted_dive_exam import (
    GoalkeeperTargetedDiveExamConfig,
    _contacts,
    _launch,
    _ready_control,
    _rollout_case,
    _sample_cases,
    _summary,
)
from rosclaw_soccer.training.goalkeeper_targeted_dive_mjwarp import (
    TARGETED_DIVE_RL_RESIDUAL_LIMITS_RAD,
    GoalkeeperTargetedDiveRLConfig,
)
from rosclaw_soccer.world.field import build_g1_stadium_model, g1_stadium_scene_hash


def _load_bound_report(path: Path, *, label: str) -> dict[str, Any]:
    report = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    claimed_hash = report.get("report_hash")
    payload = dict(report)
    payload.pop("report_hash", None)
    if claimed_hash != hash_json(payload):
        raise ValueError(f"{label} report content hash is invalid")
    return cast(dict[str, Any], report)


def _load_actor(*, path: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch import nn

    checkpoint = torch.load(path.expanduser().resolve(), map_location=device, weights_only=True)
    if (
        checkpoint.get("activation_ceiling") != "SIM_ONLY"
        or checkpoint.get("promotion_status") != "CANDIDATE_PENDING_CPU_MUJOCO_EXAM"
        or int(checkpoint.get("action_size", -1)) != 30
        or int(checkpoint.get("observation_size", -1)) != 89
    ):
        raise ValueError("targeted dive actor checkpoint boundary is invalid")
    actor = _build_actor_critic(
        torch,
        nn,
        int(checkpoint["observation_size"]),
        int(checkpoint["action_size"]),
        int(checkpoint["hidden_size"]),
    ).to(device)
    _load_actor_critic_state(actor, checkpoint["state_dict"])
    actor.eval()
    return actor, checkpoint


def _causal_target(
    *, data: Any, case: dict[str, Any], shot_active: bool, target_y_range: tuple[float, float]
) -> np.ndarray:
    if not shot_active or float(data.qvel[35]) <= 0.10:
        return np.asarray((4.44, 0.0, 0.82), dtype=np.float64)
    time_to_line = np.clip(
        (4.44 - float(data.qpos[36])) / max(float(data.qvel[35]), 0.10), 0.0, 1.2
    )
    return np.asarray(
        (
            4.44,
            np.clip(
                float(data.qpos[37]) + float(data.qvel[36]) * time_to_line,
                *target_y_range,
            ),
            np.clip(
                float(data.qpos[38])
                + float(data.qvel[37]) * time_to_line
                - 0.5 * 9.81 * time_to_line * time_to_line,
                0.10,
                1.62,
            ),
        ),
        dtype=np.float64,
    )


def _actor_observation(
    *,
    torch: Any,
    data: Any,
    target: np.ndarray,
    cue: np.ndarray,
    cue_visible: bool,
    shot_active: bool,
    previous_action: Any,
    previous_target: np.ndarray,
    step: int,
    first_end_step: int,
    episode_duration_sec: float,
    control_dt_sec: float,
) -> Any:
    root = np.asarray(data.qpos[:3], dtype=np.float64)
    ball = np.asarray(data.qpos[36:39], dtype=np.float64)
    qw, qx, qy, qz = (float(data.qpos[index]) for index in range(3, 7))
    gravity = np.asarray(
        (
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ),
        dtype=np.float64,
    )
    phase = np.asarray(
        (0.0, 1.0, 0.0) if shot_active and step < first_end_step else (1.0, 0.0, 0.0),
        dtype=np.float64,
    )
    cue_feature = cue if cue_visible else np.zeros(3, dtype=np.float64)
    observation = np.concatenate(
        (
            (ball - root) * 0.4,
            np.asarray(data.qvel[35:38], dtype=np.float64) * 0.2,
            (target - root) * 0.5,
            gravity,
            np.asarray(data.qvel[:3], dtype=np.float64) * 0.5,
            np.asarray(data.qvel[3:6], dtype=np.float64) * 0.25,
            np.asarray(data.qpos[19:36], dtype=np.float64) - previous_target[12:29],
            np.asarray(data.qvel[18:35], dtype=np.float64) * 0.05,
            np.asarray(previous_action.cpu(), dtype=np.float64),
            cue_feature,
            phase,
            np.asarray((step * control_dt_sec / episode_duration_sec,), dtype=np.float64),
        )
    )
    if observation.shape != (89,) or not np.all(np.isfinite(observation)):
        raise RuntimeError("targeted dive CPU actor observation contract changed")
    return torch.as_tensor(np.clip(observation, -10.0, 10.0), dtype=torch.float32).unsqueeze(0)


def _rollout_actor_case(
    *,
    model: Any,
    actor: Any,
    decoder: Any,
    decoder_checkpoint: dict[str, Any],
    locomotion: Any,
    case: dict[str, Any],
    exam_config: GoalkeeperTargetedDiveExamConfig,
    dive_config: GoalkeeperTargetedDiveRLConfig,
    maximum_lateral_command_mps: float,
    runtime_reach_atlas: Any | None,
    overhead_reach_prior: Any | None,
    arm_target_filter_enabled: bool,
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
    release_step = int(round(exam_config.shot_release_sec / exam_config.control_dt_sec))
    cue_step = release_step - int(
        round(dive_config.prediction_lead_sec / exam_config.control_dt_sec)
    )
    first_end_step = int(round(1.70 / exam_config.control_dt_sec))
    steps = int(round(exam_config.episode_duration_sec / exam_config.control_dt_sec))
    option_steps = int(round(dive_config.option_duration_sec / exam_config.control_dt_sec))
    hold_steps = int(round(dive_config.phase_hold_sec / exam_config.control_dt_sec))
    total_option_steps = hold_steps + option_steps
    plasticity_steps = int(
        round(dive_config.actor_plasticity_duration_sec / exam_config.control_dt_sec)
    )
    exception_steps = int(
        round(dive_config.posture_exception_duration_sec / exam_config.control_dt_sec)
    )
    arrival_sec = exam_config.shot_release_sec + float(case["flight_sec"])
    nominal_arrival_sec = exam_config.shot_release_sec + dive_config.nominal_shot_flight_time_sec
    cue = np.asarray((float(case["target_y_m"]), float(case["target_z_m"]), 1.0), dtype=np.float64)
    group_scale = np.asarray(
        (dive_config.anchor_lower_body_scale,) * 12
        + (dive_config.anchor_waist_scale,) * 3
        + (dive_config.anchor_arm_scale,) * 14,
        dtype=np.float64,
    )
    decoder_group_authority = torch.as_tensor(
        (dive_config.resolved_decoder_lower_body_residual_authority,) * 12
        + (dive_config.resolved_decoder_waist_residual_authority,) * 3
        + (dive_config.resolved_decoder_arm_residual_authority,) * 14,
        dtype=torch.float32,
    )
    residual_limits = np.asarray(TARGETED_DIVE_RL_RESIDUAL_LIMITS_RAD) * (
        dive_config.actor_residual_scale
    )
    joint_lower = np.asarray(model.jnt_range[1:30, 0], dtype=np.float64)
    joint_upper = np.asarray(model.jnt_range[1:30, 1], dtype=np.float64)
    loco_order = np.asarray(_LOCO_TO_MOTOR, dtype=np.int64)
    loco_default = np.asarray(_LOCO_DEFAULT, dtype=np.float64)
    loco_action = np.zeros(29, dtype=np.float64)
    loco_hidden = torch.zeros((1, 1, 256), dtype=torch.float32)
    loco_cell = torch.zeros_like(loco_hidden)
    previous_actor = torch.zeros(30, dtype=torch.float32)
    previous_target = ready.copy()
    previous_option_arm_target = ready[15:29].copy()
    option_started = False
    option_age = 0
    previous_option_phase = 0.0
    maximum_gate = 0.0
    maximum_runtime_reach_blend = 0.0
    maximum_overhead_reach_blend = 0.0
    minimum_substep_arm_authority = 1.0
    minimum_substep_option_lower_body_authority = 1.0
    shot_active = False
    robot_contact = False
    hand_contact = False
    contact_before_plane = False
    forbidden_latched = False
    unsafe_latched = False
    stable_steps = 0
    minimum_pelvis = math.inf
    maximum_angular = 0.0
    minimum_hand_distance = math.inf
    minimum_hand_time = math.inf
    hand_distance_at_arrival = math.inf
    arrival_sample_error = math.inf
    arrival_error = np.full(3, math.inf)
    first_contact_time: float | None = None
    support_side = 0.0
    for step in range(steps):
        time_sec = step * exam_config.control_dt_sec
        if step == release_step:
            _launch(data, case)
            shot_active = True
            mujoco.mj_forward(model, data)
        if step == first_end_step:
            shot_active = False
            data.qpos[36:43] = (-20.0, 0.0, 0.115, 1.0, 0.0, 0.0, 0.0)
            data.qvel[35:41] = 0.0
            mujoco.mj_forward(model, data)
        cue_visible = cue_step <= step < release_step
        causal_target = _causal_target(
            data=data,
            case=case,
            shot_active=shot_active,
            target_y_range=(-1.08, 1.08),
        )
        if cue_visible:
            causal_target = np.asarray((4.44, cue[0], cue[1]), dtype=np.float64)
        observation = _actor_observation(
            torch=torch,
            data=data,
            target=causal_target,
            cue=cue,
            cue_visible=cue_visible,
            shot_active=shot_active,
            previous_action=previous_actor,
            previous_target=previous_target,
            step=step,
            first_end_step=first_end_step,
            episode_duration_sec=5.0,
            control_dt_sec=exam_config.control_dt_sec,
        )
        with torch.inference_mode():
            mean, _, _ = actor(observation)
            requested = torch.tanh(mean[0])
        qw, qx, qy, qz = (float(data.qpos[index]) for index in range(3, 7))
        upright = 2.0 * (qw * qw + qz * qz) - 1.0
        angular_speed = float(np.linalg.norm(data.qvel[3:6]))
        authority = float(
            np.clip(
                (dive_config.dive_maximum_root_angular_speed_rad_s - angular_speed)
                / (dive_config.dive_maximum_root_angular_speed_rad_s - 1.35),
                0.0,
                1.0,
            )
        )
        target_visible = cue_visible or shot_active
        if shot_active and float(data.qvel[35]) > 0.10:
            estimated_time_to_arrival = float(
                np.clip((4.44 - float(data.qpos[36])) / float(data.qvel[35]), 0.0, 1.2)
            )
        else:
            estimated_time_to_arrival = max(0.0, nominal_arrival_sec - time_sec)
        lateral_error = float(causal_target[1]) - float(data.qpos[1])
        if (
            not option_started
            and target_visible
            and abs(lateral_error) >= dive_config.activation_minimum_lateral_error_m
        ):
            option_started = True
            option_age = 0
        plastic = option_started and option_age <= plasticity_steps
        shaped = requested.clone()
        shaped[0] = (
            torch.maximum(
                torch.clamp(shaped[0], 0.0, 1.0),
                torch.tensor(dive_config.minimum_option_gate),
            )
            * authority
            if plastic
            else 0.0
        )
        shaped[1:] *= authority * shaped[0] if plastic else 0.0
        if plastic and option_age >= total_option_steps:
            recovery_steps = max(
                1,
                int(
                    round(
                        dive_config.actor_recovery_plasticity_sec
                        / exam_config.control_dt_sec
                    )
                ),
            )
            recovery_phase = float(
                np.clip((option_age - total_option_steps) / recovery_steps, 0.0, 1.0)
            )
            recovery_ramp = recovery_phase**2 * (3.0 - 2.0 * recovery_phase)
            shaped[1:] *= (
                dive_config.actor_recovery_residual_authority_scale * recovery_ramp
            )
        delta = torch.clamp(shaped - previous_actor, -0.18, 0.18)
        applied = previous_actor + 0.30 * delta
        gate = float(applied[0]) if option_started and option_age < total_option_steps else 0.0
        maximum_gate = max(maximum_gate, gate)
        drive_activation = float(
            np.clip(gate / dive_config.lateral_drive_full_activation_gate, 0.0, 1.0)
        )
        lateral_drive = (
            -math.copysign(dive_config.lateral_drive_scale, lateral_error) * drive_activation
            if target_visible and gate > 0.0
            else 0.0
        )
        gravity = np.asarray(
            (
                2.0 * (-qz * qx + qw * qy),
                -2.0 * (qz * qy + qw * qx),
                1.0 - 2.0 * (qw * qw + qz * qz),
            )
        )
        loco_observation = np.zeros(96, dtype=np.float64)
        loco_observation[:3] = data.qvel[3:6]
        loco_observation[3:6] = gravity
        loco_observation[7] = lateral_drive * maximum_lateral_command_mps
        loco_observation[9:38] = data.qpos[7:36][loco_order] - loco_default
        loco_observation[38:67] = data.qvel[6:35][loco_order]
        loco_observation[67:96] = loco_action
        with torch.inference_mode():
            encoded = locomotion.normalizer.forward(
                torch.as_tensor(loco_observation, dtype=torch.float32).unsqueeze(0)
            )
            sequence, (loco_hidden, loco_cell) = locomotion.rnn.forward__0(
                encoded.unsqueeze(0), (loco_hidden, loco_cell)
            )
            loco_action = np.asarray(
                torch.clamp(locomotion.actor.forward(sequence.squeeze(0)), -100.0, 100.0)[0]
            )
        stable_target = np.zeros(29, dtype=np.float64)
        stable_target[loco_order] = 0.25 * loco_action + loco_default
        target = stable_target.copy()
        if gate > 0.0:
            raw_phase = float(
                np.clip((option_age - hold_steps) / max(option_steps - 1, 1), 0.0, 1.0)
            )
            arm_phase = raw_phase
            intercept_phase = dive_config.intercept_phase_at_arrival
            phase_sync_active = bool(
                intercept_phase is not None
                and float(causal_target[2]) >= dive_config.phase_sync_minimum_target_height_m
            )
            if phase_sync_active:
                if intercept_phase is None:
                    raise RuntimeError("targeted dive phase synchronization lost its phase")
                arrival_raw_phase = float(
                    np.clip(
                        raw_phase
                        + estimated_time_to_arrival
                        / max(dive_config.option_duration_sec, exam_config.control_dt_sec),
                        0.05,
                        0.95,
                    )
                )
                if raw_phase <= arrival_raw_phase:
                    arm_phase = intercept_phase * raw_phase / arrival_raw_phase
                else:
                    arm_phase = intercept_phase + (1.0 - intercept_phase) * (
                        raw_phase - arrival_raw_phase
                    ) / (1.0 - arrival_raw_phase)
                arm_phase = max(arm_phase, previous_option_phase)
            if float(causal_target[2]) < 0.60:
                phase_scale = dive_config.low_shot_phase_scale
            elif float(causal_target[2]) < 1.10:
                phase_scale = dive_config.mid_shot_phase_scale
            else:
                phase_scale = dive_config.high_shot_phase_scale
            body_phase = float(np.clip(raw_phase * phase_scale, 0.0, 1.0))
            arm_phase = float(np.clip(arm_phase * phase_scale, 0.0, 1.0))
            previous_option_phase = arm_phase

            feature_values = {
                "torch": torch,
                "direction": torch.tensor((-1.0 if lateral_error < 0.0 else 1.0,)),
                "target_lateral_m": torch.tensor((float(causal_target[1]),)),
                "target_height_m": torch.tensor((float(causal_target[2]),)),
                "time_to_arrival_sec": torch.tensor((estimated_time_to_arrival,)),
                "root_lateral_m": torch.tensor((float(data.qpos[1]),)),
                "root_lateral_speed_mps": torch.tensor((float(data.qvel[1]),)),
                "pelvis_height_m": torch.tensor((float(data.qpos[2]),)),
                "upright_projection": torch.tensor((upright,)),
                "support_side": torch.tensor((support_side,)),
                "root_angular_speed_rad_s": torch.tensor((angular_speed,)),
            }
            features = targeted_dive_features_torch(
                phase=torch.tensor((body_phase,)), **feature_values
            )
            option_target = np.asarray(
                decode_goalkeeper_targeted_dive(
                    model=decoder,
                    checkpoint=decoder_checkpoint,
                    features=features,
                    residual_authority=decoder_group_authority,
                )[0]
            )
            if phase_sync_active:
                arm_target = np.asarray(
                    decode_goalkeeper_targeted_dive(
                        model=decoder,
                        checkpoint=decoder_checkpoint,
                        features=targeted_dive_features_torch(
                            phase=torch.tensor((arm_phase,)), **feature_values
                        ),
                        residual_authority=decoder_group_authority,
                    )[0]
                )
                option_target[15:29] = arm_target[15:29]
            target = stable_target + gate * group_scale * (option_target - stable_target)
            lower_body_command_boost = (
                dive_config.resolved_decoder_lower_body_command_scale
                - dive_config.anchor_lower_body_scale
            )
            if lower_body_command_boost > 0.0:
                anchor_target = np.asarray(
                    decode_goalkeeper_targeted_dive(
                        model=decoder,
                        checkpoint=decoder_checkpoint,
                        features=features,
                        residual_authority=0.0,
                    )[0]
                )
                target[:12] += (
                    gate * lower_body_command_boost * (option_target[:12] - anchor_target[:12])
                )
            if runtime_reach_atlas is not None:
                from rosclaw_soccer.training.goalkeeper_reach import (
                    task_space_reach_from_target_numpy,
                )

                relative = causal_target - np.asarray(data.qpos[:3], dtype=np.float64)
                normalized_reach = task_space_reach_from_target_numpy(
                    target_relative=relative.reshape(1, 3),
                    model=runtime_reach_atlas,
                )[0]
                reach_target = ready[15:29] + normalized_reach * np.asarray(
                    tuple(runtime_reach_atlas.effective_arm_limits_rad) * 2,
                    dtype=np.float64,
                )
                reach_progress = float(
                    np.clip(
                        (dive_config.runtime_reach_approach_horizon_sec - estimated_time_to_arrival)
                        / (
                            dive_config.runtime_reach_approach_horizon_sec
                            - dive_config.runtime_reach_full_lead_sec
                        ),
                        0.0,
                        1.0,
                    )
                )
                reach_progress = reach_progress * reach_progress * (3.0 - 2.0 * reach_progress)
                reach_gate = dive_config.runtime_reach_blend * reach_progress
                reach_gate *= float(
                    np.clip(
                        gate / dive_config.runtime_reach_full_activation_gate,
                        0.0,
                        1.0,
                    )
                )
                if not target_visible:
                    reach_gate = 0.0
                target[15:29] += reach_gate * (reach_target - target[15:29])
                maximum_runtime_reach_blend = max(maximum_runtime_reach_blend, reach_gate)
            if overhead_reach_prior is not None:
                from rosclaw_soccer.growth.mosaic_overhead_reach_prior import (
                    blend_g1_mosaic_overhead_reach_target,
                )

                overhead_activation = float(
                    np.clip(
                        gate / dive_config.runtime_reach_full_activation_gate,
                        0.0,
                        1.0,
                    )
                )
                target, _, overhead_gate = blend_g1_mosaic_overhead_reach_target(
                    target=target,
                    prior=overhead_reach_prior,
                    time_to_arrival_sec=estimated_time_to_arrival,
                    target_height_m=float(causal_target[2]),
                    blend=(
                        dive_config.overhead_reach_blend * overhead_activation
                        if target_visible
                        else 0.0
                    ),
                    minimum_target_height_m=(dive_config.overhead_reach_minimum_target_height_m),
                    full_target_height_m=(dive_config.overhead_reach_full_target_height_m),
                    joint_scales=(dive_config.overhead_reach_lower_body_scale,) * 12
                    + (dive_config.overhead_reach_waist_scale,) * 3
                    + (dive_config.overhead_reach_arm_scale,) * 14,
                )
                maximum_overhead_reach_blend = max(maximum_overhead_reach_blend, overhead_gate)
            if dive_config.lateral_drive_scale == 0.0:
                loco_hidden = torch.zeros((1, 1, 256), dtype=torch.float32)
                loco_cell = torch.zeros_like(loco_hidden)
                loco_action.fill(0.0)
        if arm_target_filter_enabled:
            arm_delta = np.clip(
                target[15:29] - previous_option_arm_target,
                -dive_config.maximum_arm_target_step_rad,
                dive_config.maximum_arm_target_step_rad,
            )
            target[15:29] = previous_option_arm_target + (
                dive_config.arm_target_filter_fraction * arm_delta
            )
            previous_option_arm_target = target[15:29].copy()
        target += np.asarray(applied[1:]) * residual_limits
        target = np.clip(target, joint_lower, joint_upper)
        for substep in range(exam_config.physics_substeps):
            substep_target = target
            if dive_config.substep_option_lower_body_guard_enabled:
                substep_angular_speed = float(np.linalg.norm(data.qvel[3:6]))
                fraction = float(
                    np.clip(
                        (
                            substep_angular_speed
                            - dive_config.substep_option_lower_body_guard_onset_rad_s
                        )
                        / (
                            dive_config.substep_option_lower_body_guard_ceiling_rad_s
                            - dive_config.substep_option_lower_body_guard_onset_rad_s
                        ),
                        0.0,
                        1.0,
                    )
                )
                lower_body_authority = 1.0 - fraction * (
                    1.0 - dive_config.substep_option_lower_body_minimum_scale
                )
                substep_target = target.copy()
                substep_target[:12] = stable_target[:12] + lower_body_authority * (
                    target[:12] - stable_target[:12]
                )
                minimum_substep_option_lower_body_authority = min(
                    minimum_substep_option_lower_body_authority,
                    lower_body_authority,
                )
            position_torque = kp * (substep_target - data.qpos[7:36])
            if dive_config.substep_upper_body_guard_enabled:
                substep_angular_speed = float(np.linalg.norm(data.qvel[3:6]))
                fraction = float(
                    np.clip(
                        (
                            substep_angular_speed
                            - dive_config.substep_upper_body_guard_onset_rad_s
                        )
                        / (
                            dive_config.substep_upper_body_guard_ceiling_rad_s
                            - dive_config.substep_upper_body_guard_onset_rad_s
                        ),
                        0.0,
                        1.0,
                    )
                )
                substep_authority = 1.0 - fraction * (
                    1.0 - dive_config.substep_upper_body_minimum_position_scale
                )
                position_torque[15:] *= substep_authority
                minimum_substep_arm_authority = min(
                    minimum_substep_arm_authority,
                    substep_authority,
                )
            requested_torque = position_torque - kd * data.qvel[6:35]
            data.ctrl[:] = np.clip(requested_torque, -limits, limits)
            mujoco.mj_step(model, data)
            robot_now, hand_now, landing, forbidden, support_side = _contacts(model, data)
            del landing
            physics_time = time_sec + (substep + 1) * (
                exam_config.control_dt_sec / exam_config.physics_substeps
            )
            robot_contact |= robot_now
            hand_contact |= hand_now
            contact_before_plane |= bool(robot_now and float(data.qpos[36]) <= 4.60)
            forbidden_latched |= forbidden
            if robot_now and first_contact_time is None:
                first_contact_time = physics_time
                shot_active = False
            target_point = np.asarray((4.44, float(case["target_y_m"]), float(case["target_z_m"])))
            hand_errors = tuple(
                np.asarray(data.geom_xpos[int(model.geom(name).id)]) - target_point
                for name in ("left_hand_collision", "right_hand_collision")
            )
            selected_error = min(hand_errors, key=np.linalg.norm)
            distance = float(np.linalg.norm(selected_error))
            if distance < minimum_hand_distance:
                minimum_hand_distance = distance
                minimum_hand_time = physics_time
            sample_error = abs(physics_time - arrival_sec)
            if sample_error < arrival_sample_error:
                arrival_sample_error = sample_error
                hand_distance_at_arrival = distance
                arrival_error = selected_error.copy()
            minimum_pelvis = min(minimum_pelvis, float(data.qpos[2]))
            maximum_angular = max(maximum_angular, float(np.linalg.norm(data.qvel[3:6])))
        option_age += int(option_started)
        previous_actor = applied.detach().cpu()
        previous_target = target.copy()
        linear_speed = float(np.linalg.norm(data.qvel[:3]))
        angular_speed = float(np.linalg.norm(data.qvel[3:6]))
        in_exception = option_started and option_age <= exception_steps and maximum_gate > 0.05
        envelope = bool(
            float(data.qpos[2]) >= dive_config.dive_minimum_pelvis_height_m
            and upright >= dive_config.dive_minimum_upright_projection
            and linear_speed <= dive_config.dive_maximum_root_linear_speed_mps
            and angular_speed <= dive_config.dive_maximum_root_angular_speed_rad_s
        )
        strict_posture = bool(float(data.qpos[2]) >= 0.60 and upright >= 0.78)
        recovered_posture = bool(strict_posture and linear_speed <= 0.35 and angular_speed <= 0.75)
        unsafe_latched |= bool(
            forbidden_latched or (not envelope if in_exception else not strict_posture)
        )
        stable_steps = (
            stable_steps + 1 if option_age > total_option_steps and recovered_posture else 0
        )
    finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
    recovered = bool(stable_steps >= 10)
    safe = bool(finite and not unsafe_latched and recovered)
    saved = bool(contact_before_plane and safe)
    timing_error = minimum_hand_time - arrival_sec
    if saved:
        failure_category = "SAVED"
    elif not safe:
        failure_category = "SAFETY_OR_RECOVERY"
    elif minimum_hand_distance <= 0.18 and abs(timing_error) > 0.06:
        failure_category = "TIMING_MISMATCH"
    elif abs(float(arrival_error[2])) > 1.15 * abs(float(arrival_error[1])):
        failure_category = "HEIGHT_GAP"
    elif abs(float(arrival_error[1])) > 1.15 * abs(float(arrival_error[2])):
        failure_category = "LATERAL_GAP"
    else:
        failure_category = "CONTACT_GEOMETRY"
    return {
        **case,
        "saved": saved,
        "robot_contact": robot_contact,
        "hand_contact": hand_contact,
        "safe": safe,
        "prelaunch_violation": False,
        "forbidden_contact": forbidden_latched,
        "recovered": recovered,
        "minimum_pelvis_height_m": minimum_pelvis,
        "maximum_root_angular_speed_rad_s": maximum_angular,
        "maximum_requested_torque_fraction": 0.0,
        "minimum_hand_distance_m": minimum_hand_distance,
        "minimum_hand_distance_time_sec": minimum_hand_time,
        "hand_distance_at_arrival_m": hand_distance_at_arrival,
        "hand_target_error_at_arrival_xyz_m": arrival_error.tolist(),
        "minimum_hand_target_error_xyz_m": arrival_error.tolist(),
        "minimum_hand_timing_error_sec": timing_error,
        "first_robot_contact_time_sec": first_contact_time,
        "maximum_applied_option_gate": maximum_gate,
        "maximum_applied_runtime_reach_blend": maximum_runtime_reach_blend,
        "maximum_applied_overhead_reach_blend": maximum_overhead_reach_blend,
        "minimum_substep_arm_position_authority": minimum_substep_arm_authority,
        "minimum_substep_option_lower_body_authority": (
            minimum_substep_option_lower_body_authority
        ),
        "failure_category": failure_category,
    }


def run_goalkeeper_targeted_dive_actor_exam(
    *,
    asset_root: Path,
    locomotion_policy_path: Path,
    targeted_dive_checkpoint_path: Path,
    actor_checkpoint_path: Path,
    targeted_dive_training_report_path: Path,
    actor_training_report_path: Path,
    output_path: Path,
    config: GoalkeeperTargetedDiveExamConfig | None = None,
) -> dict[str, Any]:
    import torch

    active = config or GoalkeeperTargetedDiveExamConfig()
    actor, actor_checkpoint = _load_actor(path=actor_checkpoint_path, device=torch.device("cpu"))
    decoder, decoder_checkpoint = load_goalkeeper_targeted_dive(
        checkpoint_path=targeted_dive_checkpoint_path, device=torch.device("cpu")
    )
    targeted_report = _load_bound_report(
        targeted_dive_training_report_path, label="targeted dive training"
    )
    actor_report = _load_bound_report(actor_training_report_path, label="actor training")
    targeted_checkpoint_hash = hash_bytes(targeted_dive_checkpoint_path.read_bytes())
    actor_checkpoint_hash = hash_bytes(actor_checkpoint_path.read_bytes())
    if (
        targeted_report.get("checkpoint_hash") != targeted_checkpoint_hash
        or hash_json(targeted_report.get("config"))
        != hash_json(decoder_checkpoint.get("training_config"))
        or hash_json(targeted_report.get("curriculum"))
        != hash_json(decoder_checkpoint.get("training_metadata"))
    ):
        raise ValueError("targeted dive checkpoint/report binding is invalid")
    if (
        actor_report.get("candidate_checkpoint_hash") != actor_checkpoint_hash
        or hash_json(actor_report.get("training_config"))
        != hash_json(actor_checkpoint.get("training_config"))
        or hash_json(actor_report.get("world_config"))
        != hash_json(actor_checkpoint.get("world_config"))
        or actor_report.get("environment_summary", {}).get("targeted_dive_checkpoint_hash")
        != targeted_checkpoint_hash
    ):
        raise ValueError("targeted dive actor checkpoint/report binding is invalid")
    training_config = actor_checkpoint.get("training_config", {})
    dive_config = GoalkeeperTargetedDiveRLConfig(
        option_duration_sec=float(training_config["targeted_dive_option_duration_sec"]),
        phase_hold_sec=float(training_config.get("targeted_dive_phase_hold_sec", 0.0)),
        actor_recovery_plasticity_sec=float(
            training_config.get("targeted_dive_actor_recovery_plasticity_sec", 0.0)
        ),
        actor_recovery_residual_authority_scale=float(
            training_config.get(
                "targeted_dive_actor_recovery_residual_authority_scale",
                0.50,
            )
        ),
        nominal_shot_flight_time_sec=float(
            training_config.get("targeted_dive_nominal_shot_flight_time_sec", 0.47)
        ),
        intercept_phase_at_arrival=(
            None
            if training_config.get("targeted_dive_intercept_phase_at_arrival") is None
            else float(training_config["targeted_dive_intercept_phase_at_arrival"])
        ),
        phase_sync_minimum_target_height_m=float(
            training_config.get("targeted_dive_phase_sync_minimum_target_height_m", 0.60)
        ),
        posture_exception_duration_sec=float(
            training_config["targeted_dive_posture_exception_duration_sec"]
        ),
        decoder_residual_authority=float(
            training_config["targeted_dive_decoder_residual_authority"]
        ),
        decoder_lower_body_residual_authority=(
            None
            if training_config.get("targeted_dive_decoder_lower_body_residual_authority") is None
            else float(training_config["targeted_dive_decoder_lower_body_residual_authority"])
        ),
        decoder_lower_body_command_scale=(
            None
            if training_config.get("targeted_dive_decoder_lower_body_command_scale") is None
            else float(training_config["targeted_dive_decoder_lower_body_command_scale"])
        ),
        decoder_waist_residual_authority=(
            None
            if training_config.get("targeted_dive_decoder_waist_residual_authority") is None
            else float(training_config["targeted_dive_decoder_waist_residual_authority"])
        ),
        decoder_arm_residual_authority=(
            None
            if training_config.get("targeted_dive_decoder_arm_residual_authority") is None
            else float(training_config["targeted_dive_decoder_arm_residual_authority"])
        ),
        actor_residual_scale=float(training_config["targeted_dive_actor_residual_scale"]),
        anchor_lower_body_scale=float(
            training_config.get("targeted_dive_anchor_lower_body_scale", 0.25)
        ),
        anchor_waist_scale=float(training_config.get("targeted_dive_anchor_waist_scale", 0.50)),
        anchor_arm_scale=float(training_config.get("targeted_dive_anchor_arm_scale", 1.00)),
        minimum_option_gate=float(training_config.get("targeted_dive_minimum_option_gate", 0.0)),
        runtime_reach_blend=float(training_config.get("targeted_dive_runtime_reach_blend", 0.0)),
        overhead_reach_prior_path=training_config.get("targeted_dive_overhead_reach_prior"),
        overhead_reach_blend=float(training_config.get("targeted_dive_overhead_reach_blend", 0.0)),
        overhead_reach_minimum_target_height_m=float(
            training_config.get("targeted_dive_overhead_reach_minimum_target_height_m", 1.10)
        ),
        overhead_reach_full_target_height_m=float(
            training_config.get("targeted_dive_overhead_reach_full_target_height_m", 1.25)
        ),
        overhead_reach_lower_body_scale=float(
            training_config.get("targeted_dive_overhead_reach_lower_body_scale", 0.0)
        ),
        overhead_reach_waist_scale=float(
            training_config.get("targeted_dive_overhead_reach_waist_scale", 0.25)
        ),
        overhead_reach_arm_scale=float(
            training_config.get("targeted_dive_overhead_reach_arm_scale", 1.0)
        ),
        maximum_arm_target_step_rad=float(
            training_config.get("targeted_dive_maximum_arm_target_step_rad", 0.10)
        ),
        arm_target_filter_fraction=float(
            training_config.get("targeted_dive_arm_target_filter_fraction", 0.50)
        ),
        lateral_drive_scale=float(training_config.get("targeted_dive_lateral_drive_scale", 0.0)),
        runtime_lateral_lunge_blend=float(
            training_config.get("targeted_dive_runtime_lateral_lunge_blend", 0.0)
        ),
        runtime_lateral_lunge_hip_roll_rad=float(
            training_config.get("targeted_dive_runtime_lateral_lunge_hip_roll_rad", 0.18)
        ),
        runtime_lateral_lunge_ankle_roll_rad=float(
            training_config.get("targeted_dive_runtime_lateral_lunge_ankle_roll_rad", 0.12)
        ),
        runtime_lateral_lunge_approach_horizon_sec=float(
            training_config.get("targeted_dive_runtime_lateral_lunge_approach_horizon_sec", 0.90)
        ),
        substep_upper_body_guard_enabled=bool(
            training_config.get("targeted_dive_substep_upper_body_guard_enabled", False)
        ),
        substep_upper_body_guard_onset_rad_s=float(
            training_config.get("targeted_dive_substep_upper_body_guard_onset_rad_s", 1.80)
        ),
        substep_upper_body_guard_ceiling_rad_s=float(
            training_config.get("targeted_dive_substep_upper_body_guard_ceiling_rad_s", 3.00)
        ),
        substep_upper_body_minimum_position_scale=float(
            training_config.get(
                "targeted_dive_substep_upper_body_minimum_position_scale", 0.05
            )
        ),
        substep_option_lower_body_guard_enabled=bool(
            training_config.get(
                "targeted_dive_substep_option_lower_body_guard_enabled", False
            )
        ),
        substep_option_lower_body_guard_onset_rad_s=float(
            training_config.get(
                "targeted_dive_substep_option_lower_body_guard_onset_rad_s", 2.40
            )
        ),
        substep_option_lower_body_guard_ceiling_rad_s=float(
            training_config.get(
                "targeted_dive_substep_option_lower_body_guard_ceiling_rad_s", 3.30
            )
        ),
        substep_option_lower_body_minimum_scale=float(
            training_config.get(
                "targeted_dive_substep_option_lower_body_minimum_scale", 0.0
            )
        ),
        low_shot_phase_scale=float(training_config.get("targeted_dive_low_shot_phase_scale", 1.0)),
        mid_shot_phase_scale=float(training_config.get("targeted_dive_mid_shot_phase_scale", 1.0)),
        high_shot_phase_scale=float(
            training_config.get("targeted_dive_high_shot_phase_scale", 1.0)
        ),
    )
    world_config = actor_checkpoint.get("world_config", {})
    maximum_lateral_command_mps = float(world_config.get("maximum_lateral_command_mps", 0.40))
    model = build_g1_stadium_model(asset_root.expanduser().resolve())
    locomotion_path = locomotion_policy_path.expanduser().resolve()
    locomotion = torch.jit.load(  # type: ignore[no-untyped-call]
        str(locomotion_path), map_location="cpu"
    )
    locomotion.eval()
    runtime_reach_atlas = None
    if dive_config.runtime_reach_blend > 0.0:
        from rosclaw_soccer.training.goalkeeper_reach import (
            GoalkeeperReachAtlasConfig,
            GoalkeeperReachConfig,
            build_g1_task_space_reach_atlas,
        )

        runtime_reach_atlas = build_g1_task_space_reach_atlas(
            asset_root.expanduser().resolve(),
            config=GoalkeeperReachConfig(
                damping=0.12,
                reach_gain=0.95,
                maximum_position_error_m=0.75,
                support_arm_scale=0.60,
                central_support_scale=0.95,
                residual_scale=1.0,
                arm_authority_scale=0.88,
                workspace_scale=2.50,
            ),
            atlas_config=GoalkeeperReachAtlasConfig(
                interpolation_neighbors=42,
                interpolation_kernel="gaussian",
                interpolation_temperature=0.75,
                multistart_count=12,
            ),
        )
    overhead_reach_prior = None
    if dive_config.overhead_reach_prior_path is not None:
        from rosclaw_soccer.growth.mosaic_overhead_reach_prior import (
            load_g1_mosaic_overhead_reach_prior,
        )
        from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash

        overhead_reach_prior = load_g1_mosaic_overhead_reach_prior(
            Path(dive_config.overhead_reach_prior_path)
        )
        scene = asset_root.expanduser().resolve() / "g1_description" / "scene_with_ball.xml"
        if (
            overhead_reach_prior.body_hash != g1_body_hash(asset_root.expanduser().resolve())
            or overhead_reach_prior.physics_scene_hash != hash_bytes(scene.read_bytes())
            or actor_report.get("environment_summary", {})
            .get("mosaic_overhead_reach", {})
            .get("prior_hash")
            != overhead_reach_prior.prior_hash
        ):
            raise ValueError("targeted dive actor overhead prior binding is invalid")
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
        _rollout_actor_case(
            model=model,
            actor=actor,
            decoder=decoder,
            decoder_checkpoint=decoder_checkpoint,
            locomotion=locomotion,
            case=case,
            exam_config=active,
            dive_config=dive_config,
            maximum_lateral_command_mps=maximum_lateral_command_mps,
            runtime_reach_atlas=runtime_reach_atlas,
            overhead_reach_prior=overhead_reach_prior,
            arm_target_filter_enabled=(
                "targeted_dive_maximum_arm_target_step_rad" in training_config
            ),
        )
        for case in cases
    ]
    baseline = _summary(baseline_rows)
    candidate = _summary(candidate_rows)
    strata_pass = all(
        float(item["save_rate"]) >= active.minimum_target_save_rate
        for item in candidate["strata"].values()
    )
    passed = bool(
        targeted_report.get("fit_gate_passed", False)
        and strata_pass
        and candidate["safe_rate"] == 1.0
        and candidate["recovery_rate"] == 1.0
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.goalkeeper_targeted_dive_actor_exam.v2",
        "physics_backend": "mujoco_cpu",
        "physics_scene_hash": g1_stadium_scene_hash(asset_root.expanduser().resolve()),
        "config": asdict(active),
        "config_hash": active.config_hash,
        "dive_rl_config": asdict(dive_config),
        "dive_rl_config_hash": dive_config.config_hash,
        "challenge": "FIRST_SHOT_FAR_CORNER_LOW_MID_HIGH_ONLY",
        "target_information_contract": (
            "BOUNDED_0P30_SEC_PRE_KICK_CUE_NOMINAL_FLIGHT_PRIOR_THEN_CAUSAL_BALL_STATE"
        ),
        "paired_seed_count": len(cases),
        "targeted_dive_fit_gate_passed": bool(targeted_report.get("fit_gate_passed", False)),
        "targeted_dive_training_report_hash": targeted_report.get("report_hash"),
        "actor_training_report_hash": actor_report.get("report_hash"),
        "actor_checkpoint_hash": actor_checkpoint_hash,
        "targeted_dive_checkpoint_hash": targeted_checkpoint_hash,
        "locomotion_policy_hash": hash_bytes(locomotion_path.read_bytes()),
        "baseline": baseline,
        "candidate": candidate,
        "minimum_target_save_rate": active.minimum_target_save_rate,
        "passed_80_percent_each_stratum": strata_pass,
        "passed": passed,
        "promotion_status": (
            "OPTION_PENDING_BROADER_VALIDATION"
            if passed
            else "REJECTED_BY_TARGETED_DIVE_ACTOR_CPU_EXAM"
        ),
        "video_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--locomotion-policy", type=Path, required=True)
    parser.add_argument("--targeted-dive-checkpoint", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--targeted-dive-training-report", type=Path, required=True)
    parser.add_argument("--actor-training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots-per-stratum", type=int, default=20)
    args = parser.parse_args()
    report = run_goalkeeper_targeted_dive_actor_exam(
        asset_root=args.asset_root,
        locomotion_policy_path=args.locomotion_policy,
        targeted_dive_checkpoint_path=args.targeted_dive_checkpoint,
        actor_checkpoint_path=args.actor_checkpoint,
        targeted_dive_training_report_path=args.targeted_dive_training_report,
        actor_training_report_path=args.actor_training_report,
        output_path=args.output,
        config=GoalkeeperTargetedDiveExamConfig(shots_per_stratum=args.shots_per_stratum),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_goalkeeper_targeted_dive_actor_exam"]
