"""CPU MuJoCo diagnostic for lossless goalkeeper motion references.

This is a deliberately simple joint-space PD baseline.  Its purpose is to
measure the reality gap before training a whole-body tracker, not to qualify a
reference trajectory as a deployable controller.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.skills.athlete_foundation.full_body_goalkeeper_motion import (
    FullBodyGoalkeeperMotionLibrary,
    load_full_body_goalkeeper_motion_library,
)
from rosclaw_soccer.training.joint_guard import project_joint_safe_torque_numpy

_KP = np.asarray(
    (
        180,
        180,
        150,
        180,
        45,
        45,
        180,
        180,
        150,
        180,
        45,
        45,
        90,
        55,
        55,
        45,
        45,
        35,
        45,
        15,
        15,
        15,
        45,
        45,
        35,
        45,
        15,
        15,
        15,
    ),
    dtype=np.float64,
)
_KD = np.asarray((5,) * 12 + (3,) * 3 + (2,) * 14, dtype=np.float64)
_LIMITS = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)


@dataclass(frozen=True)
class FullBodyTrackingExamConfig:
    family_names: tuple[str, ...] = (
        "lefthand",
        "righthand",
        "leftjump",
        "rightjump",
        "leftstep",
        "rightstep",
    )
    recovery_duration_sec: float = 1.0
    maximum_joint_rmse_rad: float = 0.35
    maximum_mean_foot_slip_mps: float = 0.15
    maximum_torque_saturation_rate: float = 0.10
    maximum_p95_root_angular_speed_rad_s: float = 3.0
    maximum_joint_jerk_rms_rad_s3: float = 5000.0
    minimum_pelvis_height_m: float = 0.55
    minimum_recovery_rate: float = 0.80
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.full_body_tracking_exam_config.v1"

    def __post_init__(self) -> None:
        if len(self.family_names) != 6 or len(set(self.family_names)) != 6:
            raise ValueError("full-body tracking exam requires six unique motion families")
        if not 0.5 <= self.recovery_duration_sec <= 3.0:
            raise ValueError("full-body tracking recovery duration is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("full-body tracking exam must remain SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


DEFAULT_FULL_BODY_TRACKING_EXAM_CONFIG = FullBodyTrackingExamConfig()


def run_full_body_tracking_exam(
    *,
    asset_root: Path,
    motion_manifest_path: Path,
    output_path: Path,
    config: FullBodyTrackingExamConfig = DEFAULT_FULL_BODY_TRACKING_EXAM_CONFIG,
) -> dict[str, Any]:
    """Run all six motions through a bounded CPU-physics PD baseline."""

    import mujoco

    root = asset_root.expanduser().resolve()
    library = load_full_body_goalkeeper_motion_library(motion_manifest_path)
    scene_path = root / "g1_description" / "scene_with_ball.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = 0.002
    source_mask = np.asarray(library.manifest.source_joint_mask, dtype=np.bool_)
    family_reports: list[dict[str, Any]] = []
    all_joint_errors: list[float] = []
    all_root_angular: list[float] = []
    all_joint_jerk: list[float] = []
    all_foot_slip: list[float] = []
    total_torque_steps = 0
    saturated_torque_steps = 0
    global_minimum_pelvis = math.inf
    recovery_count = 0

    for family in config.family_names:
        report = _run_family(model=model, library=library, family=family, source_mask=source_mask)
        family_reports.append(report["summary"])
        all_joint_errors.extend(report["joint_errors"])
        all_root_angular.extend(report["root_angular"])
        all_joint_jerk.extend(report["joint_jerk"])
        all_foot_slip.extend(report["foot_slip"])
        total_torque_steps += int(report["torque_steps"])
        saturated_torque_steps += int(report["saturated_steps"])
        global_minimum_pelvis = min(
            global_minimum_pelvis, float(report["summary"]["minimum_pelvis_height_m"])
        )
        recovery_count += int(bool(report["summary"]["recovered_upright"]))

    joint_rmse = float(np.sqrt(np.mean(np.square(all_joint_errors))))
    torque_saturation_rate = saturated_torque_steps / max(1, total_torque_steps)
    p95_root_angular = float(np.percentile(all_root_angular, 95))
    recovery_rate = recovery_count / len(config.family_names)
    reasons: list[str] = []
    mean_foot_slip = float(np.mean(all_foot_slip)) if all_foot_slip else 0.0
    joint_jerk_rms = float(np.sqrt(np.mean(np.square(all_joint_jerk))))
    if any(bool(item["fell"]) for item in family_reports):
        reasons.append("fall_detected")
    if joint_rmse > config.maximum_joint_rmse_rad:
        reasons.append("joint_rmse_above_ceiling")
    if mean_foot_slip > config.maximum_mean_foot_slip_mps:
        reasons.append("foot_slip_above_ceiling")
    if torque_saturation_rate > config.maximum_torque_saturation_rate:
        reasons.append("torque_saturation_above_ceiling")
    if p95_root_angular > config.maximum_p95_root_angular_speed_rad_s:
        reasons.append("root_angular_speed_above_ceiling")
    if joint_jerk_rms > config.maximum_joint_jerk_rms_rad_s3:
        reasons.append("joint_jerk_above_ceiling")
    if global_minimum_pelvis < config.minimum_pelvis_height_m:
        reasons.append("pelvis_height_below_floor")
    if recovery_rate < config.minimum_recovery_rate:
        reasons.append("recovery_below_floor")
    report = {
        "schema_version": "rosclaw_soccer.full_body_tracking_exam.v1",
        "config": asdict(config),
        "config_hash": config.config_hash,
        "body_hash": g1_body_hash(root),
        "physics_scene_hash": hash_bytes(scene_path.read_bytes()),
        "motion_manifest_hash": library.manifest.manifest_hash,
        "physics_backend": "mujoco_cpu",
        "control_mode": "JOINT_PD_REFERENCE_BASELINE",
        "physical_truth": True,
        "family_reports": family_reports,
        "aggregate": {
            "joint_rmse_rad": joint_rmse,
            "mean_foot_slip_mps": mean_foot_slip,
            "minimum_pelvis_height_m": global_minimum_pelvis,
            "peak_torque_fraction": max(
                float(item["peak_torque_fraction"]) for item in family_reports
            ),
            "torque_saturation_rate": torque_saturation_rate,
            "p95_root_angular_speed_rad_s": p95_root_angular,
            "joint_jerk_rms_rad_s3": joint_jerk_rms,
            "recovery_rate": recovery_rate,
            "finite_state": all(bool(item["finite_state"]) for item in family_reports),
        },
        "passed": not reasons,
        "status": "PHYSICS_QUALIFIED" if not reasons else "PHYSICS_UNQUALIFIED",
        "reasons": reasons,
        "interpretation": (
            "REFERENCE_REQUIRES_LEARNED_WHOLE_BODY_TRACKER"
            if reasons
            else "REFERENCE_TRACKABLE_BY_PD_BASELINE"
        ),
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    target = output_path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return report


def _run_family(
    *,
    model: Any,
    library: FullBodyGoalkeeperMotionLibrary,
    family: str,
    source_mask: np.ndarray[Any, np.dtype[np.bool_]],
) -> dict[str, Any]:
    import mujoco

    data = mujoco.MjData(model)
    frame_count = dict(library.manifest.family_frame_counts)[family]
    duration = (frame_count - 1) / library.manifest.source_frame_rate_hz
    initial = library.sample(family, time_sec=0.0)
    data.qpos[:] = model.qpos0
    data.qpos[:3] = initial.root_position_local
    xyzw = initial.root_quaternion_xyzw
    data.qpos[3:7] = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
    data.qpos[7:36] = initial.qpos_29
    if model.nq >= 43:
        data.qpos[36:43] = (-20.0, 0.0, 0.115, 1.0, 0.0, 0.0, 0.0)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    left_foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    joint_errors: list[float] = []
    root_angular: list[float] = []
    joint_jerk: list[float] = []
    foot_slip: list[float] = []
    previous_acceleration = np.zeros(29, dtype=np.float64)
    previous_velocity = np.asarray(data.qvel[6:35], dtype=np.float64).copy()
    minimum_pelvis = float(data.qpos[2])
    peak_torque_fraction = 0.0
    torque_steps = 0
    saturated_steps = 0
    finite_state = True
    fell = False
    elapsed = 0.0
    total_duration = duration + 1.0
    while elapsed <= total_duration + 1e-12:
        in_reference = elapsed <= duration
        frame = library.sample(family, time_sec=min(elapsed, duration))
        target = frame.qpos_29 if in_reference else initial.qpos_29
        target_velocity = frame.qvel_29 if in_reference else np.zeros(29, dtype=np.float64)
        target = np.clip(target, model.jnt_range[1:30, 0], model.jnt_range[1:30, 1])
        joint_position = np.asarray(data.qpos[7:36], dtype=np.float64)
        joint_velocity = np.asarray(data.qvel[6:35], dtype=np.float64)
        torque = _KP * (target - joint_position) + _KD * (target_velocity - joint_velocity)
        torque, _ = project_joint_safe_torque_numpy(
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            commanded_torque=torque,
            joint_ranges=np.asarray(model.jnt_range[1:30]),
            limited=model.jnt_limited[1:30].astype(bool),
            margin_rad=0.04,
            prediction_horizon_sec=0.06,
            boundary_kp=80.0,
            boundary_kd=6.0,
        )
        clipped = np.clip(torque, -_LIMITS, _LIMITS)
        fractions = np.abs(clipped) / _LIMITS
        peak_torque_fraction = max(peak_torque_fraction, float(np.max(fractions)))
        saturated_steps += int(bool(np.any(fractions >= 0.999)))
        torque_steps += 1
        data.ctrl[:] = clipped
        mujoco.mj_step(model, data)
        position = np.asarray(data.qpos[7:36], dtype=np.float64)
        joint_errors.extend((position[source_mask] - target[source_mask]).tolist())
        root_angular.append(float(np.linalg.norm(data.qvel[3:6])))
        acceleration = (np.asarray(data.qvel[6:35]) - previous_velocity) / model.opt.timestep
        joint_jerk.extend(((acceleration - previous_acceleration) / model.opt.timestep).tolist())
        previous_acceleration = acceleration.copy()
        previous_velocity = np.asarray(data.qvel[6:35]).copy()
        for body_id in (left_foot, right_foot):
            if body_id >= 0 and data.xpos[body_id, 2] < 0.14:
                foot_slip.append(float(np.linalg.norm(data.cvel[body_id, 3:5])))
        minimum_pelvis = min(minimum_pelvis, float(data.qpos[2]))
        finite_state = finite_state and bool(
            np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
        )
        quaternion = np.asarray(data.qpos[3:7], dtype=np.float64)
        upright = 2.0 * (quaternion[0] ** 2 + quaternion[3] ** 2) - 1.0
        fell = bool(fell or data.qpos[2] < 0.45 or upright < 0.35 or not finite_state)
        if not finite_state:
            break
        elapsed += model.opt.timestep
    final_joint_error = float(
        np.sqrt(np.mean(np.square(np.asarray(data.qpos[7:36]) - initial.qpos_29)))
    )
    quaternion = np.asarray(data.qpos[3:7], dtype=np.float64)
    final_upright = 2.0 * (quaternion[0] ** 2 + quaternion[3] ** 2) - 1.0
    recovered = bool(data.qpos[2] >= 0.60 and final_upright >= 0.75 and final_joint_error <= 0.35)
    lower_body_delta = np.asarray(
        library.sample(family, time_sec=duration).qpos_29[:12]
        - initial.qpos_29[:12]
    )
    return {
        "summary": {
            "family": family,
            "source_frame_count": frame_count,
            "source_duration_sec": duration,
            "source_lower_body_endpoint_rms_rad": float(
                np.sqrt(np.mean(np.square(lower_body_delta)))
            ),
            "joint_rmse_rad": float(np.sqrt(np.mean(np.square(joint_errors)))),
            "minimum_pelvis_height_m": minimum_pelvis,
            "peak_torque_fraction": peak_torque_fraction,
            "p95_root_angular_speed_rad_s": float(np.percentile(root_angular, 95)),
            "final_joint_error_rad": final_joint_error,
            "recovered_upright": recovered,
            "fell": bool(fell),
            "finite_state": bool(finite_state),
        },
        "joint_errors": joint_errors,
        "root_angular": root_angular,
        "joint_jerk": joint_jerk,
        "foot_slip": foot_slip,
        "torque_steps": torque_steps,
        "saturated_steps": saturated_steps,
    }


__all__ = ["FullBodyTrackingExamConfig", "run_full_body_tracking_exam"]
