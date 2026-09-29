"""Recompute the R1 contact teacher in a SIM_ONLY, fixed-decision proxy."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rsi_mjx_contact_replay_smoke import _contact_masks
from rsi_r1_pd_proxy_search_v140 import load_tape

from rosclaw_soccer.growth.locomotion_contact_teacher import (
    G1LocomotionContactTeacherConfig,
    contact_tracking_adjustment,
    locomotion_contact_teacher_effect,
)
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.training.receiving_classroom import r1_contact_tap_receiving_configuration
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

SCHEMA = "rosclaw_soccer.rsi.r1_dynamic_teacher_proxy_v141.result.v1"
START = 15
STOP = 60
RIGHT_LEG = np.arange(6, 12)


def teacher_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    row: np.ndarray,
    base: G1LocomotionContactTeacherConfig,
    pd_position_torque: np.ndarray,
) -> np.ndarray:
    """Recompute continuous torque from the recorded *discrete* teacher choice."""
    if row.shape != (17,) or not np.isfinite(row).all() or row[0] not in (0.0, 1.0):
        raise ValueError("finite recorded same-substep teacher input required")
    if row[0] == 0.0:
        return np.zeros(29, dtype=np.float64)
    if (
        row[1] not in (1.0, 2.0)
        or row[2] not in (1.0, 2.0)
        or row[5] not in (-1.0, 1.0)
        or row[6] not in (0.0, 1.0)
    ):
        raise ValueError("bounded teacher foot, mode, lateral sign, and contact event required")
    config = replace(
        base,
        strike_foot_speed_mps=float(row[9]),
        aim_yaw_bias_rad=float(row[10]),
        receive_follow_through_speed_mps=float(row[11]),
        receive_ankle_lateral_offset_m=float(row[12]),
        velocity_damping_n_per_mps=float(row[13]),
        maximum_task_force_n=float(row[14]),
        maximum_joint_residual_nm=float(row[15]),
        contact_leg_stiffness_scale=float(row[16]),
    )
    ankle = model.body("left_ankle_roll_link" if row[1] == 1 else "right_ankle_roll_link").id
    effect = locomotion_contact_teacher_effect(
        model=model,
        data=data,
        ankle_body_id=int(ankle),
        actuated_dof_indices=np.arange(6, 35, dtype=np.int64),
        ball_position_m=np.asarray(data.qpos[36:39], dtype=np.float64),
        ball_velocity_mps=np.asarray(data.qvel[35:38], dtype=np.float64),
        desired_ball_direction_xy=row[3:5].copy(),
        contact_mode="receive" if row[2] == 1 else "strike",
        local_lateral_sign=float(row[5]),
        contact_recent=bool(row[6]),
        config=config,
        strike_progress=None if row[7] < 0 else float(row[7]),
        receive_capture_progress=None if row[8] < 0 else float(row[8]),
    )
    extra = effect.torque_nm.copy()
    if effect.active and config.contact_leg_stiffness_scale < 1:
        extra += contact_tracking_adjustment(
            pd_position_torque, use_left=row[1] == 1, scale=config.contact_leg_stiffness_scale
        )
    return extra


def replay(
    model: mujoco.MjModel, tape: dict[str, np.ndarray], weights: np.ndarray, *, amplitude_rad: float
) -> dict[str, Any]:
    if (
        weights.shape != (6,)
        or not np.isfinite(weights).all()
        or np.any(np.abs(weights) > 1)
        or amplitude_rad not in (0.05, 0.1)
    ):
        raise ValueError("bounded six-joint SIM_ONLY intervention required")
    teacher = np.asarray(tape["motor_teacher_inputs"], dtype=np.float64)
    if teacher.shape[1:] != (17,) or len(teacher) < STOP * 10:
        raise ValueError("complete same-substep teacher evidence required")
    _, base_config = r1_contact_tap_receiving_configuration()
    data = mujoco.MjData(model)
    data.qpos[:] = tape["observation_qpos"][0]
    data.qvel[:] = tape["observation_qvel"][0]
    data.time = START * 0.02
    mujoco.mj_forward(model, data)
    ball_geom, robot_mask, foot_mask = _contact_masks(model)
    guarded = 0.85 * np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    wrench = np.zeros(6, dtype=np.float64)
    first_foot: int | None = None
    nonfoot: set[int] = set()
    minimum_height = float(data.qpos[2])
    root_error: list[float] = []
    ball_error: list[float] = []
    max_torque_error = 0.0
    max_teacher_error = 0.0
    peak_teacher_error_detail: dict[str, Any] | None = None
    for frame in range(START, STOP):
        sample = frame - START
        root_error.append(
            float(np.linalg.norm(data.qpos[:3] - tape["observation_qpos"][sample, :3]))
        )
        ball_error.append(
            float(np.linalg.norm(data.qpos[36:39] - tape["observation_qpos"][sample, 36:39]))
        )
        ramp = min(1.0, max(0.0, (frame - START) / 6.0))
        fade = 1.0 if first_foot is None else max(0.0, 1.0 - (frame - first_foot) / 10.0)
        delta = amplitude_rad * ramp * fade * weights
        for substep in range(10):
            index = frame * 10 + substep
            kp = tape["motor_kp"][index]
            kd = tape["motor_kd"][index]
            target = tape["motor_pd_target_rad"][index].copy()
            target[RIGHT_LEG] += delta
            position_torque = kp * (target - data.qpos[7:36])
            pd = position_torque - kd * data.qvel[6:35]
            extra = teacher_torque(model, data, teacher[index], base_config, position_torque)
            commanded = np.clip(pd + extra, -guarded, guarded)
            if not np.any(weights):
                max_torque_error = max(
                    max_torque_error,
                    float(np.max(np.abs(commanded - tape["motor_executed_torque_nm"][index]))),
                )
                parent_pd = (
                    kp
                    * (tape["motor_pd_target_rad"][index] - tape["motor_joint_position_rad"][index])
                    - kd * tape["motor_joint_velocity_radps"][index]
                )
                parent_extra = tape["motor_raw_torque_nm"][index] - parent_pd
                difference = float(np.max(np.abs(extra - parent_extra)))
                if difference > max_teacher_error:
                    max_teacher_error = difference
                    peak_teacher_error_detail = {
                        "frame": frame,
                        "substep": substep,
                        "joint": int(np.argmax(np.abs(extra - parent_extra))),
                        "teacher_inputs": teacher[index].tolist(),
                        "expected_extra_nm": parent_extra.tolist(),
                        "computed_extra_nm": extra.tolist(),
                    }
            data.ctrl[:] = commanded
            mujoco.mj_step(model, data)
            minimum_height = min(minimum_height, float(data.qpos[2]))
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                a, b = int(contact.geom1), int(contact.geom2)
                if a != ball_geom and b != ball_geom:
                    continue
                other = b if a == ball_geom else a
                if not robot_mask[other]:
                    continue
                mujoco.mj_contactForce(model, data, contact_id, wrench)
                if wrench[0] <= 0:
                    continue
                if foot_mask[other]:
                    if first_foot is None:
                        first_foot = frame
                else:
                    nonfoot.add(frame)
    ball = np.asarray(data.qpos[36:39], dtype=np.float64)
    foot_distance = min(
        float(np.linalg.norm(data.geom_xpos[int(g)] - ball)) for g in np.flatnonzero(foot_mask)
    )
    return {
        "weights": weights.tolist(),
        "amplitude_rad": amplitude_rad,
        "safe": bool(
            minimum_height >= 0.65 and np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        ),
        "first_foot_frame": first_foot,
        "nonfoot_frames": sorted(nonfoot),
        "clean_first_foot": bool(first_foot is not None and not nonfoot),
        "minimum_pelvis_height_m": minimum_height,
        "terminal_ball_speed_mps": float(np.linalg.norm(data.qvel[35:38])),
        "terminal_nearest_foot_geom_distance_m": foot_distance,
        "root_position_rms_error_m": float(np.sqrt(np.mean(np.square(root_error)))),
        "ball_position_rms_error_m": float(np.sqrt(np.mean(np.square(ball_error)))),
        "maximum_parent_torque_difference_nm": max_torque_error,
        "maximum_parent_teacher_difference_nm": max_teacher_error,
        "peak_teacher_error_detail": peak_teacher_error_detail,
    }


def rank(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["safe"]),
        float(row["clean_first_foot"]),
        -float(len(row["nonfoot_frames"])),
        -float(row["terminal_nearest_foot_geom_distance_m"]),
        -float(row["terminal_ball_speed_mps"]),
    )


def train(
    asset_root: Path, capture_dir: Path, output: Path, *, population: int, seed: int
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or output.resolve().is_relative_to(root)
        or not 4 <= population <= 128
        or not 0 <= seed < 2**31
    ):
        raise ValueError("new bounded SIM_ONLY dynamic-teacher proxy evidence required")
    capture = json.loads((capture_dir / "report.json").read_text())
    if capture["status"] != "CAPTURE_QUALIFIED" or capture["report_hash"] != hash_json(
        {k: v for k, v in capture.items() if k != "report_hash"}
    ):
        raise ValueError("qualified same-substep teacher capture required")
    right = next(row for row in capture["rows"] if row["course"]["lateral_m"] < 0)
    tape = load_tape(capture_dir / f"course-{right['course']['seed']}.npz", right["trace_hash"])
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_dynamic_teacher_proxy_v141.py",
            "scripts/rsi_r1_contact_proxy_capture_v138.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
            "src/rosclaw_soccer/growth/locomotion_contact_teacher.py",
        )
    }
    output.mkdir(parents=True)
    baseline = replay(model, tape, np.zeros(6), amplitude_rad=0.05)
    if (
        baseline["first_foot_frame"] != 25
        or baseline["nonfoot_frames"] != [26]
        or baseline["root_position_rms_error_m"] > 1e-4
        or baseline["ball_position_rms_error_m"] > 1e-4
        or baseline["maximum_parent_torque_difference_nm"] > 0.1
        or baseline["maximum_parent_teacher_difference_nm"] > 0.1
    ):
        (output / "failed-baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
        raise ValueError("recomputed teacher cannot reproduce the eight-player parent")
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(population):
        weights = np.zeros(6) if index == 0 else np.clip(rng.normal(0, 0.45, 6), -1, 1)
        amplitude = 0.05 if index == 0 or index < population // 2 else 0.1
        row = replay(model, tape, weights, amplitude_rad=amplitude)
        row["candidate"] = index
        rows.append(row)
        (output / "progress.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "candidate",
                        "amplitude_rad",
                        "safe",
                        "clean_first_foot",
                        "first_foot_frame",
                        "nonfoot_frames",
                        "terminal_ball_speed_mps",
                    )
                }
            ),
            flush=True,
        )
    selected = max(rows, key=rank)
    report = {
        "schema": SCHEMA,
        "capture_report_hash": capture["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_RIGHT_EIGHT_G1_FIXED_DISCRETE_TEACHER_DEVELOPMENT",
        "seed": seed,
        "candidate_count": len(rows),
        "baseline": baseline,
        "baseline_qualification": "APPROXIMATE_FIXED_DISCRETE_TEACHER_ONLY",
        "maximum_allowed_baseline_teacher_difference_nm": 0.1,
        "maximum_allowed_baseline_position_rms_m": 1e-4,
        "selected": selected,
        "all_candidates": rows,
        "status": "DEVELOPMENT_PROXY_CONTACT_GAIN_UNVALIDATED"
        if selected["clean_first_foot"] and not baseline["clean_first_foot"]
        else "REJECTED_NO_PROXY_CONTACT_GAIN",
        "full_world_audition_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during dynamic teacher search")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    report = train(
        args.asset_root, args.capture_dir, args.output, population=args.population, seed=args.seed
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
