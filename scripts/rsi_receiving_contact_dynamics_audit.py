"""SIM_ONLY 500 Hz contact-point dynamics audit of clean-touch parent and GPU candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray
from rsi_mjx_clean_touch_control_es import (
    ACTION_LIMIT_RAD,
    EXAM_FRAME,
    JOINTS,
    SNAPSHOT,
    WEIGHTS,
)
from rsi_mjx_contact_replay_smoke import _contact_masks

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model


def _body_point_velocity(
    model: mujoco.MjModel, data: mujoco.MjData, body: int, point: NDArray[np.float64]
) -> NDArray[np.float64]:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp, jacr, point, body)
    return np.asarray(jacp @ data.qvel, dtype=np.float64)


def _run(
    model: mujoco.MjModel,
    arrays: dict[str, NDArray[np.float64]],
    weights: NDArray[np.float64],
    joint_indices: tuple[int, ...] = JOINTS,
    ball_course: tuple[float, float, float] | None = None,
    post_touch_bias: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    data.qpos[:] = arrays["sonic_recorded_qpos"][SNAPSHOT]
    data.qvel[:] = arrays["sonic_recorded_qvel"][SNAPSHOT]
    if ball_course is not None:
        x, y, vx = ball_course
        if (
            not all(np.isfinite(value) for value in ball_course)
            or not 2.4 <= x <= 2.7
            or not 1.2 <= y <= 1.5
            or not -0.95 <= vx <= -0.75
        ):
            raise ValueError("bounded declared CPU training ball course required")
        base_vx = float(data.qvel[35])
        data.qpos[36] = x
        data.qpos[37] = y
        data.qvel[35] = vx
        data.qvel[39] += (vx - base_vx) / 0.115
    mujoco.mj_forward(model, data)
    ball_geom, robot_mask, foot_mask = _contact_masks(model)
    index = np.asarray(joint_indices)
    limits = model.jnt_range[model.actuator_trnid[list(joint_indices), 0]]
    torque_limit = np.asarray(G1_HARD_TORQUE_LIMITS)
    features = 5 + 2 * len(joint_indices)
    matrix = weights[: len(joint_indices) * features].reshape(len(joint_indices), features)
    bias = weights[len(joint_indices) * features :]
    if post_touch_bias is not None and (
        post_touch_bias.shape != (len(joint_indices),) or not np.isfinite(post_touch_bias).all()
    ):
        raise ValueError("finite post-touch bias must match authorized joints")
    first: dict[str, Any] | None = None
    first_nonfoot: dict[str, Any] | None = None
    foot_impulse = 0.0
    nonfoot_impulse = 0.0
    exam_ball_velocity = None
    exam_ball_distance = None
    minimum = float(data.qpos[2])
    maximum_tilt = 0.0
    for frame in range(SNAPSHOT + 1, 101):
        target = arrays["sonic_recorded_target"][frame].copy()
        kp = arrays["sonic_recorded_kp"][frame]
        kd = arrays["sonic_recorded_kd"][frame]
        dx = data.qpos[36] - data.qpos[0]
        feature = np.concatenate(
            (
                np.asarray(
                    (
                        dx,
                        data.qpos[37] - data.qpos[1],
                        data.qvel[35] - data.qvel[0],
                        data.qvel[36] - data.qvel[1],
                        data.qpos[2] - 0.75,
                    )
                ),
                data.qpos[7 + index],
                data.qvel[6 + index] / 5.0,
            )
        )
        if 0.12 < dx < 0.85:
            phase_bias = (
                post_touch_bias
                if post_touch_bias is not None
                and first is not None
                and first["kind"] == "foot"
                and first_nonfoot is None
                else bias
            )
            residual = ACTION_LIMIT_RAD * np.tanh(matrix @ feature + phase_bias)
            target[index] = np.clip(target[index] + residual, limits[:, 0], limits[:, 1])
        for substep in range(10):
            data.ctrl[:] = np.clip(
                kp * (target - data.qpos[7:36]) - kd * data.qvel[6:35],
                -torque_limit,
                torque_limit,
            )
            mujoco.mj_step(model, data)
            minimum = min(minimum, float(data.qpos[2]))
            quat = data.qpos[3:7]
            maximum_tilt = max(
                maximum_tilt,
                float(np.arccos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1))),
            )
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                a, b = int(contact.geom1), int(contact.geom2)
                if a != ball_geom and b != ball_geom:
                    continue
                other = b if a == ball_geom else a
                if not robot_mask[other]:
                    continue
                kind = "foot" if foot_mask[other] else "nonfoot"
                point = np.asarray(contact.pos, dtype=np.float64)
                ball_velocity = _body_point_velocity(
                    model, data, int(model.geom_bodyid[ball_geom]), point
                )
                body_velocity = _body_point_velocity(
                    model, data, int(model.geom_bodyid[other]), point
                )
                relative = ball_velocity - body_velocity
                normal = np.asarray(contact.frame[:3], dtype=np.float64)
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, contact_id, force)
                impulse = max(0.0, float(force[0])) * model.opt.timestep
                if kind == "foot":
                    foot_impulse += impulse
                else:
                    nonfoot_impulse += impulse
                row = {
                    "frame": frame,
                    "substep": substep,
                    "kind": kind,
                    "geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
                    "contact_point_m": point.tolist(),
                    "ball_point_velocity_mps": ball_velocity.tolist(),
                    "body_point_velocity_mps": body_velocity.tolist(),
                    "relative_speed_mps": float(np.linalg.norm(relative)),
                    "absolute_normal_relative_speed_mps": abs(float(relative @ normal)),
                    "normal_force_n": float(force[0]),
                }
                if first is None:
                    first = row
                if kind == "nonfoot" and first_nonfoot is None:
                    first_nonfoot = row
        if frame == EXAM_FRAME:
            exam_ball_velocity = data.qvel[35:37].copy().tolist()
            exam_ball_distance = float(np.linalg.norm(data.qpos[36:38] - data.qpos[:2]))
    if exam_ball_velocity is None or exam_ball_distance is None:
        raise RuntimeError("missing declared half-second physics frame")
    return {
        "first_contact": first,
        "first_nonfoot_contact": first_nonfoot,
        "foot_normal_impulse_ns": foot_impulse,
        "nonfoot_normal_impulse_ns": nonfoot_impulse,
        "exam_ball_velocity_xy_mps": exam_ball_velocity,
        "exam_ball_speed_mps": float(np.linalg.norm(exam_ball_velocity)),
        "exam_ball_pelvis_distance_m": exam_ball_distance,
        "minimum_pelvis_height_m": minimum,
        "maximum_tilt_rad": maximum_tilt,
    }


def audit(*, asset_root: Path, captured: Path, candidate: Path, output_dir: Path) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY audit directory required")
    trace_path = captured / "motor-trace.npz"
    capture: dict[str, Any] = json.loads((captured / "report.json").read_text(encoding="utf-8"))
    capture_hash = capture.pop("report_hash")
    candidate_data: dict[str, Any] = json.loads(candidate.read_text(encoding="utf-8"))
    candidate_hash = candidate_data.pop("report_hash")
    if (
        capture_hash != hash_json(capture)
        or capture["trace_hash"] != hash_bytes(trace_path.read_bytes())
        or candidate_hash != hash_json(candidate_data)
        or candidate_data["promotion_authorized"] is not False
    ):
        raise ValueError("sealed unpromoted measured evidence required")
    weights32 = np.asarray(candidate_data["weights"], dtype=np.float32)
    if (
        weights32.shape != (WEIGHTS,)
        or not np.isfinite(weights32).all()
        or candidate_data["weights_hash"] != hash_bytes(weights32.tobytes())
    ):
        raise ValueError("finite bounded candidate weights required")
    weights = weights32.astype(np.float64)
    with np.load(trace_path, allow_pickle=False) as trace:
        arrays = {
            key: np.asarray(trace[key], dtype=np.float64)
            for key in (
                "sonic_recorded_qpos",
                "sonic_recorded_qvel",
                "sonic_recorded_target",
                "sonic_recorded_kp",
                "sonic_recorded_kd",
            )
        }
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    parent = _run(model, arrays, np.zeros(WEIGHTS, dtype=np.float64))
    child = _run(model, arrays, weights)
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("contact audit source changed during CPU physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_contact_dynamics_audit.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "capture_hash": capture_hash,
        "candidate_hash": candidate_hash,
        "compiled_model_hash": compiled_model_hash(model),
        "parent": parent,
        "candidate": child,
        "fresh8_opened": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--captured", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    report = audit(**vars(parser.parse_args()))
    print(
        json.dumps(
            {key: report[key] for key in ("report_hash", "parent", "candidate")}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
