"""SIM_ONLY disjoint CPU MuJoCo exam of a frozen receiver feedback actor."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray
from rsi_mjx_contact_replay_smoke import _contact_masks
from rsi_mjx_receiving_feedback_es import (
    ACTION_LIMIT_RAD,
    CONTROL_FRAMES,
    COURSES,
    FEATURES,
    JOINT_INDICES,
    SCHEMA,
    SNAPSHOT_FRAME,
)

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

FRESH = tuple(itertools.product((2.98, 3.00), (1.36, 1.40), (-1.08, -0.98)))


def _episode(
    *,
    model: mujoco.MjModel,
    arrays: dict[str, NDArray[np.float64]],
    course: tuple[float, float, float],
    parameters: NDArray[np.float64],
    masks: tuple[int, NDArray[np.bool_], NDArray[np.bool_]],
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    data.qpos[:] = arrays["sonic_recorded_qpos"][SNAPSHOT_FRAME]
    data.qvel[:] = arrays["sonic_recorded_qvel"][SNAPSHOT_FRAME]
    original_vx = float(data.qvel[35])
    data.qpos[36:38] = course[:2]
    data.qvel[35] = course[2]
    data.qvel[39] += (course[2] - original_vx) / 0.115
    mujoco.mj_forward(model, data)
    index = np.asarray(JOINT_INDICES, dtype=np.int64)
    limits = model.jnt_range[model.actuator_trnid[index, 0]]
    matrix = parameters[: 4 * FEATURES].reshape(4, FEATURES)
    bias = parameters[4 * FEATURES :]
    torque_limit = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    ball_geom, robot_mask, foot_mask = masks
    foot_seen = False
    nonfoot_seen = False
    first_contact: str | None = None
    minimum = float(data.qpos[2])
    maximum_tilt = 0.0
    projection_count = 0
    rows = []
    for frame in range(SNAPSHOT_FRAME + 1, SNAPSHOT_FRAME + CONTROL_FRAMES + 1):
        ball_dx = float(data.qpos[36] - data.qpos[0])
        feature = np.concatenate(
            (
                np.asarray(
                    (
                        ball_dx,
                        data.qpos[37] - data.qpos[1],
                        data.qvel[35] - data.qvel[0],
                        data.qvel[0],
                        data.qpos[2] - 0.75,
                    )
                ),
                data.qpos[7 + index],
                data.qvel[6 + index] / 5.0,
                np.asarray((foot_seen, nonfoot_seen), dtype=np.float64),
            )
        )
        if feature.shape != (FEATURES,) or not np.isfinite(feature).all():
            raise ValueError("finite receiver motor observation required")
        gate = float((0.05 < ball_dx < 1.2) or foot_seen)
        residual = ACTION_LIMIT_RAD * gate * np.tanh(matrix @ feature + bias)
        target = arrays["sonic_recorded_target"][frame].copy()
        modified = np.clip(target[index] + residual, limits[:, 0], limits[:, 1])
        projection_count += int(
            np.count_nonzero(np.abs(modified - target[index] - residual) > 1e-8)
        )
        target[index] = modified
        for _ in range(10):
            data.ctrl[:] = np.clip(
                arrays["sonic_recorded_kp"][frame] * (target - data.qpos[7:36])
                - arrays["sonic_recorded_kd"][frame] * data.qvel[6:35],
                -torque_limit,
                torque_limit,
            )
            mujoco.mj_step(model, data)
            minimum = min(minimum, float(data.qpos[2]))
            quat = data.qpos[3:7]
            tilt = math.acos(float(np.clip(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2), -1.0, 1.0)))
            maximum_tilt = max(maximum_tilt, tilt)
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                a, b = int(contact.geom1), int(contact.geom2)
                if a != ball_geom and b != ball_geom:
                    continue
                other = b if a == ball_geom else a
                if not robot_mask[other]:
                    continue
                if foot_mask[other]:
                    foot_seen = True
                    first_contact = first_contact or "foot"
                else:
                    nonfoot_seen = True
                    first_contact = first_contact or "nonfoot"
        rows.append(np.concatenate((data.qpos.copy(), data.qvel.copy())))
    states = np.asarray(rows)
    if not np.isfinite(states).all():
        raise ValueError("nonfinite receiver CPU trajectory")
    return {
        "course": list(course),
        "foot_seen": foot_seen,
        "nonfoot_seen": nonfoot_seen,
        "clean_foot_only": foot_seen and not nonfoot_seen,
        "first_contact": first_contact,
        "minimum_pelvis_height_m": minimum,
        "peak_tilt_rad": maximum_tilt,
        "projection_count": projection_count,
        "final_ball_vx_m_s": float(data.qvel[35]),
        "trajectory_hash": hash_bytes(states.tobytes()),
    }


def examine(
    *, asset_root: Path, captured: Path, fidelity: Path, candidate: Path, output_dir: Path
) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    trainer = source.with_name("rsi_mjx_receiving_feedback_es.py")
    helper = source.with_name("rsi_mjx_contact_replay_smoke.py")
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY CPU holdout directory required")
    payload: dict[str, Any] = json.loads(candidate.read_text(encoding="utf-8"))
    candidate_hash = payload.pop("result_hash", None)
    protocol: dict[str, Any] = json.loads(
        (candidate.parent / "protocol.json").read_text(encoding="utf-8")
    )
    protocol_hash = protocol.pop("protocol_hash", None)
    if (
        candidate_hash != hash_json(payload)
        or payload.get("schema") != SCHEMA
        or payload.get("promotion_authorized") is not False
        or payload.get("cpu_holdout_evaluated") is not False
        or protocol_hash != hash_json(protocol)
        or protocol_hash != payload["protocol_hash"]
        or protocol["source_hash"] != hash_bytes(trainer.read_bytes())
        or protocol["helper_hash"] != hash_bytes(helper.read_bytes())
        or protocol["capture_hash"] != hash_bytes((captured / "motor-trace.npz").read_bytes())
        or protocol["fidelity_hash"] != hash_bytes(fidelity.read_bytes())
        or tuple(tuple(course) for course in protocol["train_courses"]) != COURSES
        or bool(set(COURSES) & set(FRESH))
        or protocol["snapshot_frame"] != SNAPSHOT_FRAME
        or protocol["control_frames"] != CONTROL_FRAMES
        or protocol["action_joint_indices"] != list(JOINT_INDICES)
        or protocol["action_limit_rad"] != ACTION_LIMIT_RAD
    ):
        raise ValueError("frozen receiving candidate and disjoint exam required")
    parameters = np.asarray(payload["parameters"], dtype=np.float64)
    if parameters.shape != (64,) or not np.isfinite(parameters).all():
        raise ValueError("bounded 64-parameter receiver actor required")
    with np.load(captured / "motor-trace.npz", allow_pickle=False) as trace:
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
    masks = _contact_masks(model)
    center = (
        float(arrays["sonic_recorded_qpos"][SNAPSHOT_FRAME, 36]),
        float(arrays["sonic_recorded_qpos"][SNAPSHOT_FRAME, 37]),
        float(arrays["sonic_recorded_qvel"][SNAPSHOT_FRAME, 35]),
    )
    rows = []
    for course in (*FRESH, center):
        parent = [
            _episode(
                model=model,
                arrays=arrays,
                course=course,
                parameters=np.zeros(64),
                masks=masks,
            )
            for _ in range(2)
        ]
        child = [
            _episode(model=model, arrays=arrays, course=course, parameters=parameters, masks=masks)
            for _ in range(2)
        ]
        rows.append(
            {
                "course": list(course),
                "partition": "OLD_CENTER" if course == center else "FRESH_HOLDOUT",
                "parent": parent[0],
                "candidate": child[0],
                "strict_replay_equal": (
                    parent[0]["trajectory_hash"] == parent[1]["trajectory_hash"]
                    and child[0]["trajectory_hash"] == child[1]["trajectory_hash"]
                ),
            }
        )
    fresh, old = rows[:-1], rows[-1]
    parent_clean = sum(row["parent"]["clean_foot_only"] for row in fresh)
    child_clean = sum(row["candidate"]["clean_foot_only"] for row in fresh)
    parent_nonfoot = sum(row["parent"]["nonfoot_seen"] for row in fresh)
    child_nonfoot = sum(row["candidate"]["nonfoot_seen"] for row in fresh)
    passed = bool(
        child_clean >= parent_clean + 3
        and child_nonfoot <= parent_nonfoot - 2
        and all(row["candidate"]["minimum_pelvis_height_m"] >= 0.65 for row in rows)
        and all(row["candidate"]["peak_tilt_rad"] < 0.30 for row in rows)
        and all(row["candidate"]["projection_count"] == 0 for row in rows)
        and (not old["candidate"]["nonfoot_seen"] or old["parent"]["nonfoot_seen"])
        and all(row["strict_replay_equal"] for row in rows)
    )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("receiver CPU exam source changed during physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.mjx_receiving_cpu_exam.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "candidate_hash": candidate_hash,
        "training_protocol_hash": protocol_hash,
        "fresh_parent_clean_foot_count": parent_clean,
        "fresh_candidate_clean_foot_count": child_clean,
        "fresh_parent_nonfoot_count": parent_nonfoot,
        "fresh_candidate_nonfoot_count": child_nonfoot,
        "local_gate_passed": passed,
        "promotion_authorized": False,
        "rows": rows,
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
    parser.add_argument("--fidelity", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    report = examine(**vars(parser.parse_args()))
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
