"""SIM_ONLY disjoint CPU MuJoCo exam of a frozen feedback leg-residual actor.

Only a candidate selected from GPU training data may be supplied. This script
does not train, tune, or choose candidates using holdout outcomes.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray
from rsi_mjx_contact_replay_smoke import _contact_masks
from rsi_mjx_contact_residual_es import ACTION_LIMIT_RAD, FEATURES, JOINT_INDICES, SCHEMA

from rosclaw_soccer.physics.native_ball_dimensions import inspect_native_ball_dimensions
from rosclaw_soccer.providers.g1.sonic_runup import _sonic_control_parameters
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

FRESH_COURSES = tuple(itertools.product((2.10, 2.18, 2.26, 2.34), (0.08, 0.12), (-0.4, 0.4)))
OLD_STATIC = (2.20, 0.10, 0.0)


def _episode(
    *,
    model: mujoco.MjModel,
    initial_qpos: NDArray[np.float64],
    initial_qvel: NDArray[np.float64],
    targets: NDArray[np.float64],
    course: tuple[float, float, float],
    parameters: NDArray[np.float64],
    masks: tuple[int, NDArray[np.bool_], NDArray[np.bool_]],
    kp: NDArray[np.float64],
    kd: NDArray[np.float64],
) -> dict[str, Any]:
    ball_id, robot_mask, foot_mask = masks
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    x, y, vx = course
    data.qpos[36:39] = (x, y, 0.11)
    data.qvel[35] = vx
    data.qvel[39] = vx / 0.11
    mujoco.mj_forward(model, data)
    initial_state_hash = hash_bytes(data.qpos.tobytes() + data.qvel.tobytes())
    index = np.asarray(JOINT_INDICES, dtype=np.int64)
    limits = model.jnt_range[model.actuator_trnid[index, 0]]
    torque_limit = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    matrix = parameters[: 4 * FEATURES].reshape(4, FEATURES)
    bias = parameters[4 * FEATURES :]
    foot_seen = False
    nonfoot_seen = False
    min_height = float(data.qpos[2])
    projection_count = 0
    qpos_rows = []
    qvel_rows = []
    contact_rows = []
    residual_rows = []
    for base in targets:
        ball_dx = float(data.qpos[36] - data.qpos[0])
        features = np.concatenate(
            (
                np.asarray(
                    (
                        ball_dx,
                        data.qpos[37] - data.qpos[1],
                        data.qvel[35] - data.qvel[0],
                        data.qvel[0],
                        data.qpos[2] - 0.75,
                    ),
                    dtype=np.float64,
                ),
                data.qpos[7 + index],
                data.qvel[6 + index] / 5.0,
                np.asarray((foot_seen, nonfoot_seen), dtype=np.float64),
            )
        )
        if features.shape != (FEATURES,) or not np.isfinite(features).all():
            raise ValueError("nonfinite deployable motor observation")
        gate = float((0.05 < ball_dx < 1.2) or foot_seen)
        residual = ACTION_LIMIT_RAD * gate * np.tanh(matrix @ features + bias)
        proposal = base.copy()
        modified = np.clip(base[index] + residual, limits[:, 0], limits[:, 1])
        projection_count += int(np.count_nonzero(np.abs(modified - base[index] - residual) > 1e-8))
        proposal[index] = modified
        frame_contacts = []
        for _ in range(10):
            data.ctrl[:] = np.clip(
                kp * (proposal - data.qpos[7:36]) - kd * data.qvel[6:35],
                -torque_limit,
                torque_limit,
            )
            mujoco.mj_step(model, data)
            min_height = min(min_height, float(data.qpos[2]))
            substep = np.zeros(2, dtype=bool)
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                a, b = int(contact.geom1), int(contact.geom2)
                if a != ball_id and b != ball_id:
                    continue
                other = b if a == ball_id else a
                if robot_mask[other]:
                    substep[0 if foot_mask[other] else 1] = True
            foot_seen |= bool(substep[0])
            nonfoot_seen |= bool(substep[1])
            frame_contacts.append(substep)
        qpos_rows.append(data.qpos.copy())
        qvel_rows.append(data.qvel.copy())
        contact_rows.append(frame_contacts)
        residual_rows.append(residual.copy())
    arrays = {
        "qpos": np.asarray(qpos_rows),
        "qvel": np.asarray(qvel_rows),
        "contact": np.asarray(contact_rows),
        "residual": np.asarray(residual_rows),
    }
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite CPU contact exam trajectory")
    trajectory_hash = hash_bytes(b"".join(value.tobytes() for value in arrays.values()))
    return {
        "course": list(course),
        "initial_state_hash": initial_state_hash,
        "foot_seen": foot_seen,
        "nonfoot_seen": nonfoot_seen,
        "clean_foot_only": foot_seen and not nonfoot_seen,
        "minimum_pelvis_height_m": min_height,
        "projection_count": projection_count,
        "final_ball_vx_m_s": float(data.qvel[35]),
        "final_ball_position_m": data.qpos[36:39].tolist(),
        "trajectory_hash": trajectory_hash,
    }


def examine(
    *, asset_root: Path, parent_trace: Path, candidate: Path, output_dir: Path
) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    trainer_source = source.with_name("rsi_mjx_contact_residual_es.py")
    helper_source = source.with_name("rsi_mjx_contact_replay_smoke.py")
    if (
        output_dir.exists()
        or output_dir.resolve().is_relative_to(source.resolve().parents[1])
        or not all(
            path.is_file() for path in (parent_trace, candidate, trainer_source, helper_source)
        )
    ):
        raise ValueError("frozen candidate and new external CPU evidence directory required")
    payload: dict[str, Any] = json.loads(candidate.read_text(encoding="utf-8"))
    expected_hash = payload.pop("result_hash")
    if expected_hash != hash_json(payload) or payload.get("schema") != SCHEMA:
        raise ValueError("sealed SIM_ONLY training candidate required")
    params = np.asarray(payload["parameters"], dtype=np.float64)
    if (
        params.shape != (64,)
        or not np.isfinite(params).all()
        or payload.get("promotion_authorized") is not False
        or payload.get("cpu_holdout_evaluated") is not False
    ):
        raise ValueError("bounded unpromoted residual actor required")
    protocol_path = candidate.parent / "protocol.json"
    protocol: dict[str, Any] = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = protocol.pop("protocol_hash")
    if (
        protocol_hash != hash_json(protocol)
        or protocol_hash != payload["protocol_hash"]
        or protocol["source_hash"] != hash_bytes(trainer_source.read_bytes())
        or protocol["helper_hash"] != hash_bytes(helper_source.read_bytes())
        or protocol["parent_trace_hash"] != hash_bytes(parent_trace.read_bytes())
        or protocol["scene_hash"]
        != hash_bytes((asset_root / "g1_description" / "scene_with_ball.xml").read_bytes())
        or bool(
            {tuple(course) for course in protocol["train_courses"]}
            & set((*FRESH_COURSES, OLD_STATIC))
        )
        or protocol["action_joint_indices"] != list(JOINT_INDICES)
        or protocol["action_limit_rad"] != ACTION_LIMIT_RAD
    ):
        raise ValueError("training identity changed or holdout contaminated")
    with np.load(parent_trace, allow_pickle=False) as trace:
        positions = np.asarray(trace["qpos"], dtype=np.float64)
        velocities = np.asarray(trace["qvel"], dtype=np.float64)
        targets = np.asarray(trace["target"], dtype=np.float64)
    snapshot = int(protocol["snapshot_frame"])
    frames = int(protocol["control_frames"])
    if (
        positions.shape != (300, 43)
        or velocities.shape != (300, 41)
        or targets.shape != (300, 29)
        or not 20 <= frames <= 50
        or not 0 <= snapshot <= 200
        or not np.isfinite(positions).all()
        or not np.isfinite(velocities).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("qualified frozen SONIC motor memory required")
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.11, ball_mass_kg=0.43)
    )
    model.opt.timestep = 0.002
    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    ball_dimensions = inspect_native_ball_dimensions(model, geom_id=ball_geom)
    if not ball_dimensions.size_and_mass_in_ifab_range:
        raise ValueError("adult-regulation ball dimensions required")
    masks = _contact_masks(model)
    kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
    output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for course in (*FRESH_COURSES, OLD_STATIC):
        parent_rows = [
            _episode(
                model=model,
                initial_qpos=positions[snapshot],
                initial_qvel=velocities[snapshot],
                targets=targets[snapshot + 1 : snapshot + frames + 1],
                course=course,
                parameters=np.zeros(64, dtype=np.float64),
                masks=masks,
                kp=kp,
                kd=kd,
            )
            for _ in range(2)
        ]
        candidate_rows = [
            _episode(
                model=model,
                initial_qpos=positions[snapshot],
                initial_qvel=velocities[snapshot],
                targets=targets[snapshot + 1 : snapshot + frames + 1],
                course=course,
                parameters=params,
                masks=masks,
                kp=kp,
                kd=kd,
            )
            for _ in range(2)
        ]
        replay_equal = all(
            pair[0]["trajectory_hash"] == pair[1]["trajectory_hash"]
            for pair in (parent_rows, candidate_rows)
        )
        rows.append(
            {
                "course": list(course),
                "partition": "OLD_RETENTION" if course == OLD_STATIC else "FRESH_HOLDOUT",
                "parent": parent_rows[0],
                "candidate": candidate_rows[0],
                "strict_replay_equal": replay_equal,
            }
        )
    fresh = rows[:-1]
    old = rows[-1]
    parent_clean = sum(bool(row["parent"]["clean_foot_only"]) for row in fresh)
    candidate_clean = sum(bool(row["candidate"]["clean_foot_only"]) for row in fresh)
    parent_nonfoot = sum(bool(row["parent"]["nonfoot_seen"]) for row in fresh)
    candidate_nonfoot = sum(bool(row["candidate"]["nonfoot_seen"]) for row in fresh)
    local_passed = bool(
        candidate_clean >= parent_clean + 3
        and candidate_nonfoot <= parent_nonfoot - 2
        and all(row["candidate"]["minimum_pelvis_height_m"] >= 0.65 for row in rows)
        and all(row["candidate"]["projection_count"] == 0 for row in rows)
        and (not old["parent"]["clean_foot_only"] or old["candidate"]["clean_foot_only"])
        and (not old["candidate"]["nonfoot_seen"] or old["parent"]["nonfoot_seen"])
        and all(row["strict_replay_equal"] for row in rows)
    )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("CPU exam source changed during physics")
    result: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.mjx_contact_residual_cpu_exam.v1",
        "activation_ceiling": "SIM_ONLY",
        "candidate_hash": expected_hash,
        "training_protocol_hash": protocol_hash,
        "source_hash": source_hash,
        "canonical_scene_sensor_count": model.nsensor,
        "fresh_parent_clean_foot_count": parent_clean,
        "fresh_candidate_clean_foot_count": candidate_clean,
        "fresh_parent_nonfoot_count": parent_nonfoot,
        "fresh_candidate_nonfoot_count": candidate_nonfoot,
        "local_gate_passed": local_passed,
        "promotion_authorized": False,
        "rows": rows,
    }
    result["report_hash"] = hash_json(result)
    (output_dir / "report.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--parent-trace", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = examine(**vars(args))
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
