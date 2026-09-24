"""Matched MuJoCo SONIC athlete command assay; discovery-only SIM evidence.

Five predeclared command courses share one compiled G1/field model, frozen
weights, gains, initial state and torque guard. No ball skill is evaluated.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation, SonicNavigationConfig
from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.rsi.contracts import AthleteObservation, AthleticIntent, PolicyArtifact
from rosclaw_soccer.rsi.sonic_athlete import SonicAthleteAdapter
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
from rosclaw_soccer.world.field import build_g1_stadium_model


@dataclass(frozen=True)
class Course:
    name: str
    velocity_before_mps: float
    velocity_after_mps: float
    yaw_rate_rad_s: float
    switch_frame: int
    minimum_displacement_m: float = 0.0
    maximum_displacement_m: float = 10.0
    minimum_yaw_rad: float = 0.0
    maximum_final_speed_mps: float = 10.0


COURSES = (
    Course("stand", 0.0, 0.0, 0.0, 150, maximum_displacement_m=0.20, maximum_final_speed_mps=0.25),
    Course("walk", 0.4, 0.4, 0.0, 150, minimum_displacement_m=0.35),
    Course("run", 1.2, 1.2, 0.0, 150, minimum_displacement_m=1.0),
    Course("turn", 0.0, 0.0, 0.8, 150, maximum_displacement_m=0.50, minimum_yaw_rad=0.3),
    Course("stop", 0.6, 0.0, 0.0, 60, minimum_displacement_m=0.2, maximum_final_speed_mps=0.3),
)


def _source_binding() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "scripts/rsi_sonic_combine.py",
        "src/rosclaw_soccer/rsi/contracts.py",
        "src/rosclaw_soccer/rsi/sonic_athlete.py",
        "src/rosclaw_soccer/providers/g1/sonic_navigation.py",
        "src/rosclaw_soccer/providers/g1/sonic_runup.py",
    )
    return str(hash_json({name: hash_bytes((root / name).read_bytes()) for name in paths}))


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    receipt = dict(payload)
    receipt["manifest_hash"] = hash_json(payload)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _angles(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame: int,
    body_hash: str,
    joint_map_hash: str,
    physics_hash: str,
) -> AthleteObservation:
    # Contact flags are measured from the current contact manifold. The
    # adapter does not consume them, but each episode records their presence.
    foot_contact = [False, False]
    for index in range(data.ncon):
        contact = data.contact[index]
        names = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or "",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or "",
        )
        for side, token in enumerate(("left_foot", "right_foot")):
            if any(token in name for name in names):
                foot_contact[side] = True
    return AthleteObservation(
        body_id="blue.playmaker",
        body_hash=body_hash,
        joint_map_hash=joint_map_hash,
        physics_hash=physics_hash,
        frame=frame,
        time_sec=frame * 0.02,
        root_position_m=cast(tuple[float, float, float], tuple(float(v) for v in data.qpos[:3])),
        root_velocity_mps=cast(tuple[float, float, float], tuple(float(v) for v in data.qvel[:3])),
        root_angular_velocity_rad_s=cast(
            tuple[float, float, float], tuple(float(v) for v in data.qvel[3:6])
        ),
        root_quaternion_wxyz=cast(
            tuple[float, float, float, float], tuple(float(v) for v in data.qpos[3:7])
        ),
        joint_position=tuple(float(v) for v in data.qpos[7:36]),
        joint_velocity=tuple(float(v) for v in data.qvel[6:35]),
        contact_flags=tuple(foot_contact),
    )


def _course(
    model: mujoco.MjModel,
    model_root: Path,
    body_hash: str,
    joint_map_hash: str,
    code_hash: str,
    course: Course,
    output_dir: Path,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    data.qpos[:7] = (0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0)
    data.qpos[7:36] = G1SonicRunupController.default_angles
    if model.nq > 36:
        data.qpos[36:39] = (20.0, 20.0, 0.2)
    mujoco.mj_forward(model, data)
    initial_state_hash = hash_json({"qpos": data.qpos.tolist(), "qvel": data.qvel.tolist()})
    navigation = G1SonicNavigation(
        model_root,
        "blue.playmaker",
        SonicNavigationConfig(
            maximum_frames=150,
            model_variant="low_latency",
            experimental_maximum_speed_mps=1.2,
            planner_seed=920101,
        ),
    )
    qualification = navigation.backend.qualification
    artifact = PolicyArtifact(
        artifact_id="blue.playmaker.sonic.combine",
        backend_id="sonic_navigation",
        backend_contract_hash=navigation.contract_hash,
        code_hash=code_hash,
        weights_hash=hash_json(
            {
                "planner": qualification.planner_hash,
                "encoder": qualification.encoder_hash,
                "decoder": qualification.decoder_hash,
            }
        ),
        body_hash=body_hash,
        observation_hash=hash_json("rosclaw_soccer.rsi.athlete_observation.v1"),
        action_hash=hash_json("rosclaw_soccer.rsi.sonic_joint_target.v1"),
        joint_map_hash=joint_map_hash,
        physics_hash=body_hash,
        gain_hash=hash_json(
            {
                "kp": [float(value) for value in navigation.backend.kp],
                "kd": [float(value) for value in navigation.backend.kd],
            }
        ),
        action_kind="JOINT_TARGET",
        action_size=29,
        control_dt_s=0.02,
    )
    athlete = SonicAthleteAdapter(navigation, artifact)
    rows: dict[str, list[np.ndarray]] = {name: [] for name in ("qpos", "qvel", "target", "torque")}
    torque_limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    for frame in range(150):
        obs = _observation(model, data, frame, body_hash, joint_map_hash, body_hash)
        velocity = (
            course.velocity_before_mps if frame < course.switch_frame else course.velocity_after_mps
        )
        intent = AthleticIntent(
            velocity_xy_mps=(velocity, 0.0),
            heading_rad=0.0,
            yaw_rate_rad_s=course.yaw_rate_rad_s,
            body_height_m=0.793,
            contact_intent="locomotion",
            sport_context="soccer",
        )
        action = athlete.step(obs, intent)
        target = np.asarray(action.joint_target, dtype=np.float64)
        for _ in range(10):
            torque = np.clip(
                (target - data.qpos[7:36]) * navigation.backend.kp
                - data.qvel[6:35] * navigation.backend.kd,
                -torque_limits,
                torque_limits,
            )
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
        rows["qpos"].append(data.qpos[:36].copy())
        rows["qvel"].append(data.qvel[:35].copy())
        rows["target"].append(target)
        rows["torque"].append(data.ctrl.copy())
    arrays = {name: np.asarray(value) for name, value in rows.items()}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite physical trajectory")
    trace_path = output_dir / f"{course.name}.npz"
    np.savez_compressed(
        trace_path,
        qpos=arrays["qpos"],
        qvel=arrays["qvel"],
        target=arrays["target"],
        torque=arrays["torque"],
    )
    q = arrays["qpos"]
    v = arrays["qvel"]
    angles = np.asarray([_angles(item) for item in q[:, 3:7]])
    displacement = float(np.linalg.norm(q[-1, :2] - q[0, :2]))
    yaw_change = float(abs(np.unwrap(angles[:, 2])[-1] - np.unwrap(angles[:, 2])[0]))
    final_speed = float(np.linalg.norm(v[-10:, :2], axis=1).mean())
    min_height = float(q[:, 2].min())
    max_tilt = float(np.max(np.abs(angles[:, :2])))
    passed = (
        min_height >= 0.55
        and max_tilt <= 0.40
        and displacement >= course.minimum_displacement_m
        and displacement <= course.maximum_displacement_m
        and yaw_change >= course.minimum_yaw_rad
        and final_speed <= course.maximum_final_speed_mps
    )
    result = {
        "schema": "rosclaw_soccer.rsi.sonic_combine_course.v1",
        "course": course.name,
        "partition": "DISCOVERY",
        "activation_ceiling": "SIM_ONLY",
        "teacher_active": False,
        "task_success": passed,
        "thresholds": asdict(course),
        "minimum_pelvis_height_m": min_height,
        "maximum_absolute_roll_pitch_rad": max_tilt,
        "displacement_xy_m": displacement,
        "absolute_yaw_change_rad": yaw_change,
        "final_speed_mps": final_speed,
        "initial_state_hash": initial_state_hash,
        "physics_hash": body_hash,
        "artifact_hash": artifact.contract_hash,
        "trajectory_hash": hash_bytes(trace_path.read_bytes()),
    }
    _write_receipt(output_dir / f"{course.name}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--stadium-assets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=False, exist_ok=False)
    source_hash = _source_binding()
    model = build_g1_stadium_model(args.stadium_assets)
    model.opt.timestep = 0.002
    joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in G1_DDS_JOINT_NAMES]
    )
    if (
        model.nu != 29
        or np.any(joint_ids < 0)
        or not np.array_equal(model.jnt_qposadr[joint_ids], np.arange(7, 36))
        or not np.array_equal(model.jnt_dofadr[joint_ids], np.arange(6, 35))
        or not np.array_equal(model.actuator_trnid[:, 0], joint_ids)
    ):
        raise ValueError("G1 joint order or actuator mapping differs from SONIC")
    body_hash = compiled_model_hash(model)
    joint_map_hash = hash_json(list(G1_DDS_JOINT_NAMES))
    protocol = {
        "schema": "rosclaw_soccer.rsi.sonic_combine_protocol.v1",
        "partition": "DISCOVERY",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "source_hash": source_hash,
        "body_hash": body_hash,
        "joint_map_hash": joint_map_hash,
        "model_root": str(args.model_root.expanduser().resolve()),
        "stadium_assets": str(args.stadium_assets.expanduser().resolve()),
        "courses": [asdict(course) for course in COURSES],
    }
    _write_receipt(output_dir / "protocol.json", protocol)
    rows = []
    for course in COURSES:
        if _source_binding() != source_hash:
            raise RuntimeError("source changed during SONIC combine assay")
        row = _course(
            model, args.model_root, body_hash, joint_map_hash, source_hash, course, output_dir
        )
        rows.append(row)
        print(
            json.dumps(
                {
                    "course": course.name,
                    "passed": row["task_success"],
                    "min_height_m": row["minimum_pelvis_height_m"],
                    "displacement_m": row["displacement_xy_m"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if _source_binding() != source_hash:
        raise RuntimeError("source changed before SONIC combine completion")
    complete = {
        "schema": "rosclaw_soccer.rsi.sonic_combine_complete.v1",
        "source_hash": source_hash,
        "body_hash": body_hash,
        "partition": "DISCOVERY",
        "pass_count": sum(bool(row["task_success"]) for row in rows),
        "course_count": len(rows),
        "rows": rows,
        "promotion_authorized": False,
    }
    _write_receipt(output_dir / "complete.json", complete)
    print(json.dumps({"pass_count": complete["pass_count"], "course_count": len(rows)}))


if __name__ == "__main__":
    main()
