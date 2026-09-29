"""Start at the exact R1 frame-zero state and learn earlier right-foot posture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rsi_mjx_contact_replay_smoke import _contact_masks
from rsi_r1_dynamic_teacher_proxy_v141 import teacher_torque
from rsi_r1_pd_proxy_search_v140 import load_tape

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.training.receiving_classroom import r1_contact_tap_receiving_configuration
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

SCHEMA = "rosclaw_soccer.rsi.r1_early_dynamic_proxy_v142.result.v1"
STOP = 60
RIGHT_LEG = np.arange(6, 12)


def replay(
    model: mujoco.MjModel, tape: dict[str, np.ndarray], weights: np.ndarray
) -> dict[str, Any]:
    if weights.shape != (6,) or not np.isfinite(weights).all() or np.any(np.abs(weights) > 1):
        raise ValueError("bounded six-joint earlier SIM_ONLY action required")
    if tape["initial_local_qpos"].shape != (43,) or tape["initial_local_qvel"].shape != (41,):
        raise ValueError("exact frame-zero local body and ball required")
    teacher = tape["motor_teacher_inputs"]
    if teacher.shape[1:] != (17,) or len(teacher) < STOP * 10:
        raise ValueError("complete same-substep teacher schedule required")
    _, teacher_base = r1_contact_tap_receiving_configuration()
    data = mujoco.MjData(model)
    data.qpos[:] = tape["initial_local_qpos"]
    data.qvel[:] = tape["initial_local_qvel"]
    mujoco.mj_forward(model, data)
    ball_geom, robot_mask, foot_mask = _contact_masks(model)
    guarded = 0.85 * np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    wrench = np.zeros(6, dtype=np.float64)
    first_foot: int | None = None
    nonfoot: set[int] = set()
    minimum_height = float(data.qpos[2])
    maximum_parent_torque_difference = 0.0
    maximum_parent_teacher_difference = 0.0
    root_errors: list[float] = []
    ball_errors: list[float] = []
    for frame in range(STOP):
        if frame >= 15:
            sample = frame - 15
            root_errors.append(
                float(np.linalg.norm(data.qpos[:3] - tape["observation_qpos"][sample, :3]))
            )
            ball_errors.append(
                float(np.linalg.norm(data.qpos[36:39] - tape["observation_qpos"][sample, 36:39]))
            )
        ramp = min(1.0, frame / 10.0)
        fade = 1.0 if first_foot is None else max(0.0, 1.0 - (frame - first_foot) / 10.0)
        delta = 0.1 * ramp * fade * weights
        for substep in range(10):
            index = frame * 10 + substep
            kp = tape["motor_kp"][index]
            kd = tape["motor_kd"][index]
            target = tape["motor_pd_target_rad"][index].copy()
            target[RIGHT_LEG] += delta
            position_torque = kp * (target - data.qpos[7:36])
            pd = position_torque - kd * data.qvel[6:35]
            extra = teacher_torque(model, data, teacher[index], teacher_base, position_torque)
            torque = np.clip(pd + extra, -guarded, guarded)
            if not np.any(weights):
                maximum_parent_torque_difference = max(
                    maximum_parent_torque_difference,
                    float(np.max(np.abs(torque - tape["motor_executed_torque_nm"][index]))),
                )
                parent_pd = (
                    kp
                    * (tape["motor_pd_target_rad"][index] - tape["motor_joint_position_rad"][index])
                    - kd * tape["motor_joint_velocity_radps"][index]
                )
                parent_extra = tape["motor_raw_torque_nm"][index] - parent_pd
                maximum_parent_teacher_difference = max(
                    maximum_parent_teacher_difference, float(np.max(np.abs(extra - parent_extra)))
                )
            data.ctrl[:] = torque
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
        "safe": bool(
            minimum_height >= 0.65 and np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        ),
        "first_foot_frame": first_foot,
        "nonfoot_frames": sorted(nonfoot),
        "clean_first_foot": bool(first_foot is not None and not nonfoot),
        "terminal_ball_speed_mps": float(np.linalg.norm(data.qvel[35:38])),
        "terminal_nearest_foot_geom_distance_m": foot_distance,
        "minimum_pelvis_height_m": minimum_height,
        "root_position_rms_error_m": float(np.sqrt(np.mean(np.square(root_errors)))),
        "ball_position_rms_error_m": float(np.sqrt(np.mean(np.square(ball_errors)))),
        "maximum_parent_torque_difference_nm": maximum_parent_torque_difference,
        "maximum_parent_teacher_difference_nm": maximum_parent_teacher_difference,
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
        or not 4 <= population <= 512
        or not 0 <= seed < 2**31
    ):
        raise ValueError("new bounded external SIM_ONLY early proxy experiment required")
    capture = json.loads((capture_dir / "report.json").read_text())
    if capture["status"] != "CAPTURE_QUALIFIED" or capture["report_hash"] != hash_json(
        {k: v for k, v in capture.items() if k != "report_hash"}
    ):
        raise ValueError("qualified frame-zero teacher capture required")
    right = next(row for row in capture["rows"] if row["course"]["lateral_m"] < 0)
    tape = load_tape(capture_dir / f"course-{right['course']['seed']}.npz", right["trace_hash"])
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_early_dynamic_proxy_v142.py",
            "scripts/rsi_r1_dynamic_teacher_proxy_v141.py",
            "scripts/rsi_r1_contact_proxy_capture_v138.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    baseline = replay(model, tape, np.zeros(6))
    if (
        baseline["first_foot_frame"] != 25
        or baseline["nonfoot_frames"] != [26]
        or baseline["root_position_rms_error_m"] > 1e-5
        or baseline["ball_position_rms_error_m"] > 1e-5
        or baseline["maximum_parent_torque_difference_nm"] > 0.1
        or baseline["maximum_parent_teacher_difference_nm"] > 0.1
    ):
        (output / "failed-baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
        raise ValueError("frame-zero dynamic proxy cannot reproduce the eight-player parent")
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(population):
        weights = np.zeros(6) if index == 0 else np.clip(rng.normal(0, 0.5, 6), -1, 1)
        row = replay(model, tape, weights)
        row["candidate"] = index
        rows.append(row)
        (output / "progress.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        if index % 8 == 0 or row["clean_first_foot"]:
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "candidate",
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
        "partition": "CONSUMED_RIGHT_EIGHT_G1_FRAME_ZERO_PROXY_DEVELOPMENT",
        "seed": seed,
        "candidate_count": len(rows),
        "baseline": baseline,
        "selected": selected,
        "all_candidates": rows,
        "status": "DEVELOPMENT_PROXY_CONTACT_GAIN_UNVALIDATED"
        if selected["clean_first_foot"] and not baseline["clean_first_foot"]
        else "REJECTED_NO_PROXY_CONTACT_GAIN",
        "fixed_high_level_teacher_schedule": True,
        "full_world_audition_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during frame-zero proxy development")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    report = train(
        args.asset_root, args.capture_dir, args.output, population=args.population, seed=args.seed
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
