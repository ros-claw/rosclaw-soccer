"""SIM_ONLY temporal whole-body CEM on the frame-zero R1 teacher proxy."""

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

SCHEMA = "rosclaw_soccer.rsi.r1_temporal_body_proxy_v143.result.v1"
STOP = 60
KNOT_FRAMES = (0, 10, 20)
BASIS = np.zeros((8, 29), dtype=np.float64)
BASIS[0, (6, 10)] = (1.0, -0.6)
BASIS[1, (7, 11)] = (1.0, -0.6)
BASIS[2, 8] = 1.0
BASIS[3, 9] = 1.0
BASIS[4, (0, 3)] = (-0.6, 0.8)
BASIS[5, (1, 5)] = (1.0, -0.5)
BASIS[6, 12] = 1.0
BASIS[7, (15, 22)] = (1.0, -1.0)


def desired_action(weights: np.ndarray, frame: int) -> np.ndarray:
    if weights.shape != (3, 8) or not np.isfinite(weights).all() or np.any(np.abs(weights) > 1):
        raise ValueError("three bounded eight-synergy temporal knots required")
    if frame <= KNOT_FRAMES[0]:
        synergy = weights[0]
    elif frame >= KNOT_FRAMES[-1]:
        synergy = weights[-1]
    else:
        index = 0 if frame < KNOT_FRAMES[1] else 1
        fraction = (frame - KNOT_FRAMES[index]) / (KNOT_FRAMES[index + 1] - KNOT_FRAMES[index])
        synergy = (1 - fraction) * weights[index] + fraction * weights[index + 1]
    return np.asarray(np.clip(0.08 * (synergy @ BASIS), -0.08, 0.08), dtype=np.float64)


def replay(
    model: mujoco.MjModel, tape: dict[str, np.ndarray], weights: np.ndarray
) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=np.float64)
    desired_action(weights, 0)
    _, teacher_base = r1_contact_tap_receiving_configuration()
    data = mujoco.MjData(model)
    data.qpos[:] = tape["initial_local_qpos"]
    data.qvel[:] = tape["initial_local_qvel"]
    mujoco.mj_forward(model, data)
    ball_geom, robot_mask, foot_mask = _contact_masks(model)
    shin_geom = model.geom("right_shin").id
    guarded = 0.85 * np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    wrench = np.zeros(6, dtype=np.float64)
    closest = np.zeros(6, dtype=np.float64)
    filtered = np.zeros(29, dtype=np.float64)
    first_foot: int | None = None
    nonfoot: set[int] = set()
    minimum_height = float(data.qpos[2])
    minimum_shin_gap = float("inf")
    max_parent_torque_error = 0.0
    root_error: list[float] = []
    ball_error: list[float] = []
    for frame in range(STOP):
        if frame >= 15:
            sample = frame - 15
            root_error.append(
                float(np.linalg.norm(data.qpos[:3] - tape["observation_qpos"][sample, :3]))
            )
            ball_error.append(
                float(np.linalg.norm(data.qpos[36:39] - tape["observation_qpos"][sample, 36:39]))
            )
        proposal = desired_action(weights, frame)
        if first_foot is not None:
            proposal *= max(0.0, 1.0 - (frame - first_foot) / 10.0)
        filtered += np.clip(0.25 * (proposal - filtered), -0.02, 0.02)
        for substep in range(10):
            index = frame * 10 + substep
            kp = tape["motor_kp"][index]
            kd = tape["motor_kd"][index]
            target = tape["motor_pd_target_rad"][index] + filtered
            position_torque = kp * (target - data.qpos[7:36])
            pd = position_torque - kd * data.qvel[6:35]
            extra = teacher_torque(
                model, data, tape["motor_teacher_inputs"][index], teacher_base, position_torque
            )
            torque = np.clip(pd + extra, -guarded, guarded)
            if not np.any(weights):
                max_parent_torque_error = max(
                    max_parent_torque_error,
                    float(np.max(np.abs(torque - tape["motor_executed_torque_nm"][index]))),
                )
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
            minimum_height = min(minimum_height, float(data.qpos[2]))
            if 22 <= frame <= 35:
                minimum_shin_gap = min(
                    minimum_shin_gap,
                    float(mujoco.mj_geomDistance(model, data, shin_geom, ball_geom, 10.0, closest)),
                )
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
        "minimum_right_shin_gap_m": minimum_shin_gap,
        "terminal_ball_speed_mps": float(np.linalg.norm(data.qvel[35:38])),
        "terminal_nearest_foot_geom_distance_m": foot_distance,
        "minimum_pelvis_height_m": minimum_height,
        "root_position_rms_error_m": float(np.sqrt(np.mean(np.square(root_error)))),
        "ball_position_rms_error_m": float(np.sqrt(np.mean(np.square(ball_error)))),
        "maximum_parent_torque_difference_nm": max_parent_torque_error,
    }


def rank(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["safe"]),
        float(row["first_foot_frame"] is not None),
        float(row["clean_first_foot"]),
        -float(len(row["nonfoot_frames"])),
        float(row["minimum_right_shin_gap_m"]),
        -float(row["terminal_nearest_foot_geom_distance_m"]),
        -float(row["terminal_ball_speed_mps"]),
    )


def train(
    asset_root: Path,
    capture_dir: Path,
    output: Path,
    *,
    generations: int,
    population: int,
    seed: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or output.resolve().is_relative_to(root)
        or not 1 <= generations <= 8
        or not 8 <= population <= 128
        or not 0 <= seed < 2**31
    ):
        raise ValueError("new bounded external SIM_ONLY temporal-body training evidence required")
    capture = json.loads((capture_dir / "report.json").read_text())
    if capture["status"] != "CAPTURE_QUALIFIED" or capture["report_hash"] != hash_json(
        {k: v for k, v in capture.items() if k != "report_hash"}
    ):
        raise ValueError("qualified frame-zero eight-player capture required")
    right = next(row for row in capture["rows"] if row["course"]["lateral_m"] < 0)
    tape = load_tape(capture_dir / f"course-{right['course']['seed']}.npz", right["trace_hash"])
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_temporal_body_proxy_v143.py",
            "scripts/rsi_r1_dynamic_teacher_proxy_v141.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    baseline = replay(model, tape, np.zeros((3, 8)))
    if (
        baseline["first_foot_frame"] != 25
        or baseline["nonfoot_frames"] != [26]
        or baseline["root_position_rms_error_m"] > 1e-6
        or baseline["ball_position_rms_error_m"] > 1e-6
        or baseline["maximum_parent_torque_difference_nm"] > 1e-6
    ):
        (output / "failed-baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
        raise ValueError("temporal proxy baseline failed exact parent replay")
    rng = np.random.default_rng(seed)
    mean = np.zeros((3, 8), dtype=np.float64)
    sigma = np.full((3, 8), 0.45, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for generation in range(generations):
        candidates = [mean.copy()] + [
            np.clip(mean + rng.normal(size=(3, 8)) * sigma, -1, 1) for _ in range(population - 1)
        ]
        generation_rows = []
        for candidate in candidates:
            row = replay(model, tape, candidate)
            row["candidate"] = len(rows)
            row["generation"] = generation
            rows.append(row)
            generation_rows.append(row)
            (output / "progress.json").write_text(
                json.dumps(rows, indent=2, allow_nan=False) + "\n"
            )
            if row["clean_first_foot"]:
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "candidate",
                                "generation",
                                "safe",
                                "first_foot_frame",
                                "nonfoot_frames",
                                "minimum_right_shin_gap_m",
                                "terminal_ball_speed_mps",
                            )
                        }
                    ),
                    flush=True,
                )
        elite = sorted(generation_rows, key=rank, reverse=True)[: max(4, population // 8)]
        mean = np.mean(np.asarray([row["weights"] for row in elite]), axis=0)
        sigma = np.maximum(
            0.08, 0.7 * sigma + 0.3 * np.std(np.asarray([row["weights"] for row in elite]), axis=0)
        )
        leader = max(generation_rows, key=rank)
        print(
            json.dumps(
                {
                    k: leader[k]
                    for k in (
                        "candidate",
                        "generation",
                        "safe",
                        "first_foot_frame",
                        "nonfoot_frames",
                        "minimum_right_shin_gap_m",
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
        "partition": "CONSUMED_RIGHT_EIGHT_G1_TEMPORAL_BODY_PROXY_DEVELOPMENT",
        "seed": seed,
        "generations": generations,
        "population": population,
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
        raise ValueError("source drift during temporal-body proxy training")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.capture_dir,
        args.output,
        generations=args.generations,
        population=args.population,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
