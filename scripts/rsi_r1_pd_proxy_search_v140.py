"""SIM_ONLY causal-PD proxy qualification and bounded right-foot contact search.

The exogenous teacher torque is frozen from the eight-player parent; candidate
scores are only development suggestions until a full eight-player replay agrees.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rsi_mjx_contact_replay_smoke import _contact_masks

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

SCHEMA = "rosclaw_soccer.rsi.r1_pd_proxy_search_v140.result.v1"
START = 15
STOP = 60
RIGHT_LEG = np.arange(6, 12)


def load_tape(path: Path, expected_hash: str) -> dict[str, np.ndarray]:
    if hash_bytes(path.read_bytes()) != expected_hash:
        raise ValueError("sealed eight-player motor tape required")
    with np.load(path, allow_pickle=False) as tape:
        arrays = {key: np.asarray(tape[key]).copy() for key in tape.files}
    if (
        arrays["observation_qpos"].shape[1:] != (43,)
        or arrays["observation_qvel"].shape[1:] != (41,)
        or not np.array_equal(arrays["observation_frame"][: STOP - START], np.arange(START, STOP))
        or arrays["motor_pd_target_rad"].shape[1:] != (29,)
        or len(arrays["motor_pd_target_rad"]) < STOP * 10
        or any(not np.isfinite(value).all() for value in arrays.values())
    ):
        raise ValueError("finite complete right-receiving motor tape required")
    return arrays


def replay(
    model: mujoco.MjModel, tape: dict[str, np.ndarray], weights: np.ndarray
) -> dict[str, Any]:
    if weights.shape != (6,) or not np.isfinite(weights).all() or np.any(np.abs(weights) > 1):
        raise ValueError("six bounded finite right-leg coefficients required")
    data = mujoco.MjData(model)
    data.qpos[:] = tape["observation_qpos"][0]
    data.qvel[:] = tape["observation_qvel"][0]
    data.time = START * 0.02
    mujoco.mj_forward(model, data)
    ball_geom, robot_mask, foot_mask = _contact_masks(model)
    guarded_torque = 0.85 * np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    first_foot: int | None = None
    nonfoot_frames: set[int] = set()
    root_error = []
    ball_error = []
    speed_error = []
    minimum_height = float(data.qpos[2])
    max_parent_torque_difference = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for frame in range(START, STOP):
        row = frame - START
        root_error.append(float(np.linalg.norm(data.qpos[:3] - tape["observation_qpos"][row, :3])))
        ball_error.append(
            float(np.linalg.norm(data.qpos[36:39] - tape["observation_qpos"][row, 36:39]))
        )
        speed_error.append(
            float(np.linalg.norm(data.qvel[35:38] - tape["observation_qvel"][row, 35:38]))
        )
        ramp = min(1.0, max(0.0, (frame - START) / 6.0))
        fade = 1.0 if first_foot is None else max(0.0, 1.0 - (frame - first_foot) / 10.0)
        delta = 0.05 * ramp * fade * weights
        for substep in range(10):
            index = frame * 10 + substep
            target = tape["motor_pd_target_rad"][index].copy()
            target[RIGHT_LEG] += delta
            kp = tape["motor_kp"][index]
            kd = tape["motor_kd"][index]
            parent_pd = (
                kp * (tape["motor_pd_target_rad"][index] - tape["motor_joint_position_rad"][index])
                - kd * tape["motor_joint_velocity_radps"][index]
            )
            extra = tape["motor_raw_torque_nm"][index] - parent_pd
            raw = kp * (target - data.qpos[7:36]) - kd * data.qvel[6:35] + extra
            executed = np.clip(raw, -guarded_torque, guarded_torque)
            if not np.any(weights):
                max_parent_torque_difference = max(
                    max_parent_torque_difference,
                    float(np.max(np.abs(executed - tape["motor_executed_torque_nm"][index]))),
                )
            data.ctrl[:] = executed
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
                    nonfoot_frames.add(frame)
    ball = np.asarray(data.qpos[36:39], dtype=np.float64)
    foot_geoms = np.flatnonzero(foot_mask)
    foot_distance = min(float(np.linalg.norm(data.geom_xpos[int(g)] - ball)) for g in foot_geoms)
    speed = float(np.linalg.norm(data.qvel[35:38]))
    parent_first = int(np.flatnonzero(tape["ball_contact_agent_code"][:STOP] == 6)[0])
    safe = bool(
        minimum_height >= 0.65 and np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    )
    clean = bool(first_foot is not None and not nonfoot_frames)
    return {
        "weights": weights.tolist(),
        "safe": safe,
        "clean_first_foot": clean,
        "first_foot_frame": first_foot,
        "parent_first_foot_frame": parent_first,
        "nonfoot_frames": sorted(nonfoot_frames),
        "minimum_pelvis_height_m": minimum_height,
        "terminal_ball_speed_mps": speed,
        "terminal_nearest_foot_geom_distance_m": foot_distance,
        "root_position_rms_error_m": float(np.sqrt(np.mean(np.square(root_error)))),
        "ball_position_rms_error_m": float(np.sqrt(np.mean(np.square(ball_error)))),
        "ball_speed_max_error_mps": max(speed_error),
        "maximum_parent_torque_difference_nm": max_parent_torque_difference,
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
    asset_root: Path,
    capture_dir: Path,
    fidelity_report: Path,
    output: Path,
    *,
    population: int,
    seed: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or output.resolve().is_relative_to(root)
        or not 4 <= population <= 128
        or not 0 <= seed < 2**31
    ):
        raise ValueError("new bounded external SIM_ONLY PD-proxy search required")
    capture = json.loads((capture_dir / "report.json").read_text())
    fidelity = json.loads(fidelity_report.read_text())
    if (
        capture["report_hash"]
        != hash_json({k: v for k, v in capture.items() if k != "report_hash"})
        or fidelity["report_hash"]
        != hash_json({k: v for k, v in fidelity.items() if k != "report_hash"})
        or fidelity["capture_report_hash"] != capture["report_hash"]
        or fidelity["status"] != "EXACT_TORQUE_PROXY_QUALIFIED"
    ):
        raise ValueError("qualified same-asset exact-torque proxy required")
    right = next(row for row in capture["rows"] if row["course"]["lateral_m"] < 0)
    tape = load_tape(capture_dir / f"course-{right['course']['seed']}.npz", right["trace_hash"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_pd_proxy_search_v140.py",
            "scripts/rsi_r1_contact_proxy_fidelity_v139.py",
            "src/rosclaw_soccer/world/field.py",
        )
    }
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    output.mkdir(parents=True)
    baseline = replay(model, tape, np.zeros(6))
    if (
        baseline["first_foot_frame"] != baseline["parent_first_foot_frame"]
        or baseline["nonfoot_frames"] != [26]
        or baseline["root_position_rms_error_m"] > 1e-5
        or baseline["ball_position_rms_error_m"] > 1e-5
        or baseline["ball_speed_max_error_mps"] > 1e-4
        or baseline["maximum_parent_torque_difference_nm"] > 1e-5
    ):
        raise ValueError("causal-PD zero intervention does not replay the eight-player parent")
    rng = np.random.default_rng(seed)
    candidates = [np.zeros(6)] + [
        np.clip(rng.normal(0, 0.45, 6), -1, 1) for _ in range(population - 1)
    ]
    rows = []
    for index, weights in enumerate(candidates):
        row = replay(model, tape, weights)
        row["candidate"] = index
        rows.append(row)
        (output / "progress.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
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
                        "terminal_nearest_foot_geom_distance_m",
                    )
                }
            ),
            flush=True,
        )
    selected = max(rows, key=rank)
    report = {
        "schema": SCHEMA,
        "capture_report_hash": capture["report_hash"],
        "fidelity_report_hash": fidelity["report_hash"],
        "sources": sources,
        "search_partition": "CONSUMED_RIGHT_EIGHT_G1_FROZEN_EXTRA_TORQUE_PROXY",
        "seed": seed,
        "candidate_count": len(rows),
        "baseline": baseline,
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
        raise ValueError("source drift during frozen-extra PD proxy search")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--fidelity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.capture_dir,
        args.fidelity_report,
        args.output,
        population=args.population,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
