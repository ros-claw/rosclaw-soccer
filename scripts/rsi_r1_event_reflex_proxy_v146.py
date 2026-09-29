"""SIM_ONLY bounded 500 Hz post-foot reflex search in the qualified R1 proxy.

This remains a consumed, single-body, fixed high-level-schedule development test.
The post-foot update is event-driven; zero action must reproduce the source tape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray
from rsi_mjx_contact_replay_smoke import _contact_masks
from rsi_r1_dynamic_teacher_proxy_v141 import teacher_torque
from rsi_r1_pd_proxy_search_v140 import load_tape

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.training.receiving_classroom import r1_contact_tap_receiving_configuration
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

SCHEMA = "rosclaw_soccer.rsi.r1_event_reflex_proxy_v146.result.v1"
FRAMES = 60


def replay(
    model: mujoco.MjModel,
    tape: dict[str, NDArray[np.float64]],
    weights: NDArray[np.float64],
) -> dict[str, Any]:
    if weights.shape != (6,) or not np.isfinite(weights).all() or np.any(np.abs(weights) > 1):
        raise ValueError("six bounded post-foot right-leg coordinates required")
    data = mujoco.MjData(model)
    data.qpos[:] = tape["initial_local_qpos"]
    data.qvel[:] = tape["initial_local_qvel"]
    mujoco.mj_forward(model, data)
    ball_geom, robot_mask, foot_mask = _contact_masks(model)
    shin_geom = model.geom("right_shin").id
    _, teacher_base = r1_contact_tap_receiving_configuration()
    guard = 0.85 * np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    wrench = np.zeros(6, dtype=np.float64)
    closest = np.zeros(6, dtype=np.float64)
    filtered = np.zeros(29, dtype=np.float64)
    desired = np.zeros(29, dtype=np.float64)
    first_foot_substep: int | None = None
    first_shin_substep: int | None = None
    nonfoot_frames: set[int] = set()
    minimum_height = float(data.qpos[2])
    minimum_shin_gap = float("inf")
    max_parent_torque_error = 0.0
    for index in range(FRAMES * 10):
        frame = index // 10
        if first_foot_substep is not None:
            desired[6:12] = 0.25 * weights
            filtered += np.clip(0.45 * (desired - filtered), -0.04, 0.04)
        kp = tape["motor_kp"][index]
        kd = tape["motor_kd"][index]
        target = tape["motor_pd_target_rad"][index] + filtered
        position_torque = kp * (target - data.qpos[7:36])
        pd = position_torque - kd * data.qvel[6:35]
        extra = teacher_torque(
            model, data, tape["motor_teacher_inputs"][index], teacher_base, position_torque
        )
        torque = np.clip(pd + extra, -guard, guard)
        if not np.any(weights):
            max_parent_torque_error = max(
                max_parent_torque_error,
                float(np.max(np.abs(torque - tape["motor_executed_torque_nm"][index]))),
            )
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)
        minimum_height = min(minimum_height, float(data.qpos[2]))
        if 20 <= frame <= 38:
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
            if foot_mask[other] and first_foot_substep is None:
                first_foot_substep = index
            elif not foot_mask[other]:
                nonfoot_frames.add(frame)
                if other == shin_geom and first_shin_substep is None:
                    first_shin_substep = index
    ball = data.qpos[36:39]
    foot_distance = min(
        float(np.linalg.norm(data.geom_xpos[g] - ball)) for g in np.flatnonzero(foot_mask)
    )
    return {
        "weights": weights.tolist(),
        "safe": bool(
            minimum_height >= 0.65 and np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        ),
        "first_foot_substep": first_foot_substep,
        "first_shin_substep": first_shin_substep,
        "nonfoot_frames": sorted(nonfoot_frames),
        "minimum_right_shin_gap_m": minimum_shin_gap,
        "minimum_pelvis_height_m": minimum_height,
        "terminal_foot_distance_m": foot_distance,
        "terminal_ball_speed_mps": float(np.linalg.norm(data.qvel[35:38])),
        "max_parent_torque_error_nm": max_parent_torque_error,
    }


def score(row: dict[str, Any]) -> float:
    if not row["safe"] or row["first_foot_substep"] is None:
        return -1000.0
    clean = not row["nonfoot_frames"]
    return float(
        100.0 * float(clean)
        + 10.0 * min(0.02, row["minimum_right_shin_gap_m"])
        - 2.0 * row["terminal_foot_distance_m"]
        - row["terminal_ball_speed_mps"]
    )


def search(asset_root: Path, capture_dir: Path, output: Path, *, seed: int) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root) or not 0 <= seed < 2**31:
        raise ValueError("new bounded external SIM_ONLY evidence required")
    capture = json.loads((capture_dir / "report.json").read_text())
    if capture["status"] != "CAPTURE_QUALIFIED" or capture["report_hash"] != hash_json(
        {k: v for k, v in capture.items() if k != "report_hash"}
    ):
        raise ValueError("qualified physical tape required")
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    tapes = [
        load_tape(capture_dir / f"course-{row['course']['seed']}.npz", row["trace_hash"])
        for row in capture["rows"]
    ]
    if len(tapes) != 2:
        raise ValueError("paired bilateral tape required")
    source_hash = hash_bytes((root / "scripts/rsi_r1_event_reflex_proxy_v146.py").read_bytes())
    output.mkdir(parents=True)
    baseline = [replay(model, tape, np.zeros(6, dtype=np.float64)) for tape in tapes]
    if (
        [
            row["first_foot_substep"] // 10 if row["first_foot_substep"] is not None else None
            for row in baseline
        ]
        != [33, 25]
        or [row["nonfoot_frames"] for row in baseline] != [[], [26]]
        or max(row["max_parent_torque_error_nm"] for row in baseline) > 1e-8
    ):
        raise ValueError("zero-action replay must reproduce paired parent contacts")
    rng = np.random.default_rng(seed)
    mean = np.zeros(6, dtype=np.float64)
    std = np.full(6, 0.5, dtype=np.float64)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for generation in range(4):
        sampled = np.clip(rng.normal(mean, std, size=(64, 6)), -1, 1)
        sampled[0] = 0
        ranked = []
        for weights in sampled:
            right = replay(model, tapes[1], weights)
            ranked.append((score(right), weights, right))
        ranked.sort(key=lambda item: item[0], reverse=True)
        elite = np.stack([item[1] for item in ranked[:8]])
        mean = elite.mean(axis=0)
        std = np.maximum(0.08, elite.std(axis=0))
        candidate = ranked[0][2]
        if best is None or score(candidate) > score(best):
            best = candidate
        history.append(
            {
                "generation": generation + 1,
                "right_clean_count": sum(not item[2]["nonfoot_frames"] for item in ranked),
                "best_right": candidate,
            }
        )
        (output / "progress.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "generation": generation + 1,
                    "clean": history[-1]["right_clean_count"],
                    "best": candidate,
                }
            ),
            flush=True,
        )
    assert best is not None
    retained_left = replay(model, tapes[0], np.asarray(best["weights"], dtype=np.float64))
    paired_clean = all(
        row["safe"] and row["first_foot_substep"] is not None and not row["nonfoot_frames"]
        for row in (retained_left, best)
    )
    report = {
        "schema": SCHEMA,
        "capture_report_hash": capture["report_hash"],
        "source_hash": source_hash,
        "partition": "CONSUMED_POST_FOOT_RIGHT_REFLEX_PROXY",
        "seed": seed,
        "episode_count": 260,
        "baseline": baseline,
        "best_right": best,
        "retained_left": retained_left,
        "history": history,
        "status": "DEVELOPMENT_PAIRED_CLEAN_UNVALIDATED"
        if paired_clean
        else "REJECTED_PAIRED_CLEAN_GATE",
        "fixed_high_level_teacher_schedule": True,
        "full_world_audition_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if hash_bytes((root / "scripts/rsi_r1_event_reflex_proxy_v146.py").read_bytes()) != source_hash:
        raise ValueError("source drift during event-driven physical search")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    report = search(args.asset_root, args.capture_dir, args.output, seed=args.seed)
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
