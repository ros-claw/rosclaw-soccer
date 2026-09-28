"""SIM_ONLY CPU proxy fidelity for the exact clean-foot SONIC receiving teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rsi_mjx_contact_replay_smoke import _contact_masks

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

SNAPSHOT = 45
END_FRAME = 100
EXAM_FRAME = 86


def verify(*, asset_root: Path, captured: Path, output_dir: Path) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY fidelity directory required")
    trace_path = captured / "motor-trace.npz"
    capture: dict[str, Any] = json.loads((captured / "report.json").read_text(encoding="utf-8"))
    commitment = capture.pop("report_hash")
    if (
        commitment != hash_json(capture)
        or capture["trace_hash"] != hash_bytes(trace_path.read_bytes())
        or not capture["exact_physical_arrays"]
        or not capture["same_world_trajectory"]
    ):
        raise ValueError("sealed exact clean-touch motor trace required")
    with np.load(trace_path, allow_pickle=False) as trace:
        qpos = np.asarray(trace["sonic_recorded_qpos"], dtype=np.float64)
        qvel = np.asarray(trace["sonic_recorded_qvel"], dtype=np.float64)
        target = np.asarray(trace["sonic_recorded_target"], dtype=np.float64)
        kp = np.asarray(trace["sonic_recorded_kp"], dtype=np.float64)
        kd = np.asarray(trace["sonic_recorded_kd"], dtype=np.float64)
        shared_ball_pose = np.asarray(trace["ball_pose"], dtype=np.float64)
        shared_ball_velocity = np.asarray(trace["ball_velocity"], dtype=np.float64)
        contact_agent = np.asarray(trace["ball_contact_agent_code"], dtype=np.int64)
        contact_foot = np.asarray(trace["ball_contact_foot_code"], dtype=np.int64)
        nonfoot_agent = np.asarray(trace["ball_nonfoot_contact_agent_code"], dtype=np.int64)
        shared_pelvis = np.asarray(trace["red_finisher_pelvis_pose"], dtype=np.float64)
    if any(
        array.shape != shape
        for array, shape in (
            (qpos, (300, 43)),
            (qvel, (300, 41)),
            (target, (300, 29)),
            (kp, (300, 29)),
            (kd, (300, 29)),
        )
    ):
        raise ValueError("complete measured motor targets required")
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[SNAPSHOT]
    data.qvel[:] = qvel[SNAPSHOT]
    mujoco.mj_forward(model, data)
    ball_geom, robot_mask, foot_mask = _contact_masks(model)
    torque_limit = np.asarray(G1_HARD_TORQUE_LIMITS)
    hits: list[tuple[int, int, str]] = []
    minimum = float(data.qpos[2])
    ball_speed_exam = None
    distance_exam = None
    for frame in range(SNAPSHOT + 1, END_FRAME + 1):
        for substep in range(10):
            data.ctrl[:] = np.clip(
                kp[frame] * (target[frame] - data.qpos[7:36]) - kd[frame] * data.qvel[6:35],
                -torque_limit,
                torque_limit,
            )
            mujoco.mj_step(model, data)
            minimum = min(minimum, float(data.qpos[2]))
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                a, b = int(contact.geom1), int(contact.geom2)
                if a != ball_geom and b != ball_geom:
                    continue
                other = b if a == ball_geom else a
                if robot_mask[other]:
                    hits.append((frame, substep, "foot" if foot_mask[other] else "nonfoot"))
        if frame == EXAM_FRAME:
            ball_speed_exam = float(np.linalg.norm(data.qvel[35:37]))
            distance_exam = float(np.linalg.norm(data.qpos[36:38] - data.qpos[:2]))
    agents = [row["agent_id"] for row in capture["world_result"]["qualities"]]
    code = agents.index("red.finisher") + 1
    shared_frames = np.flatnonzero((contact_agent == code) | (nonfoot_agent == code))
    shared_first = int(shared_frames[0]) if len(shared_frames) else None
    proxy_first = hits[0][0] if hits else None
    shared_foot = bool(np.any((contact_agent == code) & (contact_foot > 0)))
    shared_nonfoot = bool(np.any(nonfoot_agent == code))
    proxy_foot = any(row[2] == "foot" for row in hits)
    proxy_nonfoot = any(row[2] == "nonfoot" for row in hits)
    shared_speed = float(np.linalg.norm(shared_ball_velocity[EXAM_FRAME, :2]))
    shared_distance = float(
        np.linalg.norm(shared_ball_pose[EXAM_FRAME, :2] - shared_pelvis[EXAM_FRAME, :2])
    )
    faithful = bool(
        shared_first is not None
        and proxy_first is not None
        and abs(proxy_first - shared_first) <= 2
        and proxy_foot == shared_foot
        and proxy_nonfoot == shared_nonfoot
        and ball_speed_exam is not None
        and abs(ball_speed_exam - shared_speed) <= 0.15
        and distance_exam is not None
        and abs(distance_exam - shared_distance) <= 0.10
        and minimum >= 0.65
    )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("clean proxy source changed during CPU physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_clean_proxy_fidelity.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "capture_hash": commitment,
        "compiled_model_hash": compiled_model_hash(model),
        "snapshot_frame": SNAPSHOT,
        "shared_first_contact_frame": shared_first,
        "proxy_first_contact_frame": proxy_first,
        "shared_foot_seen": shared_foot,
        "shared_nonfoot_seen": shared_nonfoot,
        "proxy_foot_seen": proxy_foot,
        "proxy_nonfoot_seen": proxy_nonfoot,
        "shared_half_second_ball_speed_mps": shared_speed,
        "proxy_half_second_ball_speed_mps": ball_speed_exam,
        "shared_half_second_ball_pelvis_distance_m": shared_distance,
        "proxy_half_second_ball_pelvis_distance_m": distance_exam,
        "minimum_proxy_pelvis_height_m": minimum,
        "fidelity_gate_passed": faithful,
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
    parser.add_argument("--output-dir", required=True, type=Path)
    print(json.dumps(verify(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
