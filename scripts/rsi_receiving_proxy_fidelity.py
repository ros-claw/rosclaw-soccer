"""SIM_ONLY CPU single-G1 fidelity gate for the consumed 8-G1 receiving miss."""

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


def verify(*, asset_root: Path, captured: Path, output_dir: Path) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY fidelity directory required")
    trace_path = captured / "motor-trace.npz"
    capture_report: dict[str, Any] = json.loads(
        (captured / "report.json").read_text(encoding="utf-8")
    )
    capture_commitment = capture_report.pop("report_hash")
    if (
        capture_commitment != hash_json(capture_report)
        or capture_report["trace_hash"] != hash_bytes(trace_path.read_bytes())
        or capture_report["promotion_authorized"] is not False
    ):
        raise ValueError("sealed consumed receiving capture required")
    with np.load(trace_path, allow_pickle=False) as trace:
        qpos = np.asarray(trace["sonic_recorded_qpos"], dtype=np.float64)
        qvel = np.asarray(trace["sonic_recorded_qvel"], dtype=np.float64)
        target = np.asarray(trace["sonic_recorded_target"], dtype=np.float64)
        kp = np.asarray(trace["sonic_recorded_kp"], dtype=np.float64)
        kd = np.asarray(trace["sonic_recorded_kd"], dtype=np.float64)
        contact_agent = np.asarray(trace["ball_contact_agent_code"], dtype=np.int64)
        contact_foot = np.asarray(trace["ball_contact_foot_code"], dtype=np.int64)
        nonfoot_agent = np.asarray(trace["ball_nonfoot_contact_agent_code"], dtype=np.int64)
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
        raise ValueError("complete measured local motor trace required")
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    snapshot = 20
    data.qpos[:] = qpos[snapshot]
    data.qvel[:] = qvel[snapshot]
    mujoco.mj_forward(model, data)
    ball_geom, robot_mask, foot_mask = _contact_masks(model)
    limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    hits = []
    min_height = float(data.qpos[2])
    for frame in range(snapshot + 1, 71):
        for substep in range(10):
            data.ctrl[:] = np.clip(
                kp[frame] * (target[frame] - data.qpos[7:36]) - kd[frame] * data.qvel[6:35],
                -limits,
                limits,
            )
            mujoco.mj_step(model, data)
            min_height = min(min_height, float(data.qpos[2]))
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                a, b = int(contact.geom1), int(contact.geom2)
                if a != ball_geom and b != ball_geom:
                    continue
                other = b if a == ball_geom else a
                if robot_mask[other]:
                    hits.append((frame, substep, bool(foot_mask[other])))
    agents = [row["agent_id"] for row in capture_report["world_result"]["qualities"]]
    focal_code = agents.index("red.finisher") + 1
    shared_frames = np.flatnonzero((contact_agent == focal_code) | (nonfoot_agent == focal_code))
    shared_first = int(shared_frames[0]) if len(shared_frames) else None
    proxy_first = hits[0][0] if hits else None
    shared_foot = bool(np.any((contact_agent == focal_code) & (contact_foot > 0)))
    shared_nonfoot = bool(np.any(nonfoot_agent == focal_code))
    proxy_foot = any(row[2] for row in hits)
    proxy_nonfoot = any(not row[2] for row in hits)
    faithful = bool(
        shared_first is not None
        and proxy_first is not None
        and abs(proxy_first - shared_first) <= 2
        and proxy_foot == shared_foot
        and proxy_nonfoot == shared_nonfoot
        and min_height >= 0.65
    )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("proxy source changed during CPU physics")
    result: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_proxy_fidelity.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "capture_hash": capture_commitment,
        "compiled_model_hash": compiled_model_hash(model),
        "snapshot_frame": snapshot,
        "shared_first_contact_frame": shared_first,
        "proxy_first_contact_frame": proxy_first,
        "shared_foot_seen": shared_foot,
        "shared_nonfoot_seen": shared_nonfoot,
        "proxy_foot_seen": proxy_foot,
        "proxy_nonfoot_seen": proxy_nonfoot,
        "minimum_proxy_pelvis_height_m": min_height,
        "fidelity_gate_passed": faithful,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--captured", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    print(json.dumps(verify(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
