"""Audit whether a one-G1 MuJoCo proxy preserves eight-G1 R1 contact physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rsi_mjx_contact_replay_smoke import _contact_masks

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.role_receiving_courses import ROSTER
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

SCHEMA = "rosclaw_soccer.rsi.r1_contact_proxy_fidelity_v139.result.v1"
FIRST_FRAME = 15
LAST_FRAME = 120


def verify_one(asset_root: Path, tape_path: Path, expected_hash: str) -> dict[str, Any]:
    if hash_bytes(tape_path.read_bytes()) != expected_hash:
        raise ValueError("sealed eight-player motor tape required")
    with np.load(tape_path, allow_pickle=False) as tape:
        qpos = np.asarray(tape["observation_qpos"], dtype=np.float64)
        qvel = np.asarray(tape["observation_qvel"], dtype=np.float64)
        frames = np.asarray(tape["observation_frame"], dtype=np.int64)
        ctrl = np.asarray(tape["motor_executed_torque_nm"], dtype=np.float64)
        parent_foot = np.asarray(tape["ball_contact_agent_code"], dtype=np.int64)
        parent_effector = np.asarray(tape["ball_contact_effector_code"], dtype=np.int64)
        parent_foot_force = np.asarray(tape["ball_contact_force_n"], dtype=np.float64)
        parent_nonfoot = np.asarray(tape["ball_nonfoot_contact_agent_code"], dtype=np.int64)
        parent_nonfoot_force = np.asarray(tape["ball_nonfoot_contact_force_n"], dtype=np.float64)
        motor_frames = np.asarray(tape["motor_control_frame"], dtype=np.int64)
    if (
        qpos.shape[1:] != (43,)
        or qvel.shape[1:] != (41,)
        or len(qpos) < LAST_FRAME - FIRST_FRAME
        or not np.array_equal(
            frames[: LAST_FRAME - FIRST_FRAME], np.arange(FIRST_FRAME, LAST_FRAME)
        )
        or ctrl.shape[1:] != (29,)
        or len(ctrl) < LAST_FRAME * 10
        or not np.array_equal(motor_frames[: LAST_FRAME * 10], np.repeat(np.arange(LAST_FRAME), 10))
        or any(not np.isfinite(array).all() for array in (qpos, qvel, ctrl))
    ):
        raise ValueError("finite complete initial state and 500 Hz executed torque required")
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    if (model.nq, model.nv, model.nu) != (43, 41, 29):
        raise ValueError("same-dimension single-G1 proxy required")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0]
    data.qvel[:] = qvel[0]
    data.time = FIRST_FRAME * 0.02
    mujoco.mj_forward(model, data)
    ball_geom, robot_mask, foot_mask = _contact_masks(model)
    proxy_qpos = []
    proxy_qvel = []
    foot_frames: list[int] = []
    nonfoot_frames: list[int] = []
    minimum_pelvis = float(data.qpos[2])
    wrench = np.zeros(6, dtype=np.float64)
    for frame in range(FIRST_FRAME, LAST_FRAME):
        proxy_qpos.append(data.qpos.copy())
        proxy_qvel.append(data.qvel.copy())
        for substep in range(10):
            data.ctrl[:] = ctrl[frame * 10 + substep]
            mujoco.mj_step(model, data)
            minimum_pelvis = min(minimum_pelvis, float(data.qpos[2]))
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
                (foot_frames if foot_mask[other] else nonfoot_frames).append(frame)
    proxy_qpos_array = np.asarray(proxy_qpos)
    proxy_qvel_array = np.asarray(proxy_qvel)
    focal_code = tuple(sorted(ROSTER)).index("red.finisher") + 1
    parent_foot_frames = np.flatnonzero(
        (parent_foot[:LAST_FRAME] == focal_code)
        & np.isin(parent_effector[:LAST_FRAME], (1, 2))
        & (parent_foot_force[:LAST_FRAME] > 0)
    )
    parent_nonfoot_frames = np.flatnonzero(
        (parent_nonfoot[:LAST_FRAME] == focal_code) & (parent_nonfoot_force[:LAST_FRAME] > 0)
    )
    first_parent = int(parent_foot_frames[0]) if len(parent_foot_frames) else None
    first_proxy = min(foot_frames) if foot_frames else None
    stop = min(LAST_FRAME, (first_parent if first_parent is not None else FIRST_FRAME) + 11)
    count = stop - FIRST_FRAME
    ball_error = np.linalg.norm(proxy_qpos_array[:count, 36:39] - qpos[:count, 36:39], axis=1)
    root_error = np.linalg.norm(proxy_qpos_array[:count, :3] - qpos[:count, :3], axis=1)
    speed_error = np.linalg.norm(proxy_qvel_array[:count, 35:38] - qvel[:count, 35:38], axis=1)
    parent_nonfoot_early = any(FIRST_FRAME <= frame < stop for frame in parent_nonfoot_frames)
    proxy_nonfoot_early = any(FIRST_FRAME <= frame < stop for frame in nonfoot_frames)
    faithful = bool(
        first_parent is not None
        and first_proxy is not None
        and abs(first_proxy - first_parent) <= 2
        and parent_nonfoot_early == proxy_nonfoot_early
        and float(np.sqrt(np.mean(ball_error**2))) <= 0.08
        and float(np.sqrt(np.mean(root_error**2))) <= 0.08
        and float(np.max(speed_error)) <= 0.30
        and minimum_pelvis >= 0.65
    )
    return {
        "tape_hash": expected_hash,
        "parent_first_foot_frame": first_parent,
        "proxy_first_foot_frame": first_proxy,
        "parent_nonfoot_frames": parent_nonfoot_frames.tolist(),
        "proxy_nonfoot_frames": sorted(set(nonfoot_frames)),
        "parent_nonfoot_early": parent_nonfoot_early,
        "proxy_nonfoot_early": proxy_nonfoot_early,
        "audited_until_frame_exclusive": stop,
        "ball_position_rms_error_m": float(np.sqrt(np.mean(ball_error**2))),
        "root_position_rms_error_m": float(np.sqrt(np.mean(root_error**2))),
        "ball_speed_max_error_mps": float(np.max(speed_error)),
        "minimum_proxy_pelvis_height_m": minimum_pelvis,
        "proxy_fidelity_passed": faithful,
    }


def verify(asset_root: Path, capture_dir: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY proxy fidelity evidence required")
    capture_report = json.loads((capture_dir / "report.json").read_text())
    if (
        capture_report["schema"] != "rosclaw_soccer.rsi.r1_contact_proxy_capture_v138.result.v1"
        or capture_report["status"] != "CAPTURE_QUALIFIED"
        or capture_report["report_hash"]
        != hash_json({k: v for k, v in capture_report.items() if k != "report_hash"})
    ):
        raise ValueError("qualified source-frozen eight-player capture required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_contact_proxy_fidelity_v139.py",
            "scripts/rsi_r1_contact_proxy_capture_v138.py",
            "src/rosclaw_soccer/world/field.py",
        )
    }
    rows = [
        verify_one(
            asset_root, capture_dir / f"course-{row['course']['seed']}.npz", row["trace_hash"]
        )
        for row in capture_report["rows"]
    ]
    report = {
        "schema": SCHEMA,
        "capture_report_hash": capture_report["report_hash"],
        "source_hashes": sources,
        "rows": rows,
        "status": "EXACT_TORQUE_PROXY_QUALIFIED"
        if all(row["proxy_fidelity_passed"] for row in rows)
        else "REJECTED_PROXY_FIDELITY",
        "promotion_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during proxy fidelity audit")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.asset_root, args.capture_dir, args.output)
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
