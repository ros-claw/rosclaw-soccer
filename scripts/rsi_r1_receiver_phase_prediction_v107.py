"""Frozen paired phase-prediction check in original contact-teacher authority."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_receiver_bridge_v71 import run

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _candidate(
    asset_root: Path,
    output: Path,
    scene: dict[str, Any],
    horizon: float,
    lateral_only: bool = False,
) -> dict[str, Any]:
    run(
        asset_root,
        output,
        enabled=True,
        preview_pass=True,
        motor_agent_id="red.playmaker",
        motor_entry_frame=0,
        duration_sec=10.0,
        directed_pass_speed_mps=1.0,
        precontact_pass_standoff_m=0.35,
        handoff_profile="tracking",
        stance_lateral_m=-0.19,
        ball_x_m=scene["ball_x_m"],
        ball_y_m=scene["ball_y_m"],
        seed=scene["seed"],
        receive_teacher_tuning=(0.0, 0.18),
        receive_velocity_prediction_sec=horizon,
        receive_velocity_prediction_lateral_only=lateral_only,
    )
    protocol_path, report_path, trace_path = (
        output / "protocol.json",
        output / "report.json",
        output / "trace.npz",
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report["report_hash"] != hash_json({k: v for k, v in report.items() if k != "report_hash"})
        or report["protocol_hash"] != hash_bytes(protocol_path.read_bytes())
        or report["trace_hash"] != hash_bytes(trace_path.read_bytes())
        or protocol["receive_velocity_prediction_sec"] != horizon
        or protocol["receive_velocity_prediction_lateral_only"] is not lateral_only
        or protocol["dual_receiver_motor"] is not False
        or protocol["activation_ceiling"] != "SIM_ONLY"
        or protocol["promotion_authorized"] is not False
    ):
        raise ValueError(f"unbound phase-prediction candidate: {output}")
    with np.load(trace_path, allow_pickle=False) as trace:
        own = np.flatnonzero(
            (trace["ball_contact_agent_code"] == 4)
            & np.isin(trace["ball_contact_foot_code"], (1, 2))
        )
        first = int(own[0]) if len(own) else None
        incoming = (
            None
            if first is None or first == 0
            else float(np.linalg.norm(trace["ball_velocity"][first - 1, :2]))
        )
        second = []
        if first is not None:
            for index in own:
                index = int(index)
                if float(trace["time"][index]) - float(trace["time"][first]) < 0.20 or np.any(
                    trace["ball_nonfoot_contact_agent_code"][first : index + 1] != 0
                ):
                    continue
                speed = float(np.linalg.norm(trace["ball_velocity"][index, :3]))
                if speed >= 2.0:
                    second.append({"frame": index, "speed_mps": speed})
        teacher_frames = int(np.count_nonzero(trace["contact_teacher_active"]))
    safe = bool(report["result"]["safe"])
    clean = bool((report["chain"] or {}).get("clean_transfer_observed"))
    collision_free = int(report["result"]["robot_robot_contact_count"]) == 0
    return {
        "scene": scene["id"],
        "horizon_sec": horizon,
        "report_hash": report["report_hash"],
        "protocol_hash": report["protocol_hash"],
        "trace_hash": report["trace_hash"],
        "safe": safe,
        "clean_transfer": clean,
        "robot_collision_free": collision_free,
        "incoming_speed_mps": incoming,
        "first_receiver_foot_frame": first,
        "clean_second_foot_events": second,
        "contact_teacher_active_frames": teacher_frames,
        "local_b6_passed": bool(
            safe
            and clean
            and collision_free
            and incoming is not None
            and incoming >= 0.30
            and second
        ),
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
    }


def train(asset_root: Path, protocol_path: Path, output: Path, workers: int = 2) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_scenes = [
        {"id": "s02", "ball_x_m": 2.24, "ball_y_m": -0.76, "seed": 512003},
        {"id": "s04", "ball_x_m": 2.32, "ball_y_m": -0.76, "seed": 512005},
    ]
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.r1_receiver_phase_prediction_v107.protocol.v1"
        or protocol["partition"] != "CONSUMED_DEV_CONTACT_ACTION"
        or protocol["prediction_sec_grid"] != [0.0, 0.04, 0.08, 0.12]
        or protocol["scenes"] != expected_scenes
        or protocol["rollout_count"] != 8
        or protocol["frozen_control"]["dual_receiver_motor"] is not False
        or not 1 <= workers <= 4
        or output.exists()
    ):
        raise ValueError("frozen bounded phase-prediction course and new output required")
    root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(root):
        raise ValueError("external immutable phase-prediction evidence required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_receiver_phase_prediction_v107.py",
            "scripts/rsi_r1_receiver_bridge_v71.py",
            "src/rosclaw_soccer/growth/locomotion_contact_teacher.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _candidate,
                asset_root,
                output / f"horizon-{horizon:.2f}-{scene['id']}",
                scene,
                horizon,
            )
            for horizon in protocol["prediction_sec_grid"]
            for scene in expected_scenes
        ]
        rows = [future.result() for future in futures]
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("phase-prediction source changed during physical rollouts")
    baseline = rows[0]
    baseline_reproduced = bool(
        baseline["local_b6_passed"]
        and baseline["first_receiver_foot_frame"] is not None
        and abs(baseline["first_receiver_foot_frame"] - 66) <= 2
        and any(abs(event["frame"] - 83) <= 2 for event in baseline["clean_second_foot_events"])
    )
    candidates = [
        {
            "horizon_sec": horizon,
            "scenes": rows[2 * index : 2 * index + 2],
            "two_scene_gate_passed": all(
                row["local_b6_passed"] for row in rows[2 * index : 2 * index + 2]
            ),
        }
        for index, horizon in enumerate(protocol["prediction_sec_grid"])
    ]
    selected = next((row for row in candidates[1:] if row["two_scene_gate_passed"]), None)
    result = {
        "schema": "rosclaw_soccer.rsi.r1_receiver_phase_prediction_v107.result.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "source_hashes": sources,
        "actual_rollouts": len(rows),
        "baseline_reproduced": baseline_reproduced,
        "candidates": candidates,
        "status": (
            "INVALID_BASELINE_REPRODUCTION"
            if not baseline_reproduced
            else "CONSUMED_PHASE_TEACHER_PASSED"
            if selected
            else "REJECTED_TRAINING"
        ),
        "selected_horizon_sec": (
            selected["horizon_sec"] if selected and baseline_reproduced else None
        ),
        "fresh_evaluation_run": False,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    (output / "selection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    result = train(args.asset_root, args.protocol, args.output, args.workers)
    print(json.dumps({"status": result["status"], "report_hash": result["report_hash"]}))


if __name__ == "__main__":
    main()
