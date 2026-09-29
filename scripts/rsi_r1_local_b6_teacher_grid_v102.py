"""Search a bounded privileged B6 receive teacher in real six-G1 MuJoCo episodes."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_receiver_bridge_v71 import run

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _candidate_task(
    asset_root: Path,
    output: Path,
    scene: dict[str, Any],
    yaw_rad: float,
    lateral_m: float,
) -> dict[str, Any]:
    run(
        asset_root,
        output,
        enabled=True,
        preview_pass=True,
        motor_agent_id="red.playmaker",
        directed_pass_speed_mps=1.0,
        precontact_pass_standoff_m=0.35,
        handoff_profile="tracking",
        stance_lateral_m=-0.19,
        ball_x_m=scene["ball_x_m"],
        ball_y_m=scene["ball_y_m"],
        seed=scene["seed"],
        receive_teacher_tuning=(yaw_rad, lateral_m),
    )
    protocol_path = output / "protocol.json"
    report_path = output / "report.json"
    trace_path = output / "trace.npz"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in report.items() if key != "report_hash"}
    if (
        report["report_hash"] != hash_json(body)
        or report["protocol_hash"] != hash_bytes(protocol_path.read_bytes())
        or report["trace_hash"] != hash_bytes(trace_path.read_bytes())
        or protocol["receive_teacher_tuning"] != [yaw_rad, lateral_m]
        or protocol["activation_ceiling"] != "SIM_ONLY"
        or protocol["promotion_authorized"] is not False
    ):
        raise ValueError(f"unbound physical B6 candidate: {output}")
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
    safe = bool(report["result"]["safe"])
    clean = bool((report["chain"] or {}).get("clean_transfer_observed"))
    collision_free = int(report["result"]["robot_robot_contact_count"]) == 0
    passed = bool(
        safe and clean and collision_free and incoming is not None and incoming >= 0.30 and second
    )
    return {
        "scene": scene["id"],
        "yaw_rad": yaw_rad,
        "lateral_m": lateral_m,
        "report_hash": report["report_hash"],
        "protocol_hash": report["protocol_hash"],
        "trace_hash": report["trace_hash"],
        "safe": safe,
        "clean_transfer": clean,
        "robot_collision_free": collision_free,
        "incoming_speed_mps": incoming,
        "first_receiver_foot_frame": first,
        "clean_second_foot_events": second,
        "local_b6_passed": passed,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
    }


def train(asset_root: Path, protocol_path: Path, output: Path, workers: int = 3) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.r1_local_b6_teacher_grid_v102.protocol.v1"
        or protocol["partition"] != "CONSUMED_DEV_TEACHER_SEARCH"
        or protocol["candidate_count"] != 9
        or protocol["rollout_count"] != 18
        or protocol["aim_yaw_grid_rad"] != [-0.2, 0.0, 0.2]
        or protocol["ankle_lateral_offset_grid_m"] != [0.12, 0.18, 0.24]
        or len(protocol["training_scenes"]) != 2
        or not 1 <= workers <= 4
        or output.exists()
    ):
        raise ValueError("frozen bounded B6 teacher course and new output required")
    root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(root):
        raise ValueError("external immutable evidence required")
    source_hashes = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_local_b6_teacher_grid_v102.py",
            "scripts/rsi_r1_receiver_bridge_v71.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    settings = [
        (float(yaw), float(lateral))
        for yaw in protocol["aim_yaw_grid_rad"]
        for lateral in protocol["ankle_lateral_offset_grid_m"]
    ]
    output.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _candidate_task,
                asset_root,
                output / f"candidate-{index:02d}-{scene['id']}",
                scene,
                yaw,
                lateral,
            )
            for index, (yaw, lateral) in enumerate(settings)
            for scene in protocol["training_scenes"]
        ]
        rows = [future.result() for future in futures]
    if any(
        hash_bytes((root / name).read_bytes()) != digest for name, digest in source_hashes.items()
    ):
        raise ValueError("B6 teacher source changed during physical rollouts")
    candidates: list[dict[str, Any]] = []
    for index, (yaw, lateral) in enumerate(settings):
        pair = rows[2 * index : 2 * index + 2]
        passed = all(row["local_b6_passed"] for row in pair)
        minimum_speed = (
            min(
                max(event["speed_mps"] for event in row["clean_second_foot_events"]) for row in pair
            )
            if passed
            else None
        )
        candidates.append(
            {
                "index": index,
                "yaw_rad": yaw,
                "lateral_m": lateral,
                "scenes": pair,
                "training_gate_passed": passed,
                "minimum_second_foot_speed_mps": minimum_speed,
            }
        )
    accepted = [row for row in candidates if row["training_gate_passed"]]
    ranked = sorted(
        accepted,
        key=lambda row: (
            -row["minimum_second_foot_speed_mps"],
            abs(row["yaw_rad"]),
            abs(row["lateral_m"] - 0.18),
        ),
    )
    result = {
        "schema": "rosclaw_soccer.rsi.r1_local_b6_teacher_grid_v102.result.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "source_hashes": source_hashes,
        "actual_rollouts": len(rows),
        "candidates": candidates,
        "status": "CONSUMED_TEACHER_TRAINING_PASSED" if ranked else "REJECTED_TRAINING",
        "selected_candidate_index": ranked[0]["index"] if ranked else None,
        "fresh_evaluation_run": False,
        "teacher_is_privileged": True,
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
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    result = train(args.asset_root, args.protocol, args.output, args.workers)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_candidate_index": result["selected_candidate_index"],
                "report_hash": result["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
