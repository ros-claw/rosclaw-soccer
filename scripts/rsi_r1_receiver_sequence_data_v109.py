"""Collect authenticated physical receiver sequences without success inflation."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_receiver_bridge_v71 import run

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _episode(
    asset_root: Path,
    output: Path,
    scene: dict[str, Any],
    setting: dict[str, Any],
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
        receive_teacher_tuning=(setting["yaw_rad"], setting["lateral_m"]),
        capture_b6_microphysics=True,
    )
    protocol_path, report_path, trace_path = (
        output / "protocol.json",
        output / "report.json",
        output / "trace.npz",
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    micro = report["b6_microphysics"]
    if (
        report["report_hash"] != hash_json({k: v for k, v in report.items() if k != "report_hash"})
        or report["protocol_hash"] != hash_bytes(protocol_path.read_bytes())
        or report["trace_hash"] != hash_bytes(trace_path.read_bytes())
        or protocol["receive_teacher_tuning"] != [setting["yaw_rad"], setting["lateral_m"]]
        or protocol["scenario"]["seed"] != scene["seed"]
        or protocol["capture_b6_microphysics"] is not True
        or protocol["dual_receiver_motor"] is not False
        or protocol["activation_ceiling"] != "SIM_ONLY"
        or protocol["promotion_authorized"] is not False
        or not isinstance(micro, dict)
        or micro["observer_fault"] is not False
    ):
        raise ValueError(f"unbound receiver sequence episode: {output}")
    if micro["complete"]:
        archive = output / "b6-microphysics.npz"
        if micro["archive_hash"] != hash_bytes(archive.read_bytes()):
            raise ValueError(f"receiver microphysics archive changed: {output}")
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
    clean = bool((report["chain"] or {}).get("clean_transfer_observed"))
    safe = bool(report["result"]["safe"])
    collision_free = int(report["result"]["robot_robot_contact_count"]) == 0
    positive = bool(
        safe and clean and collision_free and incoming is not None and incoming >= 0.30 and second
    )
    return {
        "scene": scene["id"],
        "setting": setting["id"],
        "output": str(output),
        "report_hash": report["report_hash"],
        "protocol_hash": report["protocol_hash"],
        "trace_hash": report["trace_hash"],
        "microphysics_hash": micro["archive_hash"],
        "microphysics_complete": bool(micro["complete"]),
        "safe": safe,
        "clean_transfer": clean,
        "robot_collision_free": collision_free,
        "incoming_speed_mps": incoming,
        "first_receiver_foot_frame": first,
        "clean_second_foot_events": second,
        "teacher_active_frames": teacher_frames,
        "dynamic_b6_positive": positive,
    }


def collect(
    asset_root: Path, protocol_path: Path, output: Path, workers: int = 3
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_settings = [
        {"id": "parent", "yaw_rad": 0.0, "lateral_m": 0.18},
        {"id": "inner", "yaw_rad": 0.0, "lateral_m": 0.16},
        {"id": "left", "yaw_rad": -0.1, "lateral_m": 0.18},
        {"id": "right", "yaw_rad": 0.1, "lateral_m": 0.18},
    ]
    expected_scenes = [
        {"id": "s02", "ball_x_m": 2.24, "ball_y_m": -0.76, "seed": 512003},
        {"id": "s04", "ball_x_m": 2.32, "ball_y_m": -0.76, "seed": 512005},
        {"id": "x20", "ball_x_m": 2.20, "ball_y_m": -0.76, "seed": 600101},
        {"id": "x28", "ball_x_m": 2.28, "ball_y_m": -0.76, "seed": 600102},
        {"id": "y72", "ball_x_m": 2.24, "ball_y_m": -0.72, "seed": 600103},
        {"id": "y80", "ball_x_m": 2.24, "ball_y_m": -0.80, "seed": 600104},
    ]
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.r1_receiver_sequence_data_v109.protocol.v1"
        or protocol["partition"] != "CONSUMED_DEV_DATA"
        or protocol["teacher_settings"] != expected_settings
        or protocol["scenes"] != expected_scenes
        or protocol["rollout_count"] != 24
        or protocol["frozen_control"]["duration_sec"] != 10.0
        or not 1 <= workers <= 4
        or output.exists()
    ):
        raise ValueError("frozen bounded receiver sequence collection required")
    root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(root):
        raise ValueError("external immutable sequence evidence required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_receiver_sequence_data_v109.py",
            "scripts/rsi_r1_receiver_bridge_v71.py",
            "src/rosclaw_soccer/rsi/b6_microphysics_observer.py",
            "src/rosclaw_soccer/growth/locomotion_contact_teacher.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _episode,
                asset_root,
                output / f"{setting['id']}-{scene['id']}",
                scene,
                setting,
            )
            for setting in protocol["teacher_settings"]
            for scene in protocol["scenes"]
        ]
        rows = [future.result() for future in futures]
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("sequence collection source changed during physical rollouts")
    anchor = rows[0]
    anchor_reproduced = bool(
        anchor["dynamic_b6_positive"]
        and anchor["first_receiver_foot_frame"] is not None
        and abs(anchor["first_receiver_foot_frame"] - 66) <= 2
        and any(abs(event["frame"] - 83) <= 2 for event in anchor["clean_second_foot_events"])
    )
    positive = sum(row["dynamic_b6_positive"] for row in rows)
    negative = len(rows) - positive
    captured = sum(row["microphysics_complete"] for row in rows)
    sufficient = bool(anchor_reproduced and captured >= 20 and positive >= 4 and negative >= 8)
    result = {
        "schema": "rosclaw_soccer.rsi.r1_receiver_sequence_data_v109.result.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "source_hashes": sources,
        "actual_rollouts": len(rows),
        "anchor_reproduced": anchor_reproduced,
        "microphysics_complete_count": captured,
        "positive_count": positive,
        "negative_count": negative,
        "rows": rows,
        "status": "CONSUMED_SEQUENCE_DATA_READY" if sufficient else "INSUFFICIENT_SEQUENCE_DATA",
        "sequence_actor_training_authorized": sufficient,
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
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    result = collect(args.asset_root, args.protocol, args.output, args.workers)
    print(
        json.dumps(
            {
                "status": result["status"],
                "positive_count": result["positive_count"],
                "microphysics_complete_count": result["microphysics_complete_count"],
                "report_hash": result["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
