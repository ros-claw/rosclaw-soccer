"""Test bounded sender-entry timing against measured receiver gait contact outcomes."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from rsi_r1_local_b6_teacher_grid_v102 import _candidate_task

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def train(asset_root: Path, protocol_path: Path, output: Path, workers: int = 3) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_scenes = [
        {"id": "s02", "ball_x_m": 2.24, "ball_y_m": -0.76, "seed": 512003},
        {"id": "s04", "ball_x_m": 2.32, "ball_y_m": -0.76, "seed": 512005},
    ]
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.r1_gait_aware_pass_timing_v105.protocol.v1"
        or protocol["partition"] != "CONSUMED_DEV_TEAM_TIMING"
        or protocol["motor_entry_frame_grid"] != [0, 5, 10, 15, 20, 25, 30]
        or protocol["scenes"] != expected_scenes
        or protocol["rollout_count"] != 14
        or protocol["frozen_control"]["duration_sec"] != 10.0
        or protocol["frozen_control"]["teacher_yaw_rad"] != 0.0
        or protocol["frozen_control"]["teacher_lateral_offset_m"] != 0.18
        or not 1 <= workers <= 4
        or output.exists()
    ):
        raise ValueError("frozen bounded team-timing curriculum and new output required")
    root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(root):
        raise ValueError("external immutable timing evidence required")
    source_hashes = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_gait_aware_pass_timing_v105.py",
            "scripts/rsi_r1_local_b6_teacher_grid_v102.py",
            "scripts/rsi_r1_receiver_bridge_v71.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _candidate_task,
                asset_root,
                output / f"entry-{frame:02d}-{scene['id']}",
                scene,
                0.0,
                0.18,
                frame,
                10.0,
            )
            for frame in protocol["motor_entry_frame_grid"]
            for scene in expected_scenes
        ]
        rows = [future.result() for future in futures]
    if any(
        hash_bytes((root / name).read_bytes()) != digest for name, digest in source_hashes.items()
    ):
        raise ValueError("team-timing source changed during physical rollouts")
    baseline = rows[0]
    baseline_reproduced = bool(
        baseline["local_b6_passed"]
        and baseline["first_receiver_foot_frame"] is not None
        and abs(baseline["first_receiver_foot_frame"] - 66) <= 2
        and any(abs(event["frame"] - 83) <= 2 for event in baseline["clean_second_foot_events"])
    )
    candidates = []
    for index, frame in enumerate(protocol["motor_entry_frame_grid"]):
        pair = rows[2 * index : 2 * index + 2]
        passed = all(row["local_b6_passed"] for row in pair)
        candidates.append(
            {
                "motor_entry_frame": frame,
                "scenes": pair,
                "two_scene_gate_passed": passed,
                "minimum_second_foot_speed_mps": (
                    min(
                        max(event["speed_mps"] for event in row["clean_second_foot_events"])
                        for row in pair
                    )
                    if passed
                    else None
                ),
            }
        )
    accepted = [candidate for candidate in candidates if candidate["two_scene_gate_passed"]]
    accepted.sort(
        key=lambda candidate: (
            candidate["motor_entry_frame"],
            -candidate["minimum_second_foot_speed_mps"],
        )
    )
    result = {
        "schema": "rosclaw_soccer.rsi.r1_gait_aware_pass_timing_v105.result.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "source_hashes": source_hashes,
        "actual_rollouts": len(rows),
        "baseline_reproduced": baseline_reproduced,
        "candidates": candidates,
        "status": (
            "INVALID_BASELINE_REPRODUCTION"
            if not baseline_reproduced
            else "CONSUMED_TEAM_TIMING_PASSED"
            if accepted
            else "REJECTED_TRAINING"
        ),
        "selected_entry_frame": (
            accepted[0]["motor_entry_frame"] if baseline_reproduced and accepted else None
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
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    result = train(args.asset_root, args.protocol, args.output, args.workers)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_entry_frame": result["selected_entry_frame"],
                "report_hash": result["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
