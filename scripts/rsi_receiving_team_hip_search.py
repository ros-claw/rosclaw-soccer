"""SIM_ONLY exploratory hip-roll geometry scan; proxy is not yet off-trajectory qualified."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_receiving_team_short_replay import replay

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

GRID = (-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08)


def search(*, asset_root: Path, captured: Path, output_dir: Path) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    replay_hash = hash_bytes((source.parent / "rsi_receiving_team_short_replay.py").read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.parents[1]):
        raise ValueError("new external SIM_ONLY hip search directory required")
    captured_report: dict[str, Any] = json.loads(
        (captured / "report.json").read_text(encoding="utf-8")
    )
    captured_hash = captured_report.pop("report_hash")
    if (
        captured_hash != hash_json(captured_report)
        or captured_report["schema"] != "rosclaw_soccer.rsi.receiving_team_student_motor_capture.v1"
        or captured_report["promotion_authorized"] is not False
    ):
        raise ValueError("sealed student eight-G1 motor tape required")
    rows = []
    for index, offset in enumerate(GRID):
        exam = replay(
            asset_root=asset_root,
            captured=captured,
            output_dir=output_dir / f"candidate-{index:02d}",
            left_hip_roll_offset_rad=offset,
        )
        if exam["capture_report_hash"] != captured_hash:
            raise RuntimeError("research proxy source tape changed")
        name = "short-replay.npz" if offset == 0.0 else "short-hip-probe.npz"
        with np.load(output_dir / f"candidate-{index:02d}" / name, allow_pickle=False) as data:
            tail = slice(80 - 45, 100 - 45 + 1)
            mean_speed = float(np.mean(data["ball_speed_mps"][tail]))
            mean_distance = float(np.mean(data["ball_pelvis_distance_m"][tail]))
            mean_abs_vy = float(np.mean(np.abs(data["ball_vy_mps"][tail])))
        safe = bool(
            exam["all_robot_bodies_safe"]
            and not exam["own_nonfoot_seen"]
            and exam["replay_first_foot_frame"] is not None
        )
        row = {
            "index": index,
            "left_hip_roll_offset_rad": offset,
            "proxy_report_hash": exam["report_hash"],
            "safe": safe,
            "first_foot_frame": exam["replay_first_foot_frame"],
            "frame86_ball_speed_mps": exam["replay_frame86_ball_speed_mps"],
            "frame86_ball_vy_mps": exam["replay_frame86_ball_vy_mps"],
            "frame86_ball_pelvis_distance_m": exam["replay_frame86_ball_pelvis_distance_m"],
            "tail_mean_speed_mps": mean_speed,
            "tail_mean_abs_vy_mps": mean_abs_vy,
            "tail_mean_distance_m": mean_distance,
            "development_score": mean_speed + 0.4 * mean_distance,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    candidates = [row for row in rows if row["safe"]]
    best = min(candidates, key=lambda row: row["development_score"]) if candidates else None
    if (
        hash_bytes(source.read_bytes()) != source_hash
        or hash_bytes((source.parent / "rsi_receiving_team_short_replay.py").read_bytes())
        != replay_hash
    ):
        raise RuntimeError("hip research source changed during physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_team_hip_search.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "replay_source_hash": replay_hash,
        "student_capture_report_hash": captured_hash,
        "grid_rad": list(GRID),
        "rows": rows,
        "selected_index": None if best is None else best["index"],
        "selected_development_score": None if best is None else best["development_score"],
        "off_trajectory_fidelity_established": False,
        "full_world_validated": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--captured", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    report = search(**vars(parser.parse_args()))
    print(
        json.dumps(
            {"report_hash": report["report_hash"], "selected_index": report["selected_index"]}
        )
    )


if __name__ == "__main__":
    main()
