"""Import authenticated Isaac rolling-ball episodes without frame-count inflation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contact_training_samples import BODY_ORDER, extract_precontact_samples
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _episode(folder: Path) -> dict[str, Any]:
    report_path = folder / "report.json"
    trace_path = folder / "trajectory.npz"
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    committed = {key: value for key, value in report.items() if key != "report_hash"}
    if (
        report.get("report_hash") != hash_json(committed)
        or report.get("trajectory_hash") != hash_bytes(trace_path.read_bytes())
        or report.get("schema") != "rosclaw_soccer.rsi.isaac_sonic_stand.v3"
        or report.get("activation_ceiling") != "SIM_ONLY"
        or report.get("promotion_authorized") is not False
        or report.get("trained_actor") is not False
        or report.get("ball_rolling_start") is not True
        or report.get("track_ball_contacts") is not True
        or report.get("agent_count") != 1
        or report.get("forward_command_m_s") != 1.4
        or report.get("right_knee_contact_residual_rad") != 0.0
        or report.get("right_hip_pitch_contact_residual_rad") != 0.0
        or report.get("right_ankle_pitch_contact_residual_rad") != 0.0
        or report.get("rolling_contact_adapter", False) is not False
        or report.get("ball_body_contact_labels")
        != [
            "left_foot",
            "right_foot",
            "left_ankle_pitch",
            "right_ankle_pitch",
            "left_knee",
            "right_knee",
        ]
        or report.get("contact_kinematic_body_names") != list(BODY_ORDER)
    ):
        raise ValueError(f"unqualified rolling-ball Isaac report: {folder}")
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    samples = extract_precontact_samples(arrays, body_names=BODY_ORDER)
    if (
        samples.clean_foot_only != report["clean_foot_only_contact_verified"]
        or report["min_pelvis_height_m"] < 0.65
        or report["individual"][report["agent_ids"][0]]["joint_projection_count"] != 0
    ):
        raise ValueError("Isaac contact, body safety or trajectory evidence disagrees")
    force = arrays["ball_body_contact_force_micro_n"].reshape(-1, 6)
    first = report["first_foot_ball_contact_microstep"]
    side = None
    if first is not None:
        if type(first) is not int or not 0 <= first < len(force):
            raise ValueError("first foot contact index outside authenticated physics")
        side = "left" if force[first, 0] >= force[first, 1] else "right"
    return {
        "partition": "CONSUMED_DEV",
        "course": [
            report["ball_x_m"],
            report["ball_y_m"],
            report["ball_initial_vx_m_s"],
            report["ball_initial_vy_m_s"],
        ],
        "report_hash": report["report_hash"],
        "trajectory_hash": report["trajectory_hash"],
        "source_hash": report["source_hash"],
        "model_hash": report["model_hash"],
        "asset_hash": report["asset_hash"],
        "gain_hash": report["gain_hash"],
        "joint_map_hash": report["joint_map_hash"],
        "clean_foot_only": samples.clean_foot_only,
        "first_contact_microstep": samples.first_contact_microstep,
        "first_foot_side": side,
        "precontact_frame_count": len(samples.frame_indices),
        "last_precontact_feature": samples.features[-1].tolist(),
        "minimum_pelvis_height_m": report["min_pelvis_height_m"],
    }


def import_first_touch_curriculum(folders: tuple[Path, ...]) -> dict[str, Any]:
    if not folders or len(set(folders)) != len(folders):
        raise ValueError("nonempty unique Isaac episode directories required")
    episodes = tuple(_episode(folder) for folder in folders)
    for key in ("course", "report_hash", "trajectory_hash"):
        if len({json.dumps(row[key], sort_keys=True) for row in episodes}) != len(episodes):
            raise ValueError(f"duplicate {key} cannot inflate independent physics")
    for key in ("source_hash", "model_hash", "asset_hash", "gain_hash", "joint_map_hash"):
        if len({row[key] for row in episodes}) != 1:
            raise ValueError(f"heterogeneous Isaac {key} requires a separate curriculum")
    positives = sum(bool(row["clean_foot_only"]) for row in episodes)
    sides = sorted({row["first_foot_side"] for row in episodes if row["clean_foot_only"]})
    report = {
        "schema": "rosclaw_soccer.rsi.isaac_first_touch_curriculum_import.v1",
        "partition": "CONSUMED_DEV",
        "episodes": episodes,
        "independent_physical_episode_count": len(episodes),
        "clean_foot_only_episode_count": positives,
        "positive_foot_sides": sides,
        "imitation_training_authorized": bool(positives >= 4 and sides == ["left", "right"]),
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = import_first_touch_curriculum(tuple(args.folders))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "independent_physical_episode_count",
                    "clean_foot_only_episode_count",
                    "positive_foot_sides",
                    "imitation_training_authorized",
                    "report_hash",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
