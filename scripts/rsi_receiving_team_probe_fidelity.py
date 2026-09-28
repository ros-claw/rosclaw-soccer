"""Seal the off-trajectory small-perturbation fidelity gate; never promote motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _sealed(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    commitment = value.pop("report_hash")
    if (
        commitment != hash_json(value)
        or value["schema"] != schema
        or value["activation_ceiling"] != "SIM_ONLY"
        or value["promotion_authorized"] is not False
    ):
        raise ValueError("sealed unpromoted SIM_ONLY probe evidence required")
    return value, commitment


def assess(*, short: Path, full: Path, output_dir: Path) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.parents[1]):
        raise ValueError("new external SIM_ONLY fidelity directory required")
    proxy, proxy_hash = _sealed(
        short / "report.json", "rosclaw_soccer.rsi.receiving_team_short_probe.v1"
    )
    world, world_hash = _sealed(
        full / "report.json", "rosclaw_soccer.rsi.receiving_team_full_probe.v1"
    )
    proxy_path = short / "short-probe.npz"
    world_path = full / "full-probe.npz"
    if (
        proxy["focal_probe_nm"] != world["focal_probe_nm"]
        or proxy["focal_probe_frames"] != world["focal_probe_frames"]
        or proxy["focal_probe_nm"] != 0.75
        or proxy["focal_probe_frames"] != [55, 75]
        or proxy["capture_report_hash"] != world["student_tape_report_hash"]
        or proxy["student_model_hash"] is not None
        or world["student_model_hash"] is None
        or proxy["trajectory_hash"] != hash_bytes(proxy_path.read_bytes())
        or world["full_trace_hash"] != hash_bytes(world_path.read_bytes())
    ):
        raise ValueError("predeclared matched frozen-student probe evidence required")
    with np.load(world_path, allow_pickle=False) as trajectory:
        code = 6  # sorted eight-player fixture: red.finisher
        foot = np.flatnonzero(
            (trajectory["ball_contact_agent_code"] == code)
            & (trajectory["ball_contact_foot_code"] > 0)
        )
        nonfoot = np.flatnonzero(trajectory["ball_nonfoot_contact_agent_code"] == code)
    full_first = None if len(foot) == 0 else int(foot[0])
    speed_error = abs(
        float(proxy["replay_frame86_ball_speed_mps"])
        - float(world["measurement"]["frame86_ball_speed_mps"])
    )
    distance_error = abs(
        float(proxy["replay_frame86_ball_pelvis_distance_m"])
        - float(world["measurement"]["frame86_ball_pelvis_distance_m"])
    )
    passed = bool(
        full_first is not None
        and proxy["replay_first_foot_frame"] is not None
        and abs(int(proxy["replay_first_foot_frame"]) - full_first) <= 1
        and speed_error <= 0.05
        and distance_error <= 0.05
        and proxy["all_robot_bodies_safe"] is True
        and proxy["own_nonfoot_seen"] is False
        and len(nonfoot) == 0
        and world["result"]["safe"] is True
        and all(row["safe"] for row in world["result"]["qualities"])
    )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("probe fidelity source changed during assessment")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_team_probe_fidelity.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "short_report_hash": proxy_hash,
        "full_report_hash": world_hash,
        "focal_probe_nm": 0.75,
        "focal_probe_frames": [55, 75],
        "proxy_first_foot_frame": proxy["replay_first_foot_frame"],
        "full_first_foot_frame": full_first,
        "frame86_speed_error_mps": speed_error,
        "frame86_distance_error_m": distance_error,
        "full_focal_nonfoot_frames": nonfoot.tolist(),
        "local_training_proxy_fidelity_passed": passed,
        "controlled_reception": world["measurement"]["authoritative_window"][
            "controlled_reception"
        ],
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    report = assess(**vars(parser.parse_args()))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
