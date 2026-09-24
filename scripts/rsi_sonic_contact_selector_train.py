"""Fit an interpretable SONIC lateral decision stump from pinned discovery rolls.

It prioritizes real foot-first whole-ball goals and penalizes needless sideways
motion. The output is only a SIM_ONLY navigation course selector; not motor RL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.sonic_contact_selector import SCHEMA
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _read(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    claimed = report.pop("report_hash", None)
    if claimed != hash_json(report):
        raise ValueError("discovery report hash mismatch")
    if hash_bytes(path.with_name("trajectory.npz").read_bytes()) != report["trajectory_hash"]:
        raise ValueError("discovery trajectory hash mismatch")
    return report


def fit(grid: Path, supplemental: tuple[Path, ...], output: Path) -> dict[str, Any]:
    if output.exists() or not (grid / "manifest.json").is_file():
        raise ValueError("new selector output and complete physical grid required")
    manifest: dict[str, Any] = json.loads((grid / "manifest.json").read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_hash", None)
    if claimed != hash_json(manifest) or manifest.get("course_count") != 36:
        raise ValueError("physical grid manifest hash or course count failed")
    reports = [_read(path) for path in sorted(grid.glob("*/report.json"))]
    if len(reports) != 36:
        raise ValueError("physical grid incomplete")
    if sorted(hash_json(r) for r in reports) != manifest["report_hashes"]:
        raise ValueError("physical grid reports differ from pinned manifest")
    if (
        any(report["source_hash"] != manifest["probe_hash"] for report in reports)
        or len({report["execution_id"] for report in reports}) != 36
        or len({report["physics_hash"] for report in reports}) != 1
        or len({report["model_hash"] for report in reports}) != 1
    ):
        raise ValueError("physical grid runner, execution or model provenance differs")
    by_position: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for report in reports:
        if report["partition"] != "DISCOVERY" or report["promotion_authorized"]:
            raise ValueError("selector training requires discovery-only physical evidence")
        x, y = report["ball_initial_xy_m"]
        by_position.setdefault((float(x), float(y)), []).append(report)
    if len(by_position) != 9:
        raise ValueError("expected nine declared ball-position courses")
    labels: list[tuple[float, float, float]] = []
    for (x, y), group in sorted(by_position.items()):
        candidates = [
            report
            for report in group
            if report["left_contact_residual_rad"] == [-0.15, 0.15, 0.0]
            and report["first_robot_ball_contact_is_foot"]
            and report["whole_ball_goal_crossed"]
            and report["minimum_pelvis_height_m"] >= 0.55
        ]
        if not candidates:
            raise ValueError("training position has no physically eligible action")
        chosen = max(
            candidates,
            key=lambda report: (
                0.1 * report["peak_ball_speed_mps"] - 2.0 * abs(report["run_lateral_mps"])
            ),
        )
        labels.append((x, y, float(chosen["run_lateral_mps"])))
    supplement_hashes = []
    for path in supplemental:
        report = _read(path / "report.json")
        if (
            report["partition"] != "DISCOVERY"
            or report["left_contact_residual_rad"] != [-0.15, 0.15, 0.0]
            or not report["first_robot_ball_contact_is_foot"]
            or not report["whole_ball_goal_crossed"]
            or report["run_lateral_mps"] not in (-0.1, 0.0)
        ):
            raise ValueError("supplemental label lacks a valid foot-first goal")
        x, y = report["ball_initial_xy_m"]
        labels.append((float(x), float(y), float(report["run_lateral_mps"])))
        supplement_hashes.append(hash_json(report))
    if any(lateral not in (-0.1, 0.0) for _, _, lateral in labels):
        raise ValueError("unsupported physical teacher lateral action")
    left = max(y for _, y, lateral in labels if lateral == -0.1)
    right = min(y for _, y, lateral in labels if lateral == 0.0)
    if not left < right:
        raise ValueError("one-dimensional selector cannot separate physical labels")
    model = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "trained_actor": False,
        "learner": "one_split_foot_goal_teacher_distillation",
        "threshold_y_m": float((left + right) / 2.0),
        "low_y_lateral_mps": -0.1,
        "high_y_lateral_mps": 0.0,
        "contact_residual_rad": [-0.15, 0.15, 0.0],
        "training_grid_hash": claimed,
        "supplemental_report_hashes": sorted(supplement_hashes),
        "training_position_count": len(labels),
        "training_foot_goal_count": sum(
            (-0.1 if y < (left + right) / 2.0 else 0.0) == lateral for _, y, lateral in labels
        ),
    }
    model["model_hash"] = hash_json(model)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(model, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--supplemental", nargs="*", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(fit(args.grid, tuple(args.supplemental), args.output), sort_keys=True))


if __name__ == "__main__":
    main()
