"""Seal the aborted v36 three-way exam without promoting a partial score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_foot_velocity_fresh_exam import score_parent
from scripts.rsi_team_intercept_navigation_search import _score


def _episode(folder: Path) -> dict[str, Any]:
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if (
        hash_json({key: value for key, value in report.items() if key != "report_hash"})
        != report["report_hash"]
        or hash_bytes((folder / "trajectory.npz").read_bytes()) != report["trace_hash"]
        or hash_bytes((folder / "taskspace_trace.npz").read_bytes()) != report["action_trace_hash"]
    ):
        raise ValueError("unsealed physical episode")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("partial audit output already exists")
    rows = []
    for index in range(21):
        folder = args.input_dir / f"f{index:03d}"
        parent_folder = folder / "parent"
        parent_report = _episode(parent_folder)
        parent = score_parent(parent_report, parent_folder / "trajectory.npz")
        scores = {}
        for name in ("zero", "through"):
            path = folder / name / "candidate"
            report = _episode(path)
            if report["scenario_hash"] != parent_report["scenario_hash"]:
                raise ValueError("mixed scene identity")
            scores[name] = _score(report, path / "trajectory.npz")
        rows.append(
            {
                "scene": f"f{index:03d}",
                "scenario_hash": parent_report["scenario_hash"],
                "parent": parent,
                **scores,
            }
        )
    if (args.input_dir / "f021/parent/report.json").exists():
        raise ValueError("abort marker no longer matches incomplete Parent")
    totals = {
        name: {
            "unsafe": sum(not row[name]["safe"] for row in rows),
            "safe_contact": sum(
                bool(row[name]["safe"] and row[name]["foot_contact_frames"]) for row in rows
            ),
            "useful": sum(row[name]["useful_pass"] for row in rows),
        }
        for name in ("parent", "zero", "through")
    }
    report = {
        "schema": "rsi_team_strike_partial_audit_v36",
        "activation_ceiling": "SIM_ONLY",
        "completed_paired_scenes": 21,
        "planned_scenes": 24,
        "abort_scene": "f021",
        "abort_reason": "parent_incomplete_unhandled_by_collector",
        "rows": rows,
        "totals_partial_diagnostic_only": totals,
        "fresh_exam_complete": False,
        "gate_passed": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_STRIKE_PARTIAL="
        + json.dumps(
            {key: report[key] for key in ("report_hash", "totals_partial_diagnostic_only")},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
