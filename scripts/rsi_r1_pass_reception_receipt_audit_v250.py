"""Distinguish physical pass launches from completed, clean receptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.active_team_probe import validate_probe


def audit(holdout: Path, original: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external receipt audit evidence required")
    parent = json.loads((holdout / "report.json").read_text())
    commitment = parent.pop("report_hash")
    if commitment != hash_json(parent) or parent["schema"] != (
        "rosclaw_soccer.rsi.four_vs_four_safety_holdout_v249.v1"
    ):
        raise ValueError("sealed four-course SIM_ONLY source required")
    rows = []
    courses = [
        (
            holdout / f"{'blue' if course['blue_kickoff'] else 'red'}-{course['offset_m']:+.2f}",
            course,
        )
        for course in parent["courses"]
    ]
    if len(courses) != 4 or {(c["blue_kickoff"], c["offset_m"]) for _, c in courses} != {
        (False, -0.08),
        (False, 0.08),
        (True, -0.08),
        (True, 0.08),
    }:
        raise ValueError("predeclared bilateral holdout courses differ")
    courses.append((original / "probe.json", None))
    for path, parent_course in courses:
        report = validate_probe(path if parent_course is None else path / "probe.json")
        if parent_course is not None and report["report_hash"] != parent_course["probe_hash"]:
            raise ValueError("holdout physical report differs from its parent commitment")
        events = [e for e in report["assessment"]["events"] if e["skill"] == "pass"]
        diagnoses = report["causal_pass_feedback"]
        confirmed = [d for d in diagnoses if d["physical_receive_confirmed"]]
        for diagnosis in confirmed:
            if not any(
                event["agent_id"] == diagnosis["sender"]
                and event["target_agent_id"] == diagnosis["receiver"]
                and abs(event["time_sec"] - diagnosis["foot_contact_time_sec"]) < 1e-9
                for event in events
            ):
                raise ValueError("confirmed reception lacks a matching physical pass launch")
        clean = report["assessment"]["gates"]["foot_only_ball_control"]
        safe = report["results"][0]["safe"]
        rows.append(
            {
                "course": "original" if parent_course is None else path.name,
                "probe_hash": report["report_hash"],
                "pass_launches": len(events),
                "confirmed_receptions": len(confirmed),
                "qualified_clean_receptions": len(confirmed) if clean and safe else 0,
                "safe": safe,
                "foot_only_ball_control": clean,
                "termination_reason": report["termination"]["reason"],
            }
        )
    holdout_rows = rows[:4]
    result = {
        "schema": "rosclaw_soccer.rsi.r1_pass_reception_receipt_audit_v250.v1",
        "source_hash": hash_bytes(Path(__file__).read_bytes()),
        "parent_report_hash": commitment,
        "rows": rows,
        "holdout_pass_launches": sum(r["pass_launches"] for r in holdout_rows),
        "holdout_confirmed_receptions": sum(r["confirmed_receptions"] for r in holdout_rows),
        "holdout_qualified_clean_receptions": sum(
            r["qualified_clean_receptions"] for r in holdout_rows
        ),
        "status": "REJECTED_NO_CLEAN_HOLDOUT_RECEPTION",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "READ_ONLY_SIM_ONLY",
    }
    if result["holdout_confirmed_receptions"] != 0:
        raise ValueError("holdout rejection status requires zero confirmed receptions")
    result["report_hash"] = hash_json(result)
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.holdout, args.original, args.output)
    print(
        json.dumps(
            {
                "holdout_pass_launches": result["holdout_pass_launches"],
                "holdout_confirmed_receptions": result["holdout_confirmed_receptions"],
                "report_hash": result["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
