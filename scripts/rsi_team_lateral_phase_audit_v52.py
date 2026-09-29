"""Re-score every v52 six-G1 hard-lateral action and seal causal comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_diverse_curriculum_audit_v38 import _sealed_report, _verify_episode
from scripts.rsi_team_diverse_curriculum_collect import training_courses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("v52 physical audit already exists")
    protocol = json.loads(args.protocol.read_text())
    names = [item["name"] for item in protocol["arms"]]
    if (
        protocol.get("development_gate")
        != {
            "minimum_useful_gain_over_baseline": 3,
            "maximum_unsafe_excess_over_baseline": 0,
            "require_parent_complete": True,
        }
        or len(names) != 7
        or names[0:2] != ["parent", "baseline"]
        or len(training_courses(protocol)) != 32
    ):
        raise ValueError("invalid predeclared hard-lateral physical gate")
    commitment = hash_bytes(args.protocol.read_bytes())
    batches = {name: _sealed_report(args.result_root / name / "report.json") for name in names}
    if any(
        report["protocol_hash"] != commitment
        or report["arm"]["name"] != name
        or report["batch_index"] != 0
        or len(report["rows"]) != 32
        for name, report in batches.items()
    ):
        raise ValueError("mixed or incomplete physical arm reports")
    if (
        len({json.dumps(report["source_hashes"], sort_keys=True) for report in batches.values()})
        != 1
        or len({report["asset_body_hash"] for report in batches.values()}) != 1
    ):
        raise ValueError("mixed source or G1 body")
    courses = training_courses(protocol)
    metrics: dict[str, dict[str, int]] = {
        name: {"complete": 0, "incomplete": 0, "safe": 0, "safe_contact": 0, "useful": 0}
        for name in names
    }
    entry_hashes: list[str] = []
    for index, course in enumerate(courses):
        rows = {name: batches[name]["rows"][index] for name in names}
        parent = rows["parent"]
        if (
            parent["status"] != "COMPLETE"
            or parent["scenario_hash"] != course.scenario_hash
            or not parent["entry_hash"]
        ):
            raise ValueError("missing parent predecision physical state")
        complete_entries = {
            row["entry_hash"] for row in rows.values() if row["status"] == "COMPLETE"
        }
        if complete_entries != {parent["entry_hash"]}:
            raise ValueError("arm changed causal predecision state")
        entry_hashes.append(parent["entry_hash"])
        for name, row in rows.items():
            if row["scenario_hash"] != course.scenario_hash:
                raise ValueError("unpaired hard-lateral physical scene")
            tally = metrics[name]
            if row["status"] != "COMPLETE":
                if row["safe"] or row["useful_pass"] or row["foot_contact_frames"]:
                    raise ValueError("incomplete episode credited as successful")
                tally["incomplete"] += 1
                continue
            folder = (
                args.result_root
                / name
                / f"t{index:03d}"
                / ("parent" if name == "parent" else "candidate")
            )
            _verify_episode(
                folder,
                scenario_hash=course.scenario_hash,
                row=row,
                parent=name == "parent",
            )
            tally["complete"] += 1
            tally["safe"] += int(row["safe"])
            tally["safe_contact"] += int(bool(row["safe"] and row["foot_contact_frames"]))
            tally["useful"] += int(row["useful_pass"])
    if len(set(entry_hashes)) != 32:
        raise ValueError("duplicate causal physical entry states")
    baseline = metrics["baseline"]
    comparisons: list[dict[str, Any]] = []
    for name in names[2:]:
        candidate = metrics[name]
        useful_gain = candidate["useful"] - baseline["useful"]
        unsafe_excess = (32 - candidate["safe"]) - (32 - baseline["safe"])
        comparisons.append(
            {
                "arm": name,
                "useful_gain": useful_gain,
                "unsafe_excess": unsafe_excess,
                "development_gate_passed": bool(useful_gain >= 3 and unsafe_excess <= 0),
            }
        )
    result = {
        "schema": "rsi_team_lateral_phase_audit_report_v52",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": commitment,
        "body_hash": batches["parent"]["asset_body_hash"],
        "physical_source_hashes": batches["parent"]["source_hashes"],
        "arm_report_hashes": {name: batches[name]["report_hash"] for name in names},
        "scene_count": 32,
        "episode_count": 224,
        "metrics": metrics,
        "candidate_comparisons": comparisons,
        "development_gate_passed": any(row["development_gate_passed"] for row in comparisons),
        "fresh_online_exam": False,
    }
    result["report_hash"] = hash_json(result)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_V52_AUDIT="
        + json.dumps(
            {
                "report_hash": result["report_hash"],
                "metrics": metrics,
                "candidate_comparisons": comparisons,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
