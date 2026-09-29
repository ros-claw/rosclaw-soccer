"""Seal and re-score a one- or two-batch six-G1 paired physical curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_contextual_action_dataset import entry_features
from scripts.rsi_team_diverse_curriculum_audit_v38 import _sealed_report, _verify_episode
from scripts.rsi_team_diverse_curriculum_collect import training_courses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("physical paired audit output already exists")
    protocol = json.loads(args.protocol.read_text())
    names = [arm["name"] for arm in protocol["arms"]]
    batches = protocol["curriculum"]["batch_count"]
    if (
        protocol.get("schema") != "rsi_team_diverse_curriculum_protocol_v38"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or type(batches) is not int
        or not 1 <= batches <= 2
        or protocol["curriculum"].get("training_scene_count") != 32
        or len(names) != 7
        or names[0:2] != ["parent", "baseline"]
    ):
        raise ValueError("invalid bounded physical audit curriculum")
    protocol_hash = hash_bytes(args.protocol.read_bytes())
    reports = {
        (batch, name): _sealed_report(args.result_root / f"b{batch:02d}" / name / "report.json")
        for batch in range(batches)
        for name in names
    }
    if any(
        report["protocol_hash"] != protocol_hash
        or report["batch_index"] != batch
        or report["arm"]["name"] != name
        or len(report["rows"]) != 32
        for (batch, name), report in reports.items()
    ):
        raise ValueError("mixed physical arm reports")
    if (
        len({json.dumps(row["source_hashes"], sort_keys=True) for row in reports.values()}) != 1
        or len({row["asset_body_hash"] for row in reports.values()}) != 1
    ):
        raise ValueError("mixed physical source or G1 body")
    metrics: dict[str, dict[str, int]] = {
        name: {"complete": 0, "incomplete": 0, "safe": 0, "safe_contact": 0, "useful": 0}
        for name in names
    }
    count = batches * 32
    raw = np.zeros((count, 24), dtype=np.float64)
    safe = np.zeros((count, 7), dtype=np.bool_)
    contact = np.zeros_like(safe)
    useful = np.zeros_like(safe)
    scenarios: list[str] = []
    entry_hashes: list[str] = []
    physical_inputs: list[tuple[float, float, float]] = []
    for batch in range(batches):
        for local, course in enumerate(training_courses(protocol, batch)):
            index = batch * 32 + local
            rows = {name: reports[batch, name]["rows"][local] for name in names}
            parent = rows["parent"]
            if (
                parent["status"] != "COMPLETE"
                or parent["scenario_hash"] != course.scenario_hash
                or not parent["entry_hash"]
            ):
                raise ValueError("missing causal Parent predecision observation")
            if {row["entry_hash"] for row in rows.values() if row["status"] == "COMPLETE"} != {
                parent["entry_hash"]
            }:
                raise ValueError("paired intervention altered predecision physical state")
            parent_folder = args.result_root / f"b{batch:02d}/parent/t{local:03d}/parent"
            entry = entry_features(parent_folder / "taskspace_trace.npz", 30)
            if entry["hash"] != parent["entry_hash"]:
                raise ValueError("parent measured causal input changed")
            raw[index] = np.asarray(entry["values"], dtype=np.float64)
            scenarios.append(course.scenario_hash)
            entry_hashes.append(entry["hash"])
            physical_inputs.append(
                (
                    course.ball_initial_position_m[0],
                    course.ball_initial_position_m[1],
                    course.ball_initial_velocity_mps[0],
                )
            )
            for arm_index, name in enumerate(names):
                row = rows[name]
                if row["scenario_hash"] != course.scenario_hash:
                    raise ValueError("unpaired physical scene")
                tally = metrics[name]
                if row["status"] != "COMPLETE":
                    if row["safe"] or row["useful_pass"] or row["foot_contact_frames"]:
                        raise ValueError("incomplete scene credited as safe or useful")
                    tally["incomplete"] += 1
                    continue
                folder = (
                    args.result_root
                    / f"b{batch:02d}"
                    / name
                    / f"t{local:03d}"
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
                safe[index, arm_index] = bool(row["safe"])
                contact[index, arm_index] = bool(row["safe"] and row["foot_contact_frames"])
                useful[index, arm_index] = bool(row["useful_pass"])
    if (
        len(set(scenarios)) != count
        or len(set(physical_inputs)) != count
        or len(set(entry_hashes)) != count
    ):
        raise ValueError("duplicate physical scene or causal state")
    args.output_dir.mkdir(parents=True)
    dataset_path = args.output_dir / "paired_outcomes.npz"
    np.savez_compressed(
        dataset_path,
        raw_entry=raw,
        physical_ball_input=np.asarray(physical_inputs, dtype=np.float64),
        safe=safe,
        safe_contact=contact,
        useful_pass=useful,
        arm_names=np.asarray(names),
        scenario_hashes=np.asarray(scenarios),
        entry_hashes=np.asarray(entry_hashes),
    )
    report: dict[str, Any] = {
        "schema": "rsi_team_paired_physics_audit_report_v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": protocol_hash,
        "body_hash": reports[0, "parent"]["asset_body_hash"],
        "physical_source_hashes": reports[0, "parent"]["source_hashes"],
        "arm_report_hashes": {
            f"b{batch:02d}/{name}": reports[batch, name]["report_hash"]
            for batch in range(batches)
            for name in names
        },
        "scene_count": count,
        "episode_count": count * 7,
        "metrics": metrics,
        "dataset_hash": hash_bytes(dataset_path.read_bytes()),
        "fresh_online_exam": False,
    }
    report["report_hash"] = hash_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_PAIRED_AUDIT="
        + json.dumps(
            {
                "report_hash": report["report_hash"],
                "dataset_hash": report["dataset_hash"],
                "metrics": metrics,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
