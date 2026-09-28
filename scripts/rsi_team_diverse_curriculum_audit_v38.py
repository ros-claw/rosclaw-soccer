"""Seal 512 causal paired 3v3 physical contexts and their arm outcome table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_contextual_action_dataset import entry_features
from scripts.rsi_team_diverse_curriculum_collect import training_courses
from scripts.rsi_team_foot_velocity_fresh_exam import score_parent
from scripts.rsi_team_intercept_navigation_search import _score


def _sealed_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    commitment = report.get("report_hash")
    body = {key: value for key, value in report.items() if key != "report_hash"}
    if commitment != hash_json(body):
        raise ValueError("invalid physical report commitment")
    return report


def _verify_episode(
    folder: Path,
    *,
    scenario_hash: str,
    row: dict[str, Any],
    parent: bool,
) -> None:
    episode = _sealed_report(folder / "report.json")
    if (
        episode["scenario_hash"] != scenario_hash
        or episode["trace_hash"] != hash_bytes((folder / "trajectory.npz").read_bytes())
        or episode["action_trace_hash"] != hash_bytes((folder / "taskspace_trace.npz").read_bytes())
    ):
        raise ValueError("unsealed episode physical or action trace")
    score = (
        score_parent(episode, folder / "trajectory.npz")
        if parent
        else _score(episode, folder / "trajectory.npz")
    )
    for key in ("safe", "foot_contact_frames", "useful_pass"):
        if score[key] != row[key]:
            raise ValueError("stored outcome differs from measured physical score")
    expected_speed = score["outgoing_ball_vx_mps"]
    stored_speed = row["outgoing_ball_vx_mps"]
    if (expected_speed is None) != (stored_speed is None) or (
        expected_speed is not None
        and stored_speed is not None
        and abs(expected_speed - stored_speed) > 1e-10
    ):
        raise ValueError("stored outgoing ball speed differs from physical trace")
    entry = entry_features(folder / "taskspace_trace.npz", 30)
    if entry["hash"] != row["entry_hash"]:
        raise ValueError("stored pre-decision feature differs from action trace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("physical audit output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_diverse_curriculum_audit_protocol_v38"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("batch_count") != 16
        or protocol.get("scenes_per_batch") != 32
        or protocol.get("training_batches") != list(range(12))
        or protocol.get("internal_validation_batches") != list(range(12, 16))
        or len(protocol.get("arms", ())) != 7
        or protocol.get("oracle_gate_diagnostic_only")
        != {"minimum_safe_contacts": 300, "minimum_useful_passes": 180}
    ):
        raise ValueError("invalid committed 512-context audit protocol")
    collection_path = Path(protocol["collection_protocol_path"])
    if hash_bytes(collection_path.read_bytes()) != protocol["collection_protocol_hash"]:
        raise ValueError("physical collection curriculum changed")
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    root = Path(protocol["result_root"])
    names = protocol["arms"]
    reports: dict[tuple[int, str], dict[str, Any]] = {}
    for batch in range(16):
        for name in names:
            report = _sealed_report(root / f"b{batch:02d}" / name / "report.json")
            if (
                report["batch_index"] != batch
                or report["arm"]["name"] != name
                or report["protocol_hash"] != protocol["collection_protocol_hash"]
                or len(report["rows"]) != 32
            ):
                raise ValueError("mixed batch, arm or physical curriculum")
            reports[batch, name] = report
    if (
        len({json.dumps(r["source_hashes"], sort_keys=True) for r in reports.values()}) != 1
        or len({r["asset_body_hash"] for r in reports.values()}) != 1
    ):
        raise ValueError("mixed physical source or G1 asset body")
    arm_names = names[1:]
    features: list[list[float]] = []
    scenarios: list[str] = []
    entry_hashes: list[str] = []
    physical_inputs: list[list[float]] = []
    safe = np.zeros((512, 6), dtype=np.bool_)
    contact = np.zeros_like(safe)
    useful = np.zeros_like(safe)
    outgoing = np.full((512, 6), np.nan, dtype=np.float64)
    parent_safe = np.zeros(512, dtype=np.bool_)
    incomplete = 0
    for batch in range(16):
        courses = training_courses(collection, batch)
        if len(courses) != 32:
            raise ValueError("batch course size changed")
        for index, course in enumerate(courses):
            global_index = batch * 32 + index
            rows = {name: reports[batch, name]["rows"][index] for name in names}
            if (
                len({row["scenario_hash"] for row in rows.values()}) != 1
                or rows["parent"]["status"] != "COMPLETE"
                or course.scenario_hash != rows["parent"]["scenario_hash"]
            ):
                raise ValueError("unpaired scene or missing parent pre-decision observation")
            completed_entries = {
                row["entry_hash"] for row in rows.values() if row["status"] == "COMPLETE"
            }
            if completed_entries != {rows["parent"]["entry_hash"]}:
                raise ValueError("intervention changed pre-decision physical state")
            for name in names:
                row = rows[name]
                if row["status"] != "COMPLETE":
                    if row["safe"] or row["useful_pass"] or row["foot_contact_frames"]:
                        raise ValueError("incomplete episode credited with success")
                    incomplete += 1
                    continue
                mode = "parent" if name == "parent" else "candidate"
                folder = root / f"b{batch:02d}" / name / f"t{index:03d}" / mode
                _verify_episode(
                    folder,
                    scenario_hash=course.scenario_hash,
                    row=row,
                    parent=name == "parent",
                )
            parent_folder = root / f"b{batch:02d}/parent/t{index:03d}/parent"
            entry = entry_features(parent_folder / "taskspace_trace.npz", 30)
            raw = np.asarray(entry["values"], dtype=np.float64)
            if raw.shape != (24,) or entry["hash"] != rows["parent"]["entry_hash"]:
                raise ValueError("invalid measured causal input")
            features.append(raw.tolist())
            scenarios.append(course.scenario_hash)
            entry_hashes.append(entry["hash"])
            physical_inputs.append(
                [
                    course.ball_initial_position_m[0],
                    course.ball_initial_position_m[1],
                    course.ball_initial_velocity_mps[0],
                ]
            )
            parent_safe[global_index] = bool(rows["parent"]["safe"])
            for arm_index, name in enumerate(arm_names):
                row = rows[name]
                safe[global_index, arm_index] = bool(row["safe"])
                contact[global_index, arm_index] = bool(row["safe"] and row["foot_contact_frames"])
                useful[global_index, arm_index] = bool(row["useful_pass"])
                if row.get("outgoing_ball_vx_mps") is not None:
                    outgoing[global_index, arm_index] = row["outgoing_ball_vx_mps"]
                if useful[global_index, arm_index] and not contact[global_index, arm_index]:
                    raise ValueError("effective pass without measured safe foot contact")
    if len(set(scenarios)) != 512 or len({tuple(item) for item in physical_inputs}) != 512:
        raise ValueError("seed-only or duplicate physical scenes")
    oracle_contact = int(contact.any(axis=1).sum())
    oracle_useful = int(useful.any(axis=1).sum())
    gate = bool(
        oracle_contact >= protocol["oracle_gate_diagnostic_only"]["minimum_safe_contacts"]
        and oracle_useful >= protocol["oracle_gate_diagnostic_only"]["minimum_useful_passes"]
    )
    args.output_dir.mkdir(parents=True)
    dataset_path = args.output_dir / "paired_outcomes.npz"
    np.savez_compressed(
        dataset_path,
        raw_entry=np.asarray(features, dtype="<f8"),
        physical_ball_input=np.asarray(physical_inputs, dtype="<f8"),
        safe=safe,
        safe_contact=contact,
        useful_pass=useful,
        outgoing_ball_vx_mps=outgoing,
        parent_safe=parent_safe,
        arm_names=np.asarray(arm_names),
        scenario_hashes=np.asarray(scenarios),
        entry_hashes=np.asarray(entry_hashes),
    )
    result = {
        "schema": "rsi_team_diverse_curriculum_audit_report_v38",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "collection_protocol_hash": protocol["collection_protocol_hash"],
        "physical_source_hashes": reports[0, "parent"]["source_hashes"],
        "body_hash": reports[0, "parent"]["asset_body_hash"],
        "arm_report_hashes": {
            f"b{batch:02d}/{name}": reports[batch, name]["report_hash"]
            for batch in range(16)
            for name in names
        },
        "dataset_hash": hash_bytes(dataset_path.read_bytes()),
        "physical_scene_count": 512,
        "complete_episode_count": 3584 - incomplete,
        "incomplete_episode_count": incomplete,
        "training_scene_count": 384,
        "internal_validation_scene_count": 128,
        "parent_safe_count": int(parent_safe.sum()),
        "arm_names": arm_names,
        "per_arm_safe_count": safe.sum(axis=0).astype(int).tolist(),
        "per_arm_safe_contact_count": contact.sum(axis=0).astype(int).tolist(),
        "per_arm_useful_count": useful.sum(axis=0).astype(int).tolist(),
        "oracle_safe_contact_diagnostic_only": oracle_contact,
        "oracle_useful_pass_diagnostic_only": oracle_useful,
        "oracle_gate_passed_for_training_only": gate,
        "fresh_online_exam": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_512_AUDIT="
        + json.dumps(
            {
                key: result[key]
                for key in (
                    "report_hash",
                    "dataset_hash",
                    "complete_episode_count",
                    "incomplete_episode_count",
                    "per_arm_useful_count",
                    "oracle_safe_contact_diagnostic_only",
                    "oracle_useful_pass_diagnostic_only",
                    "oracle_gate_passed_for_training_only",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
