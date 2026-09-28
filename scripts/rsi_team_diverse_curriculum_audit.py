"""Seal a causal full-information training table from paired 3v3 physical scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.team_contextual_nav_policy import measured_entry_features
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_contextual_action_dataset import entry_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("audit output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    version = protocol.get("schema")
    scene_count = 64 if version == "rsi_team_diverse_curriculum_audit_protocol_v29" else 128
    expected_gate = (
        {"minimum_safe_contacts": 40, "minimum_useful_passes": 24}
        if scene_count == 64
        else {"minimum_safe_contacts": 70, "minimum_useful_passes": 35}
    )
    if (
        version
        not in {
            "rsi_team_diverse_curriculum_audit_protocol_v29",
            "rsi_team_diverse_curriculum_audit_protocol_v31",
        }
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("training_indices") != list(range(48 if scene_count == 64 else 128))
        or protocol.get("internal_validation_indices")
        != (list(range(48, 64)) if scene_count == 64 else [])
        or len(protocol.get("arm_report_hashes", {})) != (8 if scene_count == 64 else 7)
        or protocol.get("oracle_gate") != expected_gate
    ):
        raise ValueError("invalid committed physical curriculum audit")
    root = Path(protocol["arm_result_root"])
    reports = {}
    for name, expected_hash in protocol["arm_report_hashes"].items():
        report = json.loads((root / name / "report.json").read_text(encoding="utf-8"))
        if (
            report.get("report_hash") != expected_hash
            or hash_json({key: value for key, value in report.items() if key != "report_hash"})
            != expected_hash
            or len(report["rows"]) != scene_count
        ):
            raise ValueError("unsealed physical curriculum arm")
        reports[name] = report
    if (
        len({str(report["source_hashes"]) for report in reports.values()}) != 1
        or len({report["asset_body_hash"] for report in reports.values()}) != 1
        or len({report["protocol_hash"] for report in reports.values()}) != 1
    ):
        raise ValueError("mixed physical source, body or course protocol")
    names = sorted(name for name in reports if name != "parent")
    features = []
    safety = np.zeros((scene_count, len(names)), dtype=np.bool_)
    contact = np.zeros_like(safety)
    useful = np.zeros_like(safety)
    entry_hashes = []
    scenarios = []
    per_scene = []
    for index in range(scene_count):
        rows = {name: report["rows"][index] for name, report in reports.items()}
        if len({row["scenario_hash"] for row in rows.values()}) != 1:
            raise ValueError("unpaired physical scenario")
        complete_hashes = {
            row["entry_hash"] for row in rows.values() if row["status"] == "COMPLETE"
        }
        if len(complete_hashes) != 1:
            raise ValueError("intervention changed pre-decision state")
        parent_trace = root / "parent" / f"t{index:03d}/parent/taskspace_trace.npz"
        entry = entry_features(parent_trace, 30)
        if entry["hash"] != rows["parent"]["entry_hash"]:
            raise ValueError("training feature bytes differ from sealed parent")
        raw = np.asarray(entry["values"], dtype=np.float64)
        ball = tuple(float(value) for value in raw[6:9])
        feet = tuple(tuple(float(value) for value in raw[start : start + 3]) for start in (12, 15))
        compact = measured_entry_features(
            ball=ball,
            ball_vx=float(raw[9]),
            feet=feet,  # type: ignore[arg-type]
        )
        # The local root values are directly present in the read-only
        # NavigationObservation at decision time; no future collision signal.
        features.append((*compact, raw[0], raw[1], raw[3], raw[4]))
        entry_hashes.append(entry["hash"])
        scenarios.append(rows["parent"]["scenario_hash"])
        for column, name in enumerate(names):
            row = rows[name]
            if row["status"] == "COMPLETE":
                folder = (
                    root / name / f"t{index:03d}" / ("parent" if name == "parent" else "candidate")
                )
                episode = json.loads((folder / "report.json").read_text(encoding="utf-8"))
                if (
                    episode["report_hash"]
                    != hash_json(
                        {key: value for key, value in episode.items() if key != "report_hash"}
                    )
                    or episode["scenario_hash"] != row["scenario_hash"]
                    or episode["trace_hash"] != hash_bytes((folder / "trajectory.npz").read_bytes())
                    or episode["action_trace_hash"]
                    != hash_bytes((folder / "taskspace_trace.npz").read_bytes())
                ):
                    raise ValueError("unsealed paired physical episode")
            safety[index, column] = bool(row["safe"])
            contact[index, column] = bool(row["safe"] and row["foot_contact_frames"])
            useful[index, column] = bool(row["useful_pass"])
        per_scene.append(
            {
                "scene": f"t{index:03d}",
                "any_safe_contact": bool(contact[index].any()),
                "any_useful_pass": bool(useful[index].any()),
                "safe_arm_count": int(safety[index].sum()),
            }
        )
    oracle_contacts = sum(row["any_safe_contact"] for row in per_scene)
    oracle_useful = sum(row["any_useful_pass"] for row in per_scene)
    gate = bool(
        oracle_contacts >= protocol["oracle_gate"]["minimum_safe_contacts"]
        and oracle_useful >= protocol["oracle_gate"]["minimum_useful_passes"]
    )
    args.output_dir.mkdir(parents=True)
    dataset_path = args.output_dir / "paired_outcomes.npz"
    np.savez_compressed(
        dataset_path,
        feature=np.asarray(features, dtype="<f8"),
        safe=safety,
        safe_contact=contact,
        useful_pass=useful,
        arm_names=np.asarray(names),
        entry_hashes=np.asarray(entry_hashes),
        scenario_hashes=np.asarray(scenarios),
    )
    result = {
        "schema": (
            "rsi_team_diverse_curriculum_audit_report_v29"
            if scene_count == 64
            else "rsi_team_diverse_curriculum_audit_report_v31"
        ),
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "physical_source_hashes": reports["parent"]["source_hashes"],
        "body_hash": reports["parent"]["asset_body_hash"],
        "dataset_hash": hash_bytes(dataset_path.read_bytes()),
        "arm_names": names,
        "scenario_count": scene_count,
        "training_count": 48 if scene_count == 64 else 128,
        "internal_validation_count": 16 if scene_count == 64 else 0,
        "entry_parity_verified": True,
        "oracle_safe_contact_diagnostic_only": oracle_contacts,
        "oracle_useful_pass_diagnostic_only": oracle_useful,
        "train_chooser_gate_passed": gate,
        "per_scene_oracle_diagnostic_only": per_scene,
        "fresh_holdout": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_DIVERSE_AUDIT="
        + json.dumps(
            {
                key: result[key]
                for key in (
                    "report_hash",
                    "dataset_hash",
                    "oracle_safe_contact_diagnostic_only",
                    "oracle_useful_pass_diagnostic_only",
                    "train_chooser_gate_passed",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
