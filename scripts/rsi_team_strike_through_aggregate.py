"""Integrity-check consumed strike-through arms and compare physical coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("aggregate output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_strike_through_aggregate_protocol_v27"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or len(protocol.get("arm_report_hashes", {})) != 11
    ):
        raise ValueError("invalid frozen consumed-data aggregation protocol")
    old = json.loads(Path(protocol["previous_recovery_report_path"]).read_text())
    if (
        old["report_hash"] != protocol["previous_recovery_report_hash"]
        or hash_json({key: value for key, value in old.items() if key != "report_hash"})
        != old["report_hash"]
    ):
        raise ValueError("unsealed previous physical evidence")
    reports = {}
    for name, expected_hash in protocol["arm_report_hashes"].items():
        report = json.loads((Path(protocol["arm_result_root"]) / name / "report.json").read_text())
        if (
            report["report_hash"] != expected_hash
            or hash_json({key: value for key, value in report.items() if key != "report_hash"})
            != expected_hash
            or len(report["rows"]) != 12
        ):
            raise ValueError("unsealed or incomplete strike-through arm")
        reports[name] = report
    if len({str(report["source_hashes"]) for report in reports.values()}) != 1:
        raise ValueError("mixed physical source across arms")
    rows = []
    for index in range(12):
        previous = old["per_scene_oracle_diagnostic_only"][index]
        new = [
            name
            for name, report in reports.items()
            if name != "parent"
            and report["rows"][index]["status"] == "COMPLETE"
            and report["rows"][index]["safe"]
            and report["rows"][index]["useful_pass"]
        ]
        rows.append(
            {
                "scene": previous["scene"],
                "previous_any_useful": previous["any_useful_pass"],
                "new_safe_useful_arms": new,
                "combined_any_useful": previous["any_useful_pass"] or bool(new),
            }
        )
    result = {
        "schema": "rsi_team_strike_through_aggregate_report_v27",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "source_hashes": reports["parent"]["source_hashes"],
        "parent_parity_checked": reports["parent"]["incomplete_count"] == 0,
        "previous_oracle_useful_lower_bound": old["oracle_useful_pass_lower_bound"],
        "new_oracle_useful_count": sum(bool(row["new_safe_useful_arms"]) for row in rows),
        "combined_oracle_useful_lower_bound": sum(row["combined_any_useful"] for row in rows),
        "newly_covered_scenes": [
            row["scene"]
            for row in rows
            if row["combined_any_useful"] and not row["previous_any_useful"]
        ],
        "incomplete_arm_episodes": {
            name: report["incomplete_count"]
            for name, report in reports.items()
            if report["incomplete_count"]
        },
        "rows": rows,
        "fresh_holdout": False,
        "policy_intervention_authorized": False,
        "promotion_authorized": False,
    }
    if not result["parent_parity_checked"]:
        raise ValueError("parent parity not completed")
    result["report_hash"] = hash_json(result)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_STRIKE_THROUGH_AGGREGATE="
        + json.dumps(
            {
                key: result[key]
                for key in (
                    "report_hash",
                    "previous_oracle_useful_lower_bound",
                    "new_oracle_useful_count",
                    "combined_oracle_useful_lower_bound",
                    "newly_covered_scenes",
                    "incomplete_arm_episodes",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
