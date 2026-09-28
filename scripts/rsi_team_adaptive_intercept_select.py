"""Select one safety-qualified SIM_ONLY interception policy from physical training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_adaptive_intercept_search import candidate_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("selection output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    search_path = Path(protocol["training_protocol_path"])
    search = json.loads(search_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_adaptive_intercept_selection_protocol_v28"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or search.get("schema") != "rsi_team_adaptive_intercept_search_protocol_v28"
        or len(protocol.get("candidate_report_hashes", [])) != 24
    ):
        raise ValueError("uncommitted SIM_ONLY physical selection protocol")
    parameters = candidate_parameters(search["parameter_seed"], 24)
    reports = []
    for index, expected_hash in enumerate(protocol["candidate_report_hashes"]):
        report = json.loads(
            (Path(protocol["training_result_root"]) / f"c{index:02d}/report.json").read_text()
        )
        if (
            report.pop("report_hash", None) != expected_hash
            or hash_json(report) != expected_hash
            or report["candidate_index"] != index
            or report["parameters"] != parameters[index]
            or report["protocol_hash"] != hash_bytes(search_path.read_bytes())
            or len(report["rows"]) != 6
        ):
            raise ValueError("physical candidate evidence mismatch")
        reports.append(report)
    if len({str(report["source_hashes"]) for report in reports}) != 1:
        raise ValueError("mixed physical training source")
    eligible = [
        report for report in reports if report["all_safe"] and report["incomplete_count"] == 0
    ]
    if not eligible:
        raise ValueError("no safety-qualified physical search candidate")

    def rank(report: dict) -> tuple[float, ...]:
        mean_gap = sum(row["minimum_foot_ball_gap_m"] for row in report["rows"]) / 6
        return (
            -report["useful_pass_count"],
            -report["safe_contact_count"],
            mean_gap,
            report["candidate_index"],
        )

    chosen = min(eligible, key=rank)
    model = {
        "schema": "rsi_team_adaptive_intercept_memory_v28",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "training_protocol_hash": hash_bytes(search_path.read_bytes()),
        "selection_protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "physical_source_hashes": chosen["source_hashes"],
        "asset_body_hash": chosen["asset_body_hash"],
        "chosen_candidate_index": chosen["candidate_index"],
        "parameters": chosen["parameters"],
        "training_safe_count": sum(row["safe"] for row in chosen["rows"]),
        "training_safe_contact_count": chosen["safe_contact_count"],
        "training_useful_pass_count": chosen["useful_pass_count"],
    }
    model["model_hash"] = hash_json(model)
    summary = {
        "schema": "rsi_team_adaptive_intercept_selection_report_v28",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "eligible_candidate_indices": [report["candidate_index"] for report in eligible],
        "selected_candidate_index": chosen["candidate_index"],
        "selected_candidate_rank": rank(chosen),
        "model_hash": model["model_hash"],
        "development_only": True,
        "promotion_authorized": False,
    }
    summary["report_hash"] = hash_json(summary)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "selection_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_ADAPTIVE_SELECTION="
        + json.dumps(
            {
                "model_hash": model["model_hash"],
                "report_hash": summary["report_hash"],
                "selected_candidate_index": chosen["candidate_index"],
                "eligible_count": len(eligible),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
