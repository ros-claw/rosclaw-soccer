"""Recover completed physical episodes after a collector-only summary failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_contextual_action_dataset import entry_features
from scripts.rsi_team_contextual_nav_fresh_exam import fresh_scenarios
from scripts.rsi_team_intercept_navigation_search import _score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("recovery output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_fresh_failure_library_recovery_protocol_v26"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("collector_execution_commit") != "4e0e32e"
        or protocol.get("collector_execution_source_sha256")
        != "24b61b68de546441a5de8bb3d26fb32a15cabd22fd0b82cbfb6845816763b519"
        or protocol.get("expected_incomplete_arm")
        != {"arm": "phase_front04_lat04", "first_incomplete_scene": "f03"}
    ):
        raise ValueError("uncommitted consumed-data recovery protocol")
    failed = json.loads(Path(protocol["failed_exam_report_path"]).read_text())
    if (
        failed.get("report_hash") != protocol["failed_exam_report_hash"]
        or hash_json({key: value for key, value in failed.items() if key != "report_hash"})
        != failed["report_hash"]
        or failed.get("gate_passed") is not False
    ):
        raise ValueError("unsealed failed fresh exam")
    fresh_protocol = json.loads(Path(protocol["failed_exam_protocol_path"]).read_text())
    model = json.loads(Path(fresh_protocol["model_path"]).read_text())
    scenarios = fresh_scenarios(fresh_protocol)
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    incomplete = []
    root = Path(protocol["collector_output_root"])
    for arm in model["arm_names"]:
        rows = []
        for index, scenario in enumerate(scenarios):
            scene = f"f{index:02d}"
            episode = root / arm / scene / "candidate"
            report_path = episode / "report.json"
            trajectory_path = episode / "trajectory.npz"
            action_path = episode / "taskspace_trace.npz"
            if not report_path.exists():
                status = "ABORTED" if scene == "f03" else "NOT_RUN"
                incomplete.append({"arm": arm, "scene": scene, "status": status})
                rows.append(
                    {
                        "status": status,
                        "safe": False,
                        "foot_contact_frames": [],
                        "useful_pass": False,
                    }
                )
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                report["scenario_hash"] != scenario.scenario_hash
                or report["report_hash"]
                != hash_json({key: value for key, value in report.items() if key != "report_hash"})
                or report["trace_hash"] != hash_bytes(trajectory_path.read_bytes())
                or report["action_trace_hash"] != hash_bytes(action_path.read_bytes())
                or entry_features(action_path, 30)["hash"]
                != failed["episodes"][index]["entry_hash"]
            ):
                raise ValueError("completed physical episode integrity mismatch")
            outcome = _score(report, trajectory_path)
            outcome["status"] = "COMPLETE"
            rows.append(outcome)
        rows_by_arm[arm] = rows
    expected_incomplete = [
        {
            "arm": "phase_front04_lat04",
            "scene": f"f{index:02d}",
            "status": "ABORTED" if index == 3 else "NOT_RUN",
        }
        for index in range(3, 12)
    ]
    if incomplete != expected_incomplete:
        raise ValueError("unexpected incomplete physical episodes")
    per_scene = []
    for index, scenario in enumerate(scenarios):
        eligible = [
            name
            for name, rows in rows_by_arm.items()
            if rows[index]["status"] == "COMPLETE" and rows[index]["safe"]
        ]
        per_scene.append(
            {
                "scene": f"f{index:02d}",
                "scenario_hash": scenario.scenario_hash,
                "eligible_safe_arm_count": len(eligible),
                "any_safe_foot_contact": any(
                    rows_by_arm[name][index]["foot_contact_frames"] for name in eligible
                ),
                "any_useful_pass": any(
                    rows_by_arm[name][index]["useful_pass"] for name in eligible
                ),
                "selected_online_arm": failed["episodes"][index]["selected_arm"],
                "selected_online_useful_pass": failed["episodes"][index]["candidate_useful_pass"],
            }
        )
    args.output_dir.mkdir(parents=True)
    result = {
        "schema": "rsi_team_fresh_failure_library_recovery_report_v26",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "collector_execution_commit": protocol["collector_execution_commit"],
        "collector_execution_source_sha256": protocol["collector_execution_source_sha256"],
        "failed_fresh_report_hash": failed["report_hash"],
        "per_arm": {
            name: {
                "safe_contact_count": sum(
                    bool(row["safe"] and row["foot_contact_frames"]) for row in rows
                ),
                "useful_pass_count": sum(bool(row["useful_pass"]) for row in rows),
                "incomplete_count": sum(row["status"] != "COMPLETE" for row in rows),
                "rows": rows,
            }
            for name, rows in rows_by_arm.items()
        },
        "incomplete": incomplete,
        "completed_episode_count": sum(
            row["status"] == "COMPLETE" for rows in rows_by_arm.values() for row in rows
        ),
        "per_scene_oracle_diagnostic_only": per_scene,
        "oracle_safe_foot_contact_lower_bound": sum(
            row["any_safe_foot_contact"] for row in per_scene
        ),
        "oracle_useful_pass_lower_bound": sum(row["any_useful_pass"] for row in per_scene),
        "fresh_holdout": False,
        "policy_intervention_authorized": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_LIBRARY_RECOVERY="
        + json.dumps(
            {
                key: result[key]
                for key in (
                    "report_hash",
                    "completed_episode_count",
                    "incomplete",
                    "oracle_safe_foot_contact_lower_bound",
                    "oracle_useful_pass_lower_bound",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
