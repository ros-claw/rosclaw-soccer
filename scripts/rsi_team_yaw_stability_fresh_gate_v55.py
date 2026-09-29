"""Apply frozen 64-scene all-team safety and passing gate to paired physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_diverse_curriculum_collect import training_courses


def _sealed(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text())
    if report.get("report_hash") != hash_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    ):
        raise ValueError(f"unsealed physical report: {path}")
    return report


def evaluate_fresh_gate(
    metrics: dict[str, dict[str, int]], gate: dict[str, Any], scene_count: int
) -> tuple[int, int, int, bool]:
    baseline = metrics["baseline"]
    primary = metrics[gate["primary_arm"]]
    parent = metrics["parent"]
    if any(
        score["complete"] + score["incomplete"] != scene_count
        or not 0 <= score["useful"] <= score["safe_contact"] <= score["safe"] <= score["complete"]
        for score in (baseline, primary, parent)
    ):
        raise ValueError("invalid audited team outcome accounting")
    useful_gain = primary["useful"] - baseline["useful"]
    contact_gain = primary["safe_contact"] - baseline["safe_contact"]
    unsafe_excess = baseline["safe"] - primary["safe"]
    passed = bool(
        primary["complete"] == scene_count
        and parent["complete"] == scene_count
        and primary["useful"] >= gate["minimum_primary_useful"]
        and useful_gain >= gate["minimum_useful_gain_over_baseline"]
        and contact_gain >= gate["minimum_safe_contact_gain_over_baseline"]
        and unsafe_excess <= gate["maximum_unsafe_excess_over_baseline"]
    )
    return useful_gain, contact_gain, unsafe_excess, passed


def _physical_states(protocol: dict[str, Any]) -> set[tuple[Any, Any]]:
    return {
        (scene.ball_initial_position_m, scene.ball_initial_velocity_mps)
        for batch in range(protocol["curriculum"]["batch_count"])
        for scene in training_courses(protocol, batch)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("v55 fresh gate evidence already exists")
    protocol = json.loads(args.protocol.read_text())
    gate = protocol.get("fresh_gate")
    if (
        protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("promotion_authorized") is not False
        or protocol.get("maximum_yaw_rate_radps") != 0.40
        or protocol.get("curriculum")
        != {
            "coordinate_seed": 20261034,
            "identity_seed_base": 1800000,
            "batch_count": 2,
            "training_scene_count": 32,
            "ball_x_m": [3.25, 4.25],
            "ball_y_m": [-0.75, -0.35],
            "ball_vx_mps": [-0.90, -0.20],
        }
        or gate
        != {
            "primary_arm": "gate22_cap10",
            "minimum_useful_gain_over_baseline": 6,
            "minimum_safe_contact_gain_over_baseline": 10,
            "maximum_unsafe_excess_over_baseline": 0,
            "minimum_primary_useful": 16,
            "require_primary_and_parent_complete": True,
        }
        or len(protocol.get("excluded_prior_curricula", ())) != 3
    ):
        raise ValueError("invalid precommitted 64-scene fresh physical gate")
    old_path = Path(protocol["prior_development_report_path"])
    old_protocol_path = Path(protocol["prior_development_protocol_path"])
    if (
        hash_bytes(old_path.read_bytes()) != protocol["prior_development_report_file_hash"]
        or hash_bytes(old_protocol_path.read_bytes())
        != protocol["prior_development_protocol_file_hash"]
    ):
        raise ValueError("prior team development evidence changed")
    old = _sealed(old_path)
    if (
        old["report_hash"] != protocol["prior_development_report_hash"]
        or old["protocol_hash"] != protocol["prior_development_protocol_file_hash"]
        or old["metrics"]["gate22_cap10"]["complete"] != 32
    ):
        raise ValueError("frozen main candidate lacks complete prior development")
    fresh_states = _physical_states(protocol)
    if len(fresh_states) != 64:
        raise ValueError("duplicate fresh physical states")
    for item in protocol["excluded_prior_curricula"]:
        path = Path(item["path"])
        if hash_bytes(path.read_bytes()) != item["file_hash"]:
            raise ValueError("prior curriculum changed")
        if fresh_states & _physical_states(json.loads(path.read_text())):
            raise ValueError("fresh physical state previously consumed")
    audit = _sealed(args.audit_report)
    if (
        audit.get("schema") != "rsi_team_paired_physics_audit_report_v1"
        or audit.get("protocol_hash") != hash_bytes(args.protocol.read_bytes())
        or audit.get("scene_count") != 64
        or audit.get("body_hash") != old["body_hash"]
        or audit.get("physical_source_hashes") != old["physical_source_hashes"]
    ):
        raise ValueError("fresh team physics changed body or motor semantics")
    metrics = audit["metrics"]
    gain, contact_gain, unsafe_excess, passed = evaluate_fresh_gate(metrics, gate, 64)
    result = {
        "schema": "rsi_team_yaw_stability_fresh_gate_report_v55",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "prior_development_report_hash": old["report_hash"],
        "fresh_physical_audit_hash": audit["report_hash"],
        "body_hash": audit["body_hash"],
        "scene_count": 64,
        "baseline": metrics["baseline"],
        "primary": metrics["gate22_cap10"],
        "useful_gain": gain,
        "safe_contact_gain": contact_gain,
        "unsafe_excess": unsafe_excess,
        "fresh_gate_passed": passed,
        "continuous_match_validated": False,
    }
    result["report_hash"] = hash_json(result)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RSI_TEAM_V55_FRESH=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
