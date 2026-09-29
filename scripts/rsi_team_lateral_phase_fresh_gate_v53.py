"""Apply the precommitted primary v53 gate to independently audited MuJoCo scenes."""

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
        raise ValueError(f"unsealed report: {path}")
    return report


def evaluate_primary_gate(
    metrics: dict[str, dict[str, int]], gate: dict[str, Any]
) -> tuple[int, int, int, bool]:
    """Keep missing episodes as failures, even when completed ones improve."""
    baseline = metrics["baseline"]
    primary = metrics[gate["primary_arm"]]
    parent = metrics["parent"]
    if any(
        score["complete"] + score["incomplete"] != 32 or not 0 <= score["safe"] <= score["complete"]
        for score in (baseline, primary, parent)
    ):
        raise ValueError("invalid audited fresh physical episode counts")
    useful_gain = primary["useful"] - baseline["useful"]
    contact_gain = primary["safe_contact"] - baseline["safe_contact"]
    unsafe_excess = (32 - primary["safe"]) - (32 - baseline["safe"])
    passed = bool(
        primary["complete"] == 32
        and parent["complete"] == 32
        and useful_gain >= gate["minimum_useful_gain_over_baseline"]
        and contact_gain >= gate["minimum_safe_contact_gain_over_baseline"]
        and unsafe_excess <= gate["maximum_unsafe_excess_over_baseline"]
    )
    return useful_gain, contact_gain, unsafe_excess, passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("v53 fresh gate output already exists")
    protocol = json.loads(args.protocol.read_text())
    gate = protocol.get("fresh_gate")
    if gate != {
        "primary_arm": "gate22_cap10",
        "minimum_useful_gain_over_baseline": 3,
        "minimum_safe_contact_gain_over_baseline": 5,
        "maximum_unsafe_excess_over_baseline": 0,
        "require_primary_and_parent_complete": True,
    }:
        raise ValueError("invalid frozen primary fresh gate")
    old_path = Path(protocol["prior_development_report_path"])
    old_protocol_path = Path(protocol["prior_development_protocol_path"])
    if (
        hash_bytes(old_path.read_bytes()) != protocol["prior_development_report_file_hash"]
        or hash_bytes(old_protocol_path.read_bytes())
        != protocol["prior_development_protocol_file_hash"]
    ):
        raise ValueError("prior development evidence changed")
    old = _sealed(old_path)
    if (
        old["report_hash"] != protocol["prior_development_report_hash"]
        or old["protocol_hash"] != protocol["prior_development_protocol_file_hash"]
        or not any(
            item["arm"] == "gate22_cap10" and item["development_gate_passed"]
            for item in old["candidate_comparisons"]
        )
    ):
        raise ValueError("frozen candidate lacks prior physical development evidence")
    old_protocol = json.loads(old_protocol_path.read_text())
    old_scenes = {scene.scenario_hash for scene in training_courses(old_protocol)}
    new_scenes = {scene.scenario_hash for scene in training_courses(protocol)}
    old_states = {
        (scene.ball_initial_position_m, scene.ball_initial_velocity_mps)
        for scene in training_courses(old_protocol)
    }
    new_states = {
        (scene.ball_initial_position_m, scene.ball_initial_velocity_mps)
        for scene in training_courses(protocol)
    }
    if (
        len(old_scenes) != 32
        or len(new_scenes) != 32
        or old_scenes & new_scenes
        or old_states & new_states
    ):
        raise ValueError("fresh physical scenes overlap development scenes")
    audit = _sealed(args.audit_report)
    if (
        audit.get("protocol_hash") != hash_bytes(args.protocol.read_bytes())
        or audit.get("scene_count") != 32
        or audit.get("body_hash") != old["body_hash"]
        or audit.get("physical_source_hashes") != old["physical_source_hashes"]
    ):
        raise ValueError("independent exam changed action or G1 body sources")
    metrics = audit["metrics"]
    baseline = metrics["baseline"]
    primary = metrics[gate["primary_arm"]]
    useful_gain, contact_gain, unsafe_excess, passed = evaluate_primary_gate(metrics, gate)
    result = {
        "schema": "rsi_team_lateral_phase_fresh_gate_report_v53",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "prior_development_report_hash": old["report_hash"],
        "fresh_physical_audit_hash": audit["report_hash"],
        "body_hash": audit["body_hash"],
        "scene_count": 32,
        "baseline": baseline,
        "primary": primary,
        "useful_gain": useful_gain,
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
    print("RSI_TEAM_V53_FRESH=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
