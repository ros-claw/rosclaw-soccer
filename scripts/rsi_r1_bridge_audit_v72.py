"""Read-only auditor for the frozen R1 receiver bridge local exam."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def audit(protocol_path: Path, evidence_root: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != "rosclaw_soccer.rsi.r1_receiver_bridge_fresh_v72.protocol.v1":
        raise ValueError("frozen v72 protocol required")
    scenes = protocol["scenes"]
    if len(scenes) != 8 or len({scene["id"] for scene in scenes}) != 8:
        raise ValueError("exactly eight unique frozen scenes required")
    rows: list[dict[str, Any]] = []
    sources: dict[str, str] | None = None
    chosen = protocol["chosen_from_consumed"]
    action = {
        "entry_frame": 30,
        "forward_cap_m": 0.16,
        "lateral_cap_m": 0.10,
        "vertical_offset_m": 0.04,
        "swing_foot_acquisition_gap_m": 0.55,
        "swing_acquisition_max_lateral_gap_m": 0.22,
        "revalidate_swing_side": True,
        "joint_boundary_recovery_cap_rad": 0.04,
        "rolling_cap_rad": chosen["rolling_cap_rad"],
        "rolling_radius_m": chosen["rolling_radius_m"],
        "rolling_vertical_m": chosen["rolling_vertical_m"],
        "rolling_requires_commitment": chosen["rolling_requires_commitment"],
    }
    expected_motor_hash = hash_json(
        {
            "schema": "rsi_team_swing_motor_v13",
            "agent": "red.finisher",
            "enabled": True,
            "action": action,
            "activation_selector_hash": None,
            "activation_ceiling": "SIM_ONLY",
        }
    )
    for scene in scenes:
        row: dict[str, Any] = {"scene": scene["id"]}
        for arm in ("control", "candidate"):
            directory = evidence_root / f"rsi-r1-bridge-fresh-v72-{scene['id']}-{arm}"
            run_protocol_path = directory / "protocol.json"
            report_path = directory / "report.json"
            trace_path = directory / "trace.npz"
            run_protocol = json.loads(run_protocol_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                report["protocol_hash"] != hash_bytes(run_protocol_path.read_bytes())
                or report["trace_hash"] != hash_bytes(trace_path.read_bytes())
                or report["report_hash"]
                != hash_json({key: value for key, value in report.items() if key != "report_hash"})
            ):
                raise ValueError(f"broken evidence integrity: {scene['id']} {arm}")
            actual = run_protocol["scenario"]
            if (
                actual["seed"] != scene["seed"]
                or actual["ball_initial_position_m"]
                != [scene["ball_x_m"], scene["ball_y_m"], 0.115]
                or run_protocol["activation_ceiling"] != "SIM_ONLY"
                or run_protocol["promotion_authorized"] is not False
            ):
                raise ValueError(f"protocol mismatch: {scene['id']} {arm}")
            if sources is None:
                sources = run_protocol["source_hashes"]
            elif sources != run_protocol["source_hashes"]:
                raise ValueError("source drift across paired exam")
            if arm == "candidate":
                if (
                    run_protocol["enabled"] is not True
                    or run_protocol["motor_present"] is not True
                    or run_protocol["grounded"] is not True
                    or run_protocol["retired_ankle_braking"]
                    != chosen["retired_motor_option_ankle_braking_damping"]
                    or run_protocol["motor_contract_hash"] != expected_motor_hash
                ):
                    raise ValueError("candidate arm differs from frozen intervention")
            elif run_protocol["motor_present"] is not False:
                raise ValueError("control arm must not own receiver motor")
            chain = report["chain"]
            row[arm] = {
                "safe": bool(report["result"]["safe"]),
                "clean_transfer": bool(chain and chain["clean_transfer_observed"]),
                "chain_success": bool(report["chain_success"]),
                "motor_active_frames": report["motor_active_frames"],
                "failure_reasons": [] if chain is None else chain["reasons"],
                "report_hash": report["report_hash"],
            }
        rows.append(row)
    controls = [row["control"] for row in rows]
    candidates = [row["candidate"] for row in rows]
    counts = {
        arm: {
            key: sum(bool(row[key]) for row in records)
            for key in ("safe", "clean_transfer", "chain_success")
        }
        for arm, records in (("control", controls), ("candidate", candidates))
    }
    newly_unsafe = [
        row["scene"] for row in rows if row["control"]["safe"] and not row["candidate"]["safe"]
    ]
    gate = protocol["frozen_gate"]
    passed = bool(
        len(newly_unsafe) <= gate["maximum_newly_unsafe"]
        and counts["candidate"]["safe"] >= gate["minimum_candidate_safe"]
        and counts["candidate"]["clean_transfer"] >= gate["minimum_candidate_clean_transfer"]
        and counts["candidate"]["chain_success"] >= gate["minimum_candidate_chain_success"]
        and counts["candidate"]["chain_success"] - counts["control"]["chain_success"]
        >= gate["minimum_chain_gain_over_control"]
    )
    result = {
        "schema": "rosclaw_soccer.rsi.r1_receiver_bridge_fresh_v72.audit.v1",
        "activation_ceiling": "SIM_ONLY",
        "frozen_protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "source_hashes": sources,
        "counts": counts,
        "newly_unsafe": newly_unsafe,
        "rows": rows,
        "frozen_gate_passed": passed,
        "promotion_authorized": False,
        "video_authorized": False,
    }
    result["audit_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.protocol, args.evidence_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
