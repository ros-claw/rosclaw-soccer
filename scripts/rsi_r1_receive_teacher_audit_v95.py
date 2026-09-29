"""Audit privileged receive-teacher reachability in frozen six-G1 physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _read(directory: Path, expected_scene: dict[str, Any], profile: str) -> dict[str, Any]:
    protocol_file = directory / "protocol.json"
    trace_file = directory / "trace.npz"
    report_file = directory / "report.json"
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    report = json.loads(report_file.read_text(encoding="utf-8"))
    report_hash = report.pop("report_hash")
    scenario = protocol["scenario"]
    if (
        hash_json(report) != report_hash
        or hash_bytes(protocol_file.read_bytes()) != report["protocol_hash"]
        or hash_bytes(trace_file.read_bytes()) != report["trace_hash"]
        or scenario["ball_initial_position_m"][:2]
        != [expected_scene["ball_x_m"], expected_scene["ball_y_m"]]
        or scenario["seed"] != expected_scene["seed"]
        or protocol.get("receive_teacher_profile", "default") != profile
        or protocol["precontact_pass_standoff_m"] != 0.35
        or protocol["handoff_profile"] != "tracking"
        or protocol["stance_lateral_m"] != -0.19
        or protocol["activation_ceiling"] != "SIM_ONLY"
        or protocol["promotion_authorized"] is not False
        or protocol["video_authorized"] is not False
    ):
        raise ValueError(f"unbound or tampered receive-teacher evidence: {directory}")
    with np.load(trace_file, allow_pickle=False) as trace:
        receiver = np.flatnonzero(
            (trace["ball_contact_agent_code"] == 4)
            & np.isin(trace["ball_contact_foot_code"], (1, 2))
        )
        first = None if not len(receiver) else int(receiver[0])
        later = (
            []
            if first is None
            else [
                int(frame)
                for frame in receiver
                if float(trace["time"][frame]) >= float(trace["time"][first]) + 0.20
                and float(np.linalg.norm(trace["ball_velocity"][frame, :3])) >= 2.0
            ]
        )
        return {
            "report_hash": report_hash,
            "protocol_hash": report["protocol_hash"],
            "trace_hash": report["trace_hash"],
            "safe": bool(report["result"]["safe"]),
            "clean_transfer": bool((report["chain"] or {}).get("clean_transfer_observed")),
            "receiver_foot_frame": first,
            "post_receive_speed_mps": (
                None if first is None else float(np.linalg.norm(trace["ball_velocity"][first, :3]))
            ),
            "later_foot_shot_frames": later,
            "nonfoot_ball_contact": bool(np.any(trace["ball_nonfoot_contact_agent_code"] != 0)),
            "robot_robot_contact_count": int(report["result"]["robot_robot_contact_count"]),
            "in_goal_crossing": bool((report["crossing"] or {}).get("inside_geometry")),
        }


def audit(evidence_root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["schema"] != "rosclaw_soccer.rsi.r1_shared_receive_teacher_v95.protocol.v1":
        raise ValueError("frozen teacher protocol required")
    scenes = protocol["consumed_scenes"]
    arms = protocol["arms"]
    if len(scenes) != 3 or [arm["name"] for arm in arms] != [
        "neutral",
        "soft",
        "cushion",
        "combined",
    ]:
        raise ValueError("frozen teacher course changed")
    baseline = {
        scene["id"]: _read(
            evidence_root / f"rsi-r1-pass-acquisition-v88-{scene['id']}-standoff0.35",
            scene,
            "default",
        )
        for scene in scenes
    }
    results = []
    for arm in arms:
        name = arm["name"]
        rows = {
            scene["id"]: _read(
                evidence_root / f"rsi-r1-receive-teacher-v95-{scene['id']}-{name}",
                scene,
                name,
            )
            for scene in scenes
        }
        ratios = [
            rows[scene["id"]]["post_receive_speed_mps"]
            / baseline[scene["id"]]["post_receive_speed_mps"]
            if rows[scene["id"]]["post_receive_speed_mps"] is not None
            and baseline[scene["id"]]["post_receive_speed_mps"] is not None
            else None
            for scene in scenes
        ]
        finite_ratios = [value for value in ratios if value is not None]
        gates = {
            "all_safe_clean_no_forbidden_contacts": all(
                row["safe"]
                and row["clean_transfer"]
                and not row["nonfoot_ball_contact"]
                and row["robot_robot_contact_count"] == 0
                for row in rows.values()
            ),
            "each_receipt_slower_than_parent": len(finite_ratios) == len(scenes)
            and all(value < 1.0 for value in finite_ratios),
            "mean_speed_ratio_at_most_075": len(finite_ratios) == len(scenes)
            and float(np.mean(finite_ratios)) <= 0.75,
        }
        results.append(
            {
                "name": name,
                "rows": rows,
                "post_receive_speed_ratios": ratios,
                "mean_speed_ratio": (
                    float(np.mean(finite_ratios)) if len(finite_ratios) == len(scenes) else None
                ),
                "later_foot_shot_count": sum(
                    len(row["later_foot_shot_frames"]) for row in rows.values()
                ),
                "gates": gates,
                "teacher_positive_data_reachable": all(gates.values()),
            }
        )
    result = {
        "schema": "rosclaw_soccer.rsi.r1_receive_teacher_audit_v95.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "status": (
            "TEACHER_REACHABLE_CONSUMED_ONLY"
            if any(arm["teacher_positive_data_reachable"] for arm in results)
            else "REJECTED_TEACHER_REACHABILITY"
        ),
        "baseline": baseline,
        "arms": results,
        "teacher_is_privileged": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new immutable teacher audit output required")
    result = audit(args.evidence_root, args.protocol)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "report_hash": result["report_hash"]}))


if __name__ == "__main__":
    main()
