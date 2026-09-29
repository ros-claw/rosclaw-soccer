"""Verify measured-contact capture in consumed six-G1 physics scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _read(
    directory: Path, scene: dict[str, Any], profile: str, teacher_profile: str
) -> dict[str, Any]:
    protocol_file = directory / "protocol.json"
    trace_file = directory / "trace.npz"
    report_file = directory / "report.json"
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    report = json.loads(report_file.read_text(encoding="utf-8"))
    report_hash = report.pop("report_hash")
    if (
        hash_json(report) != report_hash
        or hash_bytes(protocol_file.read_bytes()) != report["protocol_hash"]
        or hash_bytes(trace_file.read_bytes()) != report["trace_hash"]
        or protocol["scenario"]["ball_initial_position_m"][:2]
        != [scene["ball_x_m"], scene["ball_y_m"]]
        or protocol["scenario"]["seed"] != scene["seed"]
        or protocol.get("capture_profile", "none") != profile
        or protocol["precontact_pass_standoff_m"] != 0.35
        or protocol["handoff_profile"] != "tracking"
        or protocol["stance_lateral_m"] != -0.19
        or protocol["navigation_profile"] != "none"
        or protocol["receive_teacher_profile"] != "default"
        or protocol["teacher_profile"] != teacher_profile
        or protocol["activation_ceiling"] != "SIM_ONLY"
        or protocol["promotion_authorized"] is not False
        or protocol["video_authorized"] is not False
    ):
        raise ValueError(f"unbound or tampered capture evidence: {directory}")
    with np.load(trace_file, allow_pickle=False) as trace:
        receiver_foot = np.flatnonzero(
            (trace["ball_contact_agent_code"] == 4)
            & np.isin(trace["ball_contact_foot_code"], (1, 2))
        )
        first = int(receiver_foot[0]) if len(receiver_foot) else None
        launch = np.flatnonzero(trace["ball_contact_agent_code"] == 6)
        start = int(launch[0]) if len(launch) else 0
        stop = (
            len(trace["time"])
            if first is None
            else int(np.searchsorted(trace["time"], float(trace["time"][first]) + 0.60))
        )
        later = (
            []
            if first is None
            else [
                int(frame)
                for frame in receiver_foot
                if float(trace["time"][frame]) >= float(trace["time"][first]) + 0.20
                and float(np.linalg.norm(trace["ball_velocity"][frame, :3])) >= 2.0
            ]
        )
        return {
            "report_hash": report_hash,
            "safe": bool(report["result"]["safe"]),
            "clean_transfer": bool((report["chain"] or {}).get("clean_transfer_observed")),
            "receiver_first_foot_sec": (None if first is None else float(trace["time"][first])),
            "post_receive_capture_context_frames": (
                0
                if first is None
                else int(np.count_nonzero(trace["post_receive_capture_context"][first:stop]))
            ),
            "post_receive_teacher_torque_frames": (
                0
                if first is None
                else int(
                    np.count_nonzero(
                        np.any(trace["post_receive_capture_context"][first + 1 : stop], axis=1)
                        & (trace["contact_teacher_peak_torque_nm"][first + 1 : stop] > 1.0e-6)
                    )
                )
            ),
            "ball_nonfoot_contact_window_count": int(
                np.count_nonzero(trace["ball_nonfoot_contact_agent_code"][start:stop])
            ),
            "robot_robot_contact_count": int(report["result"]["robot_robot_contact_count"]),
            "later_receiver_foot_shot_frames": later,
            "in_goal_crossing": bool((report["crossing"] or {}).get("inside_geometry")),
        }


def audit(evidence_root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    schema = protocol["schema"]
    if schema not in (
        "rosclaw_soccer.rsi.r1_shared_postreceive_capture_v96.protocol.v1",
        "rosclaw_soccer.rsi.r1_shared_capture_activation_v97.protocol.v1",
        "rosclaw_soccer.rsi.r1_shared_capture_authority_v98.protocol.v1",
        "rosclaw_soccer.rsi.r1_shared_capture_stance_v99.protocol.v1",
    ):
        raise ValueError("frozen capture protocol required")
    version = next(value for value in ("v99", "v98", "v97", "v96") if value in schema)
    teacher_profile = "live_after_receive" if version != "v96" else "default"
    directory_prefix = {
        "v96": "rsi-r1-postreceive-capture-v96",
        "v97": "rsi-r1-capture-activation-v97",
        "v98": "rsi-r1-capture-authority-v98",
        "v99": "rsi-r1-capture-stance-v99",
    }[version]
    scenes = protocol["consumed_scenes"]
    arms = protocol["arms"]
    if len(scenes) != 3 or [arm["name"] for arm in arms] != ["short", "medium", "long"]:
        raise ValueError("frozen capture course changed")
    results = []
    for arm in arms:
        name = arm["name"]
        rows = {
            scene["id"]: _read(
                evidence_root / f"{directory_prefix}-{scene['id']}-{name}",
                scene,
                name,
                teacher_profile,
            )
            for scene in scenes
        }
        gates = {
            "capture_actually_active": all(
                row["post_receive_capture_context_frames"] > 0 for row in rows.values()
            ),
            "teacher_has_postreceive_torque": (
                all(
                    row["post_receive_teacher_torque_frames"] >= (5 if version == "v99" else 3)
                    for row in rows.values()
                )
                if version in ("v98", "v99")
                else True
            ),
            "all_safe_clean": all(row["safe"] and row["clean_transfer"] for row in rows.values()),
            "no_early_nonfoot_contact": all(
                row["ball_nonfoot_contact_window_count"] == 0 for row in rows.values()
            ),
            "no_robot_collision": all(
                row["robot_robot_contact_count"] == 0 for row in rows.values()
            ),
            "two_true_later_shots": sum(
                bool(row["later_receiver_foot_shot_frames"]) for row in rows.values()
            )
            >= 2,
        }
        results.append(
            {
                "name": name,
                "rows": rows,
                "gates": gates,
                "teacher_positive_data_reachable": all(gates.values()),
            }
        )
    result = {
        "schema": f"rosclaw_soccer.rsi.r1_postreceive_capture_audit_{version}.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "status": (
            "TEACHER_REACHABLE_CONSUMED_ONLY"
            if any(arm["teacher_positive_data_reachable"] for arm in results)
            else "REJECTED_TEACHER_REACHABILITY"
        ),
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
        raise ValueError("new immutable capture audit output required")
    result = audit(args.evidence_root, args.protocol)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "report_hash": result["report_hash"]}))


if __name__ == "__main__":
    main()
