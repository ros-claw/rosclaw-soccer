"""Audit read-only 500 Hz B6 capture against unchanged six-G1 parent physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _verified_episode(folder: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = folder / "protocol.json"
    report_path = folder / "report.json"
    trace_path = folder / "trace.npz"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in report.items() if key != "report_hash"}
    if (
        report["report_hash"] != hash_json(body)
        or report["protocol_hash"] != hash_bytes(protocol_path.read_bytes())
        or report["trace_hash"] != hash_bytes(trace_path.read_bytes())
        or protocol["activation_ceiling"] != "SIM_ONLY"
        or protocol["promotion_authorized"] is not False
        or protocol["video_authorized"] is not False
    ):
        raise ValueError(f"unbound six-G1 physics evidence: {folder}")
    return protocol, report


def audit(evidence_root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.r1_b6_microphysics_observer_v104.protocol.v1"
        or protocol["partition"] != "CONSUMED_B6_CONTACT_DIAGNOSIS"
        or protocol["learning_authorized"] is not False
        or [row["id"] for row in protocol["scenes"]] != ["s02", "s04"]
    ):
        raise ValueError("frozen read-only B6 microphysics course required")
    rows = []
    keys = (
        "ball_pose",
        "ball_velocity",
        "ball_contact_agent_code",
        "ball_contact_foot_code",
        "ball_nonfoot_contact_agent_code",
        "red_finisher_joint_position",
        "red_finisher_joint_velocity",
    )
    for scene in protocol["scenes"]:
        scene_id = scene["id"]
        parent_dir = (
            evidence_root / "rsi-r1-local-b6-teacher-grid-v103" / f"candidate-04-{scene_id}"
        )
        observed_dir = evidence_root / f"rsi-r1-b6-microphysics-v104-r3-{scene_id}"
        parent_protocol, parent = _verified_episode(parent_dir)
        observed_protocol, observed = _verified_episode(observed_dir)
        micro = observed["b6_microphysics"]
        archive_path = observed_dir / "b6-microphysics.npz"
        if (
            parent_protocol["scenario"]["seed"] != scene["seed"]
            or observed_protocol["scenario"]["seed"] != scene["seed"]
            or parent_protocol["scenario"]["ball_initial_position_m"][:2]
            != [scene["ball_x_m"], scene["ball_y_m"]]
            or observed_protocol["scenario"]["ball_initial_position_m"][:2]
            != [scene["ball_x_m"], scene["ball_y_m"]]
            or observed_protocol["receive_teacher_tuning"] != [0.0, 0.18]
            or observed_protocol["capture_b6_microphysics"] is not True
            or parent_protocol["receive_teacher_tuning"] != [0.0, 0.18]
            or parent_protocol["motor_entry_frame"] != 0
            or observed_protocol["motor_entry_frame"] != 0
            or parent_protocol["duration_sec"] != 10.0
            or observed_protocol["duration_sec"] != 10.0
            or micro["observer_fault"] is not False
            or micro["complete"] is not True
            or micro["sample_count"] != 326
            or micro["archive_hash"] != hash_bytes(archive_path.read_bytes())
        ):
            raise ValueError(f"unbound or incomplete B6 microphysics capture: {observed_dir}")
        with (
            np.load(parent_dir / "trace.npz", allow_pickle=False) as parent_trace,
            np.load(observed_dir / "trace.npz", allow_pickle=False) as observed_trace,
            np.load(archive_path, allow_pickle=False) as physics,
        ):
            equivalent = {
                key: bool(np.array_equal(parent_trace[key], observed_trace[key])) for key in keys
            }
            time = physics["time_sec"]
            own_force = physics["own_foot_normal_force_n"]
            geometry = physics["contact_geometry_complete"]
            safe = physics["world_bodies_safe"]
            if (
                time.shape != (326,)
                or physics["qpos"].shape != (326, 43)
                or physics["qvel"].shape != (326, 41)
                or not np.all(np.isfinite(physics["qpos"]))
                or not np.all(np.isfinite(physics["qvel"]))
                or not np.allclose(np.diff(time), 0.002, atol=1e-6, rtol=0)
                or not np.isclose(time[75], micro["first_foot_time_sec"], atol=1e-6)
                or own_force[75] <= 0
                or not bool(geometry[75])
            ):
                raise ValueError(f"invalid 500 Hz foot-contact window: {archive_path}")
            rows.append(
                {
                    "scene": scene_id,
                    "parent_report_hash": parent["report_hash"],
                    "observed_report_hash": observed["report_hash"],
                    "microphysics_archive_hash": micro["archive_hash"],
                    "array_equivalence": equivalent,
                    "full_episode_safe_agrees": bool(
                        parent["result"]["safe"] == observed["result"]["safe"]
                    ),
                    "incoming_ball_speed_mps": micro["incoming_ball_speed_mps"],
                    "first_foot_time_sec": micro["first_foot_time_sec"],
                    "peak_own_foot_force_n": float(np.max(own_force)),
                    "minimum_window_body_safe": bool(np.all(safe)),
                    "microphysics_complete": True,
                }
            )
    gates = {
        "both_500hz_captures_complete": all(row["microphysics_complete"] for row in rows),
        "read_only_trajectory_equivalent": all(
            all(row["array_equivalence"].values()) and row["full_episode_safe_agrees"]
            for row in rows
        ),
        "both_dynamic_incoming": all(row["incoming_ball_speed_mps"] >= 0.30 for row in rows),
    }
    result = {
        "schema": "rosclaw_soccer.rsi.r1_b6_microphysics_audit_v104.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "rows": rows,
        "gates": gates,
        "status": "OBSERVATIONAL_CAPTURE_ACCEPTED" if all(gates.values()) else "REJECTED_CAPTURE",
        "learning_authorized": False,
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
        raise ValueError("new immutable microphysics audit output required")
    result = audit(args.evidence_root, args.protocol)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "report_hash": result["report_hash"]}))


if __name__ == "__main__":
    main()
