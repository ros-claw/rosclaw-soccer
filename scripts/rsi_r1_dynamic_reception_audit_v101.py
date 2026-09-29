"""Separate moving-pass reception from stationary re-strikes in consumed evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def audit(
    evidence_root: Path, source_report_path: Path, source_protocol_path: Path, protocol_path: Path
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source = json.loads(source_report_path.read_text(encoding="utf-8"))
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.r1_shared_dynamic_reception_v101.protocol.v1"
        or protocol["partition"] != "CONSUMED_FAILURE_REFINEMENT"
        or protocol["source_manifest"] != "r1-shared-contact-mechanism-v100.json"
        or protocol["source_protocol"] != "r1-shared-contact-mechanism-v100.json"
        or protocol["learning_authorized"] is not False
        or source["schema"] != "rosclaw_soccer.rsi.r1_shared_contact_mechanism_v100.v1"
        or source["report_hash"]
        != hash_json({key: value for key, value in source.items() if key != "report_hash"})
        or source["protocol_hash"] != hash_bytes(source_protocol_path.read_bytes())
    ):
        raise ValueError("authenticated consumed v100 contact map required")
    arms = {arm["name"]: arm["folder_pattern"] for arm in source_protocol["arms"]}
    scenes = {scene["id"]: scene for scene in source_protocol["scenes"]}
    if len(source["rows"]) != 15 or len(arms) != 5 or len(scenes) != 3:
        raise ValueError("frozen 5x3 contact map required")
    rows = []
    for row in source["rows"]:
        folder = evidence_root / arms[row["arm"]].format(scene=row["scene"])
        trace_path = folder / "trace.npz"
        if hash_bytes(trace_path.read_bytes()) != row["source_trace_hash"]:
            raise ValueError(f"physical trace changed: {folder}")
        diagnostic = row["contact_diagnostic"]
        incoming = None if diagnostic is None else float(diagnostic["incoming_ball_speed_mps"])
        dynamic = incoming is not None and incoming >= protocol["dynamic_incoming_min_speed_mps"]
        first = row["first_receiver_foot_frame"]
        qualifying: list[dict[str, Any]] = []
        with np.load(trace_path, allow_pickle=False) as trace:
            if first is not None:
                own = np.flatnonzero(
                    (trace["ball_contact_agent_code"] == 4)
                    & np.isin(trace["ball_contact_foot_code"], (1, 2))
                )
                for frame in own:
                    frame = int(frame)
                    if (
                        float(trace["time"][frame]) - float(trace["time"][first])
                        < protocol["second_foot_min_delay_sec"]
                        or float(np.linalg.norm(trace["ball_velocity"][frame, :3]))
                        < protocol["second_foot_min_outgoing_speed_mps"]
                    ):
                        continue
                    if np.any(trace["ball_nonfoot_contact_agent_code"][first : frame + 1] != 0):
                        continue
                    qualifying.append(
                        {
                            "frame": frame,
                            "time_sec": float(trace["time"][frame]),
                            "outgoing_ball_speed_mps": float(
                                np.linalg.norm(trace["ball_velocity"][frame, :3])
                            ),
                        }
                    )
        if not dynamic:
            label = "NO_DYNAMIC_RECEPTION" if first is None else "STATIONARY_RESTRIKE"
        elif qualifying and row["safe"] and row["clean_transfer"]:
            label = "LOCAL_B6_TEACHER_EVENT_CONSUMED"
        elif not row["safe"]:
            label = "UNSAFE_DYNAMIC_RECEPTION"
        else:
            label = "NO_CLEAN_SECOND_FOOT"
        rows.append(
            {
                "arm": row["arm"],
                "scene": row["scene"],
                "source_report_hash": row["source_report_hash"],
                "incoming_ball_speed_mps": incoming,
                "dynamic_incoming": dynamic,
                "first_receiver_foot_frame": first,
                "clean_second_foot_events": qualifying,
                "label": label,
                "promotion_authorized": False,
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    result = {
        "schema": "rosclaw_soccer.rsi.r1_shared_dynamic_reception_audit_v101.v1",
        "partition": "CONSUMED_FAILURE_REFINEMENT",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "source_v100_report_hash": source["report_hash"],
        "rows": rows,
        "label_counts": counts,
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
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new immutable dynamic reception output required")
    result = audit(args.evidence_root, args.source_report, args.source_protocol, args.protocol)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"label_counts": result["label_counts"], "report_hash": result["report_hash"]})
    )


if __name__ == "__main__":
    main()
