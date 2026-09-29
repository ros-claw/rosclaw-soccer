"""Mine authenticated, consumed six-G1 receive-contact failure mechanisms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_velocity_effects import receiving_velocity_effects


def _episode(folder: Path, scene: dict[str, Any], name: str) -> dict[str, Any]:
    protocol_path = folder / "protocol.json"
    report_path = folder / "report.json"
    trace_path = folder / "trace.npz"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in report.items() if key != "report_hash"}
    identities = sorted(row["agent_id"] for row in report["result"]["qualities"])
    if (
        report.get("report_hash") != hash_json(body)
        or report["protocol_hash"] != hash_bytes(protocol_path.read_bytes())
        or report["trace_hash"] != hash_bytes(trace_path.read_bytes())
        or protocol["scenario"]["ball_initial_position_m"][:2]
        != [scene["ball_x_m"], scene["ball_y_m"]]
        or protocol["scenario"]["seed"] != scene["seed"]
        or protocol["activation_ceiling"] != "SIM_ONLY"
        or protocol["promotion_authorized"] is not False
        or protocol["video_authorized"] is not False
        or len(identities) != 6
        or identities.index("red.finisher") + 1 != 4
    ):
        raise ValueError(f"unbound physical receive evidence: {folder}")
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    contact = trace["ball_contact_agent_code"]
    foot = trace["ball_contact_foot_code"]
    own = np.flatnonzero((contact == 4) & np.isin(foot, (1, 2)))
    first = int(own[0]) if len(own) else None
    diagnostic = None
    if first is not None:
        diagnostic = receiving_velocity_effects(
            trace, agent_id="red.finisher", agent_code=4, tail=slice(first + 15, first + 25)
        )
        if diagnostic["first_foot_frame"] != first:
            raise ValueError("receiver foot window changed during diagnosis")
    later = (
        []
        if first is None
        else [
            int(index)
            for index in own
            if float(trace["time"][index]) >= float(trace["time"][first]) + 0.20
            and float(np.linalg.norm(trace["ball_velocity"][index, :3])) >= 2.0
        ]
    )
    if not bool(report["result"]["safe"]):
        mechanism = "unsafe_support"
    elif first is None:
        mechanism = "no_receiver_foot"
    elif diagnostic is not None and (
        abs(diagnostic["outgoing_lateral_mps"]) > 0.50
        or diagnostic["outgoing_ball_speed_mps"] > diagnostic["incoming_ball_speed_mps"]
    ):
        mechanism = "off_axis_rebound"
    elif diagnostic is not None and diagnostic["tail_maximum_foot_distance_m"] > 0.35:
        mechanism = "rapid_foot_separation"
    else:
        mechanism = "successor_not_ready"
    return {
        "arm": name,
        "scene": scene["id"],
        "source_report_hash": report["report_hash"],
        "source_protocol_hash": report["protocol_hash"],
        "source_trace_hash": report["trace_hash"],
        "source_hashes": protocol["source_hashes"],
        "safe": bool(report["result"]["safe"]),
        "clean_transfer": bool((report["chain"] or {}).get("clean_transfer_observed")),
        "first_receiver_foot_frame": first,
        "later_receiver_foot_shot_count": len(later),
        "contact_diagnostic": diagnostic,
        "primary_failure_mechanism": mechanism,
        "promotion_authorized": False,
    }


def mine(evidence_root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.r1_shared_contact_mechanism_v100.protocol.v1"
        or protocol["partition"] != "CONSUMED_FAILURE_DIAGNOSIS"
        or protocol["learning_authorized"] is not False
    ):
        raise ValueError("frozen consumed-only mechanism protocol required")
    arms = protocol["arms"]
    scenes = protocol["scenes"]
    if len(arms) != 5 or len(scenes) != 3:
        raise ValueError("expected frozen 5x3 physical evidence course")
    rows = [
        _episode(
            evidence_root / arm["folder_pattern"].format(scene=scene["id"]),
            scene,
            arm["name"],
        )
        for arm in arms
        for scene in scenes
    ]
    if len({row["source_report_hash"] for row in rows}) != len(rows):
        raise ValueError("duplicate physical rollout in mechanism map")
    counts: dict[str, int] = {}
    for row in rows:
        key = row["primary_failure_mechanism"]
        counts[key] = counts.get(key, 0) + 1
    result = {
        "schema": "rosclaw_soccer.rsi.r1_shared_contact_mechanism_v100.v1",
        "partition": "CONSUMED_FAILURE_DIAGNOSIS",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "rows": rows,
        "failure_counts": counts,
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
        raise ValueError("new immutable mechanism output required")
    result = mine(args.evidence_root, args.protocol)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"failure_counts": result["failure_counts"], "report_hash": result["report_hash"]}
        )
    )


if __name__ == "__main__":
    main()
