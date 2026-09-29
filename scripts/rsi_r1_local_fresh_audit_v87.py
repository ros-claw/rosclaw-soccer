"""Audit the frozen eight-scene six-G1 local chain without training on it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _read_arm(root: Path, scene_id: str, arm: str) -> dict[str, Any]:
    directory = root / f"rsi-r1-strike-fresh-v87-{scene_id}-{arm}"
    report_file = directory / "report.json"
    protocol_file = directory / "protocol.json"
    trace_file = directory / "trace.npz"
    report = json.loads(report_file.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    stated_hash = report.pop("report_hash")
    if (
        hash_json(report) != stated_hash
        or hash_bytes(protocol_file.read_bytes()) != report["protocol_hash"]
        or hash_bytes(trace_file.read_bytes()) != report["trace_hash"]
        or protocol["partition"] != "CONSUMED_DEV"
        or protocol["activation_ceiling"] != "SIM_ONLY"
        or protocol["promotion_authorized"] is not False
        or protocol["video_authorized"] is not False
        or protocol["scenario"]["scenario_id"] != "s199.rsi.r1.receiver-bridge.consumed"
    ):
        raise ValueError(f"invalid immutable run evidence: {directory}")
    with np.load(trace_file, allow_pickle=False) as trace:
        times = trace["time"]
        crossing = report["crossing"]
        receiver_contact = (report["chain"] or {}).get("receiver_contact_sec")
        crossing_frame = None if crossing is None else int(crossing["frame"])
        code = 4  # sorted six-G1 roster: red.finisher
        foot = trace["ball_contact_agent_code"] == code
        foot &= np.isin(trace["ball_contact_foot_code"], (1, 2))
        candidate_shots = (
            []
            if receiver_contact is None or crossing_frame is None
            else [
                int(frame)
                for frame in np.flatnonzero(foot)
                if float(times[frame]) >= float(receiver_contact) + 0.20
                and frame < crossing_frame
                and float(np.linalg.norm(trace["ball_velocity"][frame, :3])) >= 2.0
            ]
        )
        nonfoot = bool(np.any(trace["ball_nonfoot_contact_agent_code"] != 0))
        minimum_margin = float(np.min(trace["red_finisher_joint_safety_margin_rad"]))
        shot_frame = candidate_shots[0] if candidate_shots else None
        chain_pass = bool(
            report["chain_success"]
            and report["result"]["safe"]
            and not nonfoot
            and shot_frame is not None
        )
        return {
            "scene_id": scene_id,
            "arm": arm,
            "report_hash": stated_hash,
            "protocol_hash": report["protocol_hash"],
            "trace_hash": report["trace_hash"],
            "safe": bool(report["result"]["safe"]),
            "clean_transfer": bool((report["chain"] or {}).get("clean_transfer_observed")),
            "nonfoot_ball_contact": nonfoot,
            "shot_frame": shot_frame,
            "shot_time_sec": None if shot_frame is None else float(times[shot_frame]),
            "minimum_finisher_joint_margin_rad": minimum_margin,
            "crossing": crossing,
            "chain_pass": chain_pass,
        }


def audit(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["schema"] != "rosclaw_soccer.rsi.r1_strike_local_fresh_v87.protocol.v1":
        raise ValueError("frozen local exam required")
    scenes = protocol["scenes"]
    if len(scenes) != 8 or len({scene["id"] for scene in scenes}) != 8:
        raise ValueError("exactly eight distinct frozen scenes required")
    rows = []
    for scene in scenes:
        pair = {arm: _read_arm(root, scene["id"], arm) for arm in ("parent", "candidate")}
        for arm, record in pair.items():
            run_protocol = json.loads(
                (root / f"rsi-r1-strike-fresh-v87-{scene['id']}-{arm}" / "protocol.json").read_text(
                    encoding="utf-8"
                )
            )
            fixture = run_protocol["scenario"]
            if (
                fixture["ball_initial_position_m"][:2] != [scene["ball_x_m"], scene["ball_y_m"]]
                or fixture["seed"] != scene["seed"]
                or run_protocol["stance_lateral_m"]
                != protocol["paired_parent" if arm == "parent" else "frozen_candidate"][
                    "strike_stance_lateral_m"
                ]
                or run_protocol["handoff_profile"] != "tracking"
                or run_protocol["directed_pass_speed_mps"] != 1.0
            ):
                raise ValueError("run does not match the frozen paired scene")
            record["run_protocol_hash"] = hash_bytes(
                (
                    root / f"rsi-r1-strike-fresh-v87-{scene['id']}-{arm}" / "protocol.json"
                ).read_bytes()
            )
        rows.append({"scene": scene, **pair})
    candidate = [row["candidate"] for row in rows]
    parent = [row["parent"] for row in rows]
    candidate_pass = sum(row["chain_pass"] for row in candidate)
    parent_pass = sum(row["chain_pass"] for row in parent)
    gates = {
        "all_candidate_six_body_safe": all(row["safe"] for row in candidate),
        "at_least_six_candidate_chains": candidate_pass >= 6,
        "at_least_two_more_chains_than_parent": candidate_pass >= parent_pass + 2,
        "no_parent_success_lost": all(
            not p["chain_pass"] or c["chain_pass"] for p, c in zip(parent, candidate, strict=True)
        ),
        "all_evidence_integrity_verified": True,
    }
    result = {
        "schema": "rosclaw_soccer.rsi.r1_strike_local_fresh_v87.audit.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "status": "PASS_LOCAL_ONLY" if all(gates.values()) else "REJECTED_LOCAL",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "video_authorized": False,
        "candidate_chain_count": candidate_pass,
        "parent_chain_count": parent_pass,
        "gates": gates,
        "rows": rows,
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
        raise ValueError("new immutable audit path required")
    result = audit(args.evidence_root, args.protocol)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "candidate_chain_count", "parent_chain_count", "report_hash")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
