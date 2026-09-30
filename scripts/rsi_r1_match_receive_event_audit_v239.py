"""Read-only event-clock audit on a genuine three-G1 pass/receive/shot chain."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked

from rosclaw_soccer.rsi.receiving_contact_episode_gate import ReceivingContactEpisodeGate
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rosclaw_soccer.rsi.r1_match_receive_event_audit_v239.result.v1"


def audit(parent_dir: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external read-only match event evidence required")
    report = _checked(parent_dir / "report.json")
    archive = parent_dir / "current-parent.npz"
    if (
        report["pass_shot_chain_succeeded"] is not True
        or report["receiver_id"] != "red.finisher"
        or hash_bytes(archive.read_bytes()) != report["trace_hash"]
    ):
        raise ValueError("sealed genuine pass-receive-shot physical chain required")
    ids = tuple(sorted(row["agent_id"] for row in report["result"]["qualities"]))
    code = ids.index("red.finisher") + 1
    with np.load(archive, allow_pickle=False) as trace:
        time = np.asarray(trace["time"], dtype=np.float64)
        ball = np.asarray(trace["ball_pose"][:, :2], dtype=np.float64)
        velocity = np.asarray(trace["ball_velocity"][:, :2], dtype=np.float64)
        pelvis = np.asarray(trace["red_finisher_pelvis_pose"][:, :2], dtype=np.float64)
        own_foot = (trace["ball_contact_agent_code"] == code) & (trace["ball_contact_force_n"] > 0)
        if not all(len(item) == len(time) for item in (ball, velocity, pelvis, own_foot)):
            raise ValueError("aligned same-frame measured physical arrays required")
        pelvis_velocity = np.gradient(pelvis, time, axis=0)
    gate = ReceivingContactEpisodeGate("red.finisher")
    states: list[str] = []
    for frame, (clock, ball_xy, ball_v, pelvis_xy, pelvis_v, touched) in enumerate(
        zip(time, ball, velocity, pelvis, pelvis_velocity, own_foot, strict=True)
    ):
        states.append(
            gate.observe(
                frame=frame,
                time_sec=float(clock),
                ball_relative_position_xy_m=tuple(float(value) for value in ball_xy - pelvis_xy),
                ball_relative_velocity_xy_mps=tuple(float(value) for value in ball_v - pelvis_v),
                own_foot_contact=bool(touched),
            )
        )
    first_foot = int(np.flatnonzero(own_foot)[0])
    matched = [
        episode
        for episode in gate.episodes
        if episode.contact_frame == first_foot
        and episode.outcome == "MEASURED_FOOT_CONTACT_AND_RECOVERY"
    ]
    if len(matched) != 1 or not matched[0].arm_frame < first_foot < matched[0].end_frame:
        raise ValueError("event gate must precede and retire after real receiving contact")
    output.mkdir(parents=True)
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_match_receive_event_audit_v239.py",
            "src/rosclaw_soccer/rsi/receiving_contact_episode_gate.py",
        )
    }
    result = {
        "schema": SCHEMA,
        "parent_report_hash": report["report_hash"],
        "parent_trace_hash": report["trace_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_REAL_PASS_CHAIN_READ_ONLY_CLOCK_AUDIT",
        "gate_contract_hash": gate.contract_hash,
        "agent_id": gate.agent_id,
        "first_own_foot_frame": first_foot,
        "first_own_foot_time_sec": float(time[first_foot]),
        "episodes": [asdict(episode) for episode in gate.episodes],
        "matching_contact_episode": asdict(matched[0]),
        "state_counts": {state: states.count(state) for state in sorted(set(states))},
        "status": "DEVELOPMENT_MATCH_EVENT_CLOCK_VERIFIED_ONLY",
        "motor_activated": False,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "READ_ONLY_SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during read-only event audit")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.parent_dir, args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "matching_contact_episode",
                    "report_hash",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
