"""Fail-closed paired full-episode first-touch transfer verdict; SIM_ONLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.full_episode_late_swing_evidence import audit_full_episode_late_swing
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def clean_foot_lanes(report: dict[str, Any]) -> set[int]:
    return {
        index
        for index, row in enumerate(report["environments"])
        if row["course"]["ball_vx_m_s"] < 0
        and row["contact_body_indices"]
        and set(row["contact_body_indices"]) <= {0, 1}
    }


def paired_verdict(
    protocol_path: Path,
    actor_path: Path,
    parents: tuple[Path, Path],
    candidates: tuple[Path, Path],
    *,
    split: str,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seeds = protocol["course_seed_split"][
        "consumed_development"
        if split == "development"
        else "sealed_fresh_if_development_gate_passes"
    ]
    if (
        protocol.get("schema") != "rsi_first_touch_full_episode_transfer_protocol_v11b"
        or split not in ("development", "fresh")
        or len(seeds) != 2
        or len(parents) != 2
        or len(candidates) != 2
        or json.loads(actor_path.read_text(encoding="utf-8")).get("actor_hash")
        != protocol["frozen_late_swing_actor_hash"]
    ):
        raise ValueError("uncommitted full-episode protocol or actor")
    rows = []
    total_gain = 0
    accepted = True
    for seed, parent_folder, candidate_folder in zip(seeds, parents, candidates, strict=True):
        audit = audit_full_episode_late_swing(candidate_folder, actor_path, parent_folder)
        parent = json.loads((parent_folder / "report.json").read_text(encoding="utf-8"))
        candidate = json.loads((candidate_folder / "report.json").read_text(encoding="utf-8"))
        if (
            parent.get("training_course_seed") != seed
            or candidate.get("training_course_seed") != seed
            or candidate.get("source_hash")
            != hash_bytes(Path(__file__).with_name("rsi_isaac_vector_first_touch.py").read_bytes())
        ):
            raise ValueError("source seed or full-episode runner changed")
        old = clean_foot_lanes(parent)
        new = clean_foot_lanes(candidate)
        if len([row for row in parent["environments"] if row["course"]["ball_vx_m_s"] < 0]) != 8:
            raise ValueError("expected eight incoming lanes")
        gain = len(new) - len(old)
        total_gain += gain
        minimum_pelvis = min(row["minimum_pelvis_z_m"] for row in candidate["environments"])
        accepted &= old <= new and gain >= 0 and minimum_pelvis >= 0.65
        with np.load(candidate_folder / "body_trace.npz", allow_pickle=False) as body:
            postcontact_root_speed = []
            for lane, row in enumerate(candidate["environments"]):
                first = row["first_contact_frame"]
                if first is not None:
                    postcontact_root_speed.append(
                        float(
                            np.mean(
                                np.linalg.norm(
                                    body["root_velocity_world"][first:, lane, :3], axis=1
                                )
                            )
                        )
                    )
        rows.append(
            {
                "seed": seed,
                "parent_report_hash": parent["report_hash"],
                "candidate_report_hash": candidate["report_hash"],
                "action_audit": audit,
                "parent_incoming_clean_lanes": sorted(old),
                "candidate_incoming_clean_lanes": sorted(new),
                "rescued_lanes": sorted(new - old),
                "lost_lanes": sorted(old - new),
                "incoming_clean_gain": gain,
                "minimum_pelvis_z_m": minimum_pelvis,
                "minimum_cross_robot_distance_m": candidate["minimum_cross_robot_distance_m"],
                "minimum_cross_ball_distance_m": candidate["minimum_cross_ball_distance_m"],
                "mean_postcontact_root_speed_m_s": float(np.mean(postcontact_root_speed)),
            }
        )
    accepted &= total_gain >= 4
    verdict = {
        "schema": "rsi_full_episode_late_swing_verdict_v11b",
        "activation_ceiling": "SIM_ONLY",
        "split": split,
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "late_swing_actor_hash": protocol["frozen_late_swing_actor_hash"],
        "paired_seeds": rows,
        "combined_incoming_clean_gain": total_gain,
        "verdict": "PASSED_FULL_EPISODE_FIRST_TOUCH_ONLY" if accepted else "REJECTED",
        "team_match_validated": False,
        "publicity_ready": False,
        "promotion_authorized": False,
    }
    verdict["verdict_hash"] = hash_json(verdict)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--actor", required=True, type=Path)
    parser.add_argument("--parents", required=True, nargs=2, type=Path)
    parser.add_argument("--candidates", required=True, nargs=2, type=Path)
    parser.add_argument("--split", required=True, choices=("development", "fresh"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("verdict output already exists")
    verdict = paired_verdict(
        args.protocol, args.actor, tuple(args.parents), tuple(args.candidates), split=args.split
    )
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verdict, sort_keys=True))


if __name__ == "__main__":
    main()
