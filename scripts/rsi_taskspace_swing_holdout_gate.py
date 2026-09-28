"""Audit fresh paired SIM_ONLY task-space swing holdout without promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.baseline_retention_phase import load_guarded_phase_actor
from rosclaw_soccer.rsi.first_touch_snapshot_bank import audit_snapshot_bank
from rosclaw_soccer.rsi.snapshot_replay_evidence import audit_snapshot_replay
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_contextual_phase_train import lane_outcomes


def _outcome(folder: Path) -> dict[str, Any]:
    reward, clean = lane_outcomes(folder, 8)
    with np.load(folder / "replay.npz", allow_pickle=False) as trace:
        root_min = float(np.min(trace["observed_root_pose_local_xyzw_m"][:, :, 2]))
    return {
        "clean_foot_count": int(np.count_nonzero(clean)),
        "clean_foot_by_lane": clean.astype(int).tolist(),
        "mean_reward": float(np.mean(reward)),
        "minimum_root_height_m": root_min,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True, type=Path)
    parser.add_argument("--development-verdict", required=True, type=Path)
    parser.add_argument("--set", dest="sets", nargs=3, action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or len(args.sets) != 2:
        parser.error("two fresh sealed seeds and immutable output required")
    actor = load_guarded_phase_actor(args.actor)
    development = json.loads(args.development_verdict.read_text(encoding="utf-8"))
    if (
        development.get("schema") != "rsi_taskspace_swing_development_gate_v1"
        or development.get("actor_hash") != actor["actor_hash"]
        or development.get("fresh_holdout_open_authorized") is not True
        or development.get("selected_forward_cap_m") != 0.08
        or development.get("verdict_hash")
        != hash_json({key: value for key, value in development.items() if key != "verdict_hash"})
    ):
        raise ValueError("unauthenticated development authorization")
    rows = []
    for bank_raw, parent_raw, candidate_raw in args.sets:
        bank, parent, candidate = map(Path, (bank_raw, parent_raw, candidate_raw))
        bank_audit = audit_snapshot_bank(bank)
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("partition") != "SEALED_HOLDOUT" or manifest["snapshot_count"] != 8:
            raise ValueError("not a fresh sealed eight-lane course")
        source = Path(manifest["snapshots"][0]["source_folder"])
        source_report = json.loads((source / "report.json").read_text(encoding="utf-8"))
        seed = source_report.get("training_course_seed")
        if seed not in (20260947, 20260948):
            raise ValueError("not a pre-registered fresh task-space course")
        parent_audit = audit_snapshot_replay(
            parent, snapshot_bank=bank, local_phase_policy_path=args.actor
        )
        candidate_audit = audit_snapshot_replay(
            candidate, snapshot_bank=bank, local_phase_policy_path=args.actor
        )
        candidate_report = json.loads((candidate / "report.json").read_text(encoding="utf-8"))
        if (
            parent_audit["sample_count"] != 8
            or candidate_audit["sample_count"] != 8
            or parent_audit.get("taskspace_action_audited") is not None
            or not candidate_audit.get("taskspace_action_audited")
            or candidate_report.get("taskspace_forward_m") != 0.08
        ):
            raise ValueError("unverified paired task-space sealed comparison")
        rows.append(
            {
                "course_seed": seed,
                "bank_hash": bank_audit["manifest_hash"],
                "parent_audit_hash": parent_audit["report_hash"],
                "candidate_audit_hash": candidate_audit["report_hash"],
                "parent_source_first_frame_equal_count": parent_audit[
                    "first_contact_frame_equal_count"
                ],
                "parent_source_contact_class_equal_count": parent_audit[
                    "contact_body_class_equal_count"
                ],
                "parent": _outcome(parent),
                "candidate": _outcome(candidate),
            }
        )
    if {row["course_seed"] for row in rows} != {20260947, 20260948}:
        raise ValueError("duplicate or missing sealed task-space course")
    gains = [
        row["candidate"]["clean_foot_count"] - row["parent"]["clean_foot_count"] for row in rows
    ]
    retained = sum(
        int(parent and candidate)
        for row in rows
        for parent, candidate in zip(
            row["parent"]["clean_foot_by_lane"],
            row["candidate"]["clean_foot_by_lane"],
            strict=True,
        )
    )
    parent_success = sum(row["parent"]["clean_foot_count"] for row in rows)
    pass_gate = (
        sum(gains) >= 4
        and min(gains) >= 0
        and all(row["candidate"]["minimum_root_height_m"] >= 0.65 for row in rows)
    )
    result = {
        "schema": "rsi_taskspace_swing_two_seed_sealed_holdout_verdict_v1",
        "activation_ceiling": "SIM_ONLY",
        "actor_hash": actor["actor_hash"],
        "development_verdict_hash": development["verdict_hash"],
        "selected_forward_cap_m": 0.08,
        "rows": rows,
        "per_seed_clean_foot_gains": gains,
        "combined_clean_foot_gain": sum(gains),
        "retained_parent_successes": retained,
        "total_parent_successes": parent_success,
        "verdict": "PASS_SHORT_WINDOW_ONLY" if pass_gate else "REJECTED",
        "full_episode_and_team_match_required_for_promotion": True,
        "promotion_authorized": False,
    }
    result["verdict_hash"] = hash_json(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
