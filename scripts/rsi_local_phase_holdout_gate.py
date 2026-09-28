"""Fail-closed paired two-seed sealed Isaac verdict for local phase memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.baseline_retention_phase import load_guarded_phase_actor
from rosclaw_soccer.rsi.first_touch_snapshot_bank import audit_snapshot_bank
from rosclaw_soccer.rsi.local_phase_memory import load_local_phase_actor
from rosclaw_soccer.rsi.snapshot_replay_evidence import audit_snapshot_replay
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_contextual_phase_train import lane_outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="sets", nargs=3, action="append", required=True)
    parser.add_argument("--actor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or len(args.sets) != 2:
        parser.error("two fresh sealed sets and immutable verdict output required")
    actor_schema = json.loads(args.actor.read_text(encoding="utf-8")).get("schema")
    actor = (
        load_guarded_phase_actor(args.actor)
        if actor_schema == "rsi_baseline_retention_phase_actor_v5"
        else load_local_phase_actor(args.actor)
    )
    if actor.get("holdout_open_authorized") is not True:
        raise ValueError("development gate never opened sealed local phase holdout")
    rows = []
    for bank_raw, parent_raw, candidate_raw in args.sets:
        bank_path, parent, candidate = map(Path, (bank_raw, parent_raw, candidate_raw))
        bank_audit = audit_snapshot_bank(bank_path)
        manifest = json.loads((bank_path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("partition") != "SEALED_HOLDOUT":
            raise ValueError("not a sealed physical course")
        parent_audit = audit_snapshot_replay(parent, snapshot_bank=bank_path)
        candidate_audit = audit_snapshot_replay(
            candidate, snapshot_bank=bank_path, local_phase_policy_path=args.actor
        )
        count = manifest["snapshot_count"]
        if (
            parent_audit["sample_count"] != count
            or candidate_audit["sample_count"] != count
            or parent_audit["intervention_action_audited"]
            or parent_audit["contact_body_class_equal_count"] != count
            or parent_audit["first_contact_frame_equal_count"] < count - 1
            or not candidate_audit["intervention_action_audited"]
            or candidate_audit["local_phase_actor_hash"] != actor["actor_hash"]
        ):
            raise ValueError("unauthenticated or mismatched physical sealed comparison")
        base_reward, base_clean = lane_outcomes(parent, count)
        actor_reward, actor_clean = lane_outcomes(candidate, count)
        with (
            np.load(parent / "replay.npz", allow_pickle=False) as parent_trace,
            np.load(candidate / "replay.npz", allow_pickle=False) as actor_trace,
        ):
            parent_root_min = float(
                np.min(parent_trace["observed_root_pose_local_xyzw_m"][:, :, 2])
            )
            actor_root_min = float(np.min(actor_trace["observed_root_pose_local_xyzw_m"][:, :, 2]))
        rows.append(
            {
                "bank_hash": bank_audit["manifest_hash"],
                "parent_audit_hash": parent_audit["report_hash"],
                "actor_audit_hash": candidate_audit["report_hash"],
                "sample_count": count,
                "parent_clean_foot_count": int(np.count_nonzero(base_clean)),
                "actor_clean_foot_count": int(np.count_nonzero(actor_clean)),
                "parent_mean_reward": float(np.mean(base_reward)),
                "actor_mean_reward": float(np.mean(actor_reward)),
                "parent_minimum_root_height_m": parent_root_min,
                "actor_minimum_root_height_m": actor_root_min,
            }
        )
    if rows[0]["bank_hash"] == rows[1]["bank_hash"]:
        raise ValueError("sealed course seeds are not independent")
    gains = [row["actor_clean_foot_count"] - row["parent_clean_foot_count"] for row in rows]
    total = sum(row["sample_count"] for row in rows)
    base_mean = sum(row["parent_mean_reward"] * row["sample_count"] for row in rows) / total
    actor_mean = sum(row["actor_mean_reward"] * row["sample_count"] for row in rows) / total
    pass_gate = (
        sum(gains) >= 4
        and min(gains) >= 0
        and actor_mean > base_mean
        and all(row["actor_minimum_root_height_m"] >= 0.65 for row in rows)
    )
    result = {
        "schema": (
            "rsi_baseline_retention_two_seed_sealed_holdout_verdict_v1"
            if actor_schema == "rsi_baseline_retention_phase_actor_v5"
            else "rsi_local_phase_two_seed_sealed_holdout_verdict_v1"
        ),
        "activation_ceiling": "SIM_ONLY",
        "actor_hash": actor["actor_hash"],
        "rows": rows,
        "total_sample_count": total,
        "per_seed_clean_foot_gains": gains,
        "combined_clean_foot_gain": sum(gains),
        "parent_mean_reward": base_mean,
        "actor_mean_reward": actor_mean,
        "verdict": "PASS_SHORT_WINDOW_ONLY" if pass_gate else "REJECTED",
        "full_episode_and_team_match_required_for_promotion": True,
        "promotion_authorized": False,
    }
    result["verdict_hash"] = hash_json(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
