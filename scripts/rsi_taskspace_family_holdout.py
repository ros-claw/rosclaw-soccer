"""Fail-closed verdict for frozen task-space family on fresh paired G1 physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.first_touch_snapshot_bank import audit_snapshot_bank
from rosclaw_soccer.rsi.snapshot_replay_evidence import audit_snapshot_replay
from rosclaw_soccer.rsi.taskspace_family_memory import load_taskspace_family_actor
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_contextual_phase_train import lane_outcomes

PROTOCOL = Path("docs/rsi/protocols/first-touch-taskspace-action-family-expansion-v9b.json")


def score_pair(
    parent_clean: np.ndarray[Any, Any], candidate_clean: np.ndarray[Any, Any]
) -> dict[str, Any]:
    if (
        parent_clean.ndim != 1
        or candidate_clean.shape != parent_clean.shape
        or len(parent_clean) < 2
        or not np.isin(parent_clean, (0, 1)).all()
        or not np.isin(candidate_clean, (0, 1)).all()
    ):
        raise ValueError("invalid paired first-touch labels")
    rescued = np.flatnonzero((parent_clean == 0) & (candidate_clean == 1)).tolist()
    regressed = np.flatnonzero((parent_clean == 1) & (candidate_clean == 0)).tolist()
    return {
        "parent_clean": int(np.sum(parent_clean)),
        "candidate_clean": int(np.sum(candidate_clean)),
        "rescued_lane_indices": rescued,
        "regressed_lane_indices": regressed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-actor", required=True, type=Path)
    parser.add_argument("--family-actor", required=True, type=Path)
    parser.add_argument("--set", dest="sets", nargs=3, action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or len(args.sets) != 2:
        parser.error("two fresh paired holdout groups and new output required")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    actor = load_taskspace_family_actor(args.family_actor)
    if actor["protocol_hash"] != hash_json(protocol):
        raise ValueError("actor protocol does not match sealed holdout gate")
    rows = []
    seeds = []
    for bank_raw, parent_raw, candidate_raw in args.sets:
        bank, parent, candidate = map(Path, (bank_raw, parent_raw, candidate_raw))
        bank_audit = audit_snapshot_bank(bank)
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        source = Path(manifest["snapshots"][0]["source_folder"])
        seed = json.loads((source / "report.json").read_text(encoding="utf-8")).get(
            "training_course_seed"
        )
        count = manifest["snapshot_count"]
        if (
            seed in seeds
            or seed not in protocol["new_sealed_holdout_seeds_if_development_gate_passes"]
            or manifest.get("partition") != "SEALED_HOLDOUT"
            or not 2 <= count <= 16
        ):
            raise ValueError("unregistered or reused fresh holdout")
        seeds.append(seed)
        parent_audit = audit_snapshot_replay(
            parent, snapshot_bank=bank, local_phase_policy_path=args.phase_actor
        )
        candidate_audit = audit_snapshot_replay(
            candidate,
            snapshot_bank=bank,
            local_phase_policy_path=args.phase_actor,
            taskspace_family_policy_path=args.family_actor,
        )
        if (
            parent_audit["sample_count"] != count
            or candidate_audit["sample_count"] != count
            or parent_audit.get("taskspace_action_audited") is not None
            or not candidate_audit.get("taskspace_action_audited")
            or candidate_audit.get("taskspace_family_actor_hash") != actor["actor_hash"]
        ):
            raise ValueError("incomplete paired physical action audit")
        parent_reward, parent_clean = lane_outcomes(parent, count)
        candidate_reward, candidate_clean = lane_outcomes(candidate, count)
        with np.load(candidate / "replay.npz", allow_pickle=False) as replay:
            min_root = float(np.min(replay["observed_root_pose_local_xyzw_m"][:, :, 2]))
        row = score_pair(parent_clean, candidate_clean)
        row.update(
            seed=seed,
            sample_count=count,
            parent_mean_reward=float(np.mean(parent_reward)),
            candidate_mean_reward=float(np.mean(candidate_reward)),
            minimum_root_height_m=min_root,
            bank_hash=bank_audit["manifest_hash"],
            parent_audit_hash=parent_audit["report_hash"],
            candidate_audit_hash=candidate_audit["report_hash"],
        )
        rows.append(row)
    if set(seeds) != set(protocol["new_sealed_holdout_seeds_if_development_gate_passes"]):
        raise ValueError("missing preregistered fresh course seed")
    gate = protocol["holdout_gate"]
    parent_total = sum(row["parent_clean"] for row in rows)
    candidate_total = sum(row["candidate_clean"] for row in rows)
    passed = bool(
        candidate_total - parent_total >= gate["minimum_combined_clean_foot_gain_over_parent"]
        and all(row["candidate_clean"] >= row["parent_clean"] for row in rows)
        and all(not row["regressed_lane_indices"] for row in rows)
        and all(row["minimum_root_height_m"] >= gate["minimum_root_height_m"] for row in rows)
    )
    result = {
        "schema": "rsi_taskspace_family_holdout_verdict_v9b",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_json(protocol),
        "actor_hash": actor["actor_hash"],
        "cross_validation_report_hash": actor["cross_validation_report_hash"],
        "rows": rows,
        "parent_clean_total": parent_total,
        "candidate_clean_total": candidate_total,
        "clean_gain": candidate_total - parent_total,
        "verdict": "PASSED_FIRST_TOUCH_HOLDOUT_ONLY" if passed else "REJECTED",
        "full_episode_and_team_match_validated": False,
        "promotion_authorized": False,
        "real_robot_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "report_hash": result["report_hash"]}))


if __name__ == "__main__":
    main()
