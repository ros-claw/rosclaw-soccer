"""Fail-closed sealed-course verdict for a SIM_ONLY contextual phase actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.contextual_phase_policy import load_phase_actor
from rosclaw_soccer.rsi.first_touch_snapshot_bank import audit_snapshot_bank
from rosclaw_soccer.rsi.snapshot_replay_evidence import audit_snapshot_replay
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_contextual_phase_train import lane_outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--actor-replay", required=True, type=Path)
    parser.add_argument("--actor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("immutable sealed-course verdict already exists")
    bank_audit = audit_snapshot_bank(args.bank)
    bank_manifest = json.loads((args.bank / "manifest.json").read_text(encoding="utf-8"))
    actor_manifest = json.loads(args.actor.read_text(encoding="utf-8"))
    actor_hash, _ = load_phase_actor(args.actor)
    if (
        bank_manifest.get("partition") != "SEALED_HOLDOUT"
        or actor_manifest.get("holdout_open_authorized") is not True
        or actor_manifest.get("actor_hash") != actor_hash
    ):
        raise ValueError("holdout not sealed or development gate not open")
    parent_audit = audit_snapshot_replay(args.parent, snapshot_bank=args.bank)
    actor_audit = audit_snapshot_replay(
        args.actor_replay, snapshot_bank=args.bank, phase_policy_path=args.actor
    )
    count = bank_manifest["snapshot_count"]
    if (
        not parent_audit["parent_replay_equivalent"]
        or not actor_audit["intervention_action_audited"]
        or actor_audit["contextual_phase_actor_hash"] != actor_hash
        or parent_audit["sample_count"] != count
        or actor_audit["sample_count"] != count
    ):
        raise ValueError("unauthenticated paired sealed-course execution")
    parent_reward, parent_clean = lane_outcomes(args.parent, count)
    actor_reward, actor_clean = lane_outcomes(args.actor_replay, count)
    parent_count = int(np.count_nonzero(parent_clean))
    actor_count = int(np.count_nonzero(actor_clean))
    parent_mean = float(np.mean(parent_reward))
    actor_mean = float(np.mean(actor_reward))
    pass_gate = actor_count >= parent_count + 2 and actor_mean > parent_mean
    result = {
        "schema": "rsi_contextual_phase_sealed_holdout_verdict_v1",
        "activation_ceiling": "SIM_ONLY",
        "bank_manifest_hash": bank_audit["manifest_hash"],
        "actor_hash": actor_hash,
        "parent_audit_hash": parent_audit["report_hash"],
        "actor_audit_hash": actor_audit["report_hash"],
        "sample_count": count,
        "parent_clean_foot_count": parent_count,
        "actor_clean_foot_count": actor_count,
        "parent_mean_reward": parent_mean,
        "actor_mean_reward": actor_mean,
        "gain_clean_foot_count": actor_count - parent_count,
        "verdict": "PASS_SHORT_WINDOW_ONLY" if pass_gate else "REJECTED",
        "full_episode_required_for_skill_promotion": True,
        "promotion_authorized": False,
    }
    result["verdict_hash"] = hash_json(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
