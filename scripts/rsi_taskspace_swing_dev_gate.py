"""Fail-closed paired SIM_ONLY development gate for G1 task-space swing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.baseline_retention_phase import load_guarded_phase_actor
from rosclaw_soccer.rsi.first_touch_snapshot_bank import audit_snapshot_bank
from rosclaw_soccer.rsi.snapshot_replay_evidence import audit_snapshot_replay
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_contextual_phase_train import lane_outcomes

PROTOCOL = Path("docs/rsi/protocols/first-touch-taskspace-swing-v7.json")
CAPS = (0.08, 0.16)


def _outcome(folder: Path) -> dict[str, Any]:
    reward, clean = lane_outcomes(folder, 8)
    with np.load(folder / "replay.npz", allow_pickle=False) as trace:
        minimum_root = float(np.min(trace["observed_root_pose_local_xyzw_m"][:, :, 2]))
    return {
        "clean_foot_count": int(np.count_nonzero(clean)),
        "clean_foot_by_lane": clean.astype(int).tolist(),
        "mean_reward": float(np.mean(reward)),
        "minimum_root_height_m": minimum_root,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True, type=Path)
    parser.add_argument("--set", dest="sets", nargs=4, action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or len(args.sets) != 2:
        parser.error("two independent development seeds and fresh output required")
    actor = load_guarded_phase_actor(args.actor)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if actor["actor_hash"] != protocol["frozen_parent_actor_hash"]:
        raise ValueError("task-space probe actor differs from pre-registered parent")
    seeds = []
    for bank_raw, parent_raw, *candidate_raw in args.sets:
        bank, parent = Path(bank_raw), Path(parent_raw)
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        bank_audit = audit_snapshot_bank(bank)
        if manifest.get("partition") != "SEALED_HOLDOUT" or manifest["snapshot_count"] != 8:
            raise ValueError("task-space development requires consumed v5 eight-lane source")
        parent_audit = audit_snapshot_replay(
            parent, snapshot_bank=bank, local_phase_policy_path=args.actor
        )
        if not parent_audit["intervention_action_audited"]:
            raise ValueError("frozen v5 parent action not audited")
        candidates = []
        for cap, raw in zip(CAPS, candidate_raw, strict=True):
            folder = Path(raw)
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            audit = audit_snapshot_replay(
                folder, snapshot_bank=bank, local_phase_policy_path=args.actor
            )
            if report.get("taskspace_forward_m") != cap or not audit.get(
                "taskspace_action_audited"
            ):
                raise ValueError("unverified task-space physics action")
            candidates.append(
                {"forward_cap_m": cap, "audit_hash": audit["report_hash"], **_outcome(folder)}
            )
        seeds.append(
            {
                "bank_hash": bank_audit["manifest_hash"],
                "parent_audit_hash": parent_audit["report_hash"],
                "parent": _outcome(parent),
                "candidates": candidates,
            }
        )
    if seeds[0]["bank_hash"] == seeds[1]["bank_hash"]:
        raise ValueError("development seeds are not independent")
    scores = []
    for index, cap in enumerate(CAPS):
        gains = [
            row["candidates"][index]["clean_foot_count"] - row["parent"]["clean_foot_count"]
            for row in seeds
        ]
        reward_gain = sum(
            row["candidates"][index]["mean_reward"] - row["parent"]["mean_reward"] for row in seeds
        )
        safe = all(row["candidates"][index]["minimum_root_height_m"] >= 0.65 for row in seeds)
        scores.append(
            {
                "forward_cap_m": cap,
                "per_seed_clean_gains": gains,
                "total_clean_gain": sum(gains),
                "total_mean_reward_gain": reward_gain,
                "short_window_safe": safe,
            }
        )
    best = max(
        scores,
        key=lambda row: (
            row["short_window_safe"],
            min(row["per_seed_clean_gains"]),
            row["total_clean_gain"],
            row["total_mean_reward_gain"],
        ),
    )
    holdout_open = (
        best["short_window_safe"]
        and min(best["per_seed_clean_gains"]) >= 0
        and best["total_clean_gain"] >= 2
    )
    result = {
        "schema": "rsi_taskspace_swing_development_gate_v1",
        "activation_ceiling": "SIM_ONLY",
        "actor_hash": actor["actor_hash"],
        "protocol_hash": hash_bytes(PROTOCOL.read_bytes()),
        "seeds": seeds,
        "scores": scores,
        "selected_forward_cap_m": best["forward_cap_m"] if holdout_open else None,
        "fresh_holdout_open_authorized": bool(holdout_open),
        "promotion_authorized": False,
    }
    result["verdict_hash"] = hash_json(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "seeds"}))


if __name__ == "__main__":
    main()
