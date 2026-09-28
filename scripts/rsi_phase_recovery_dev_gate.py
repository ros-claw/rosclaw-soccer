"""Audit pre-registered contact-triggered phase recovery without opening holdout."""

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

PROTOCOL = Path("docs/rsi/protocols/first-touch-postcontact-phase-recovery-v6.json")
RECOVERY_FRAMES = (12, 20, 30)


def _outcomes(folder: Path, count: int) -> dict[str, Any]:
    reward, clean = lane_outcomes(folder, count)
    with np.load(folder / "replay.npz", allow_pickle=False) as trace:
        forces = trace["observed_ball_body_contact_force_peak_n"]
        root_height = float(np.min(trace["observed_root_pose_local_xyzw_m"][:, :, 2]))
    first_foot = 0
    for lane in range(count):
        active = np.flatnonzero(np.max(forces[:, lane], axis=1) > 1.0)
        if len(active):
            bodies = set(np.flatnonzero(forces[active[0], lane] > 1.0).tolist())
            first_foot += int(bool(bodies and bodies <= {0, 1}))
    return {
        "clean_foot_count": int(np.count_nonzero(clean)),
        "first_foot_count": first_foot,
        "mean_reward": float(np.mean(reward)),
        "minimum_root_height_m": root_height,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True, type=Path)
    parser.add_argument("--set", dest="sets", nargs=5, action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or len(args.sets) != 2:
        parser.error("two development seeds and fresh immutable output required")
    actor = load_guarded_phase_actor(args.actor)
    if (
        actor.get("actor_hash")
        != json.loads(PROTOCOL.read_text(encoding="utf-8"))["frozen_parent_actor_hash"]
    ):
        raise ValueError("development probe does not use pre-registered frozen actor")
    seed_rows = []
    for bank_raw, parent_raw, *candidates_raw in args.sets:
        bank, parent = Path(bank_raw), Path(parent_raw)
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        bank_audit = audit_snapshot_bank(bank)
        if manifest.get("partition") != "SEALED_HOLDOUT" or manifest["snapshot_count"] != 8:
            raise ValueError("dev probe requires consumed v5 sealed eight-lane source")
        parent_audit = audit_snapshot_replay(
            parent, snapshot_bank=bank, local_phase_policy_path=args.actor
        )
        if not parent_audit["intervention_action_audited"]:
            raise ValueError("frozen parent action unverified")
        base = _outcomes(parent, 8)
        candidate_rows = []
        for frames, raw in zip(RECOVERY_FRAMES, candidates_raw, strict=True):
            folder = Path(raw)
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            audit = audit_snapshot_replay(
                folder, snapshot_bank=bank, local_phase_policy_path=args.actor
            )
            if (
                report.get("phase_recovery_frames") != frames
                or not audit["intervention_action_audited"]
            ):
                raise ValueError("phase recovery action or causal audit missing")
            candidate_rows.append(
                {
                    "recovery_frames": frames,
                    "audit_hash": audit["report_hash"],
                    **_outcomes(folder, 8),
                }
            )
        seed_rows.append(
            {
                "bank_hash": bank_audit["manifest_hash"],
                "parent_audit_hash": parent_audit["report_hash"],
                "parent": base,
                "candidates": candidate_rows,
            }
        )
    if seed_rows[0]["bank_hash"] == seed_rows[1]["bank_hash"]:
        raise ValueError("not independent development courses")
    scores = []
    for index, frames in enumerate(RECOVERY_FRAMES):
        gains = [
            row["candidates"][index]["clean_foot_count"] - row["parent"]["clean_foot_count"]
            for row in seed_rows
        ]
        first_gain = sum(
            row["candidates"][index]["first_foot_count"] - row["parent"]["first_foot_count"]
            for row in seed_rows
        )
        reward_gain = sum(
            row["candidates"][index]["mean_reward"] - row["parent"]["mean_reward"]
            for row in seed_rows
        )
        safe = all(row["candidates"][index]["minimum_root_height_m"] >= 0.65 for row in seed_rows)
        scores.append(
            {
                "recovery_frames": frames,
                "per_seed_clean_gains": gains,
                "total_clean_gain": sum(gains),
                "total_first_foot_gain": first_gain,
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
            row["total_first_foot_gain"],
            row["total_mean_reward_gain"],
        ),
    )
    pass_gate = (
        best["short_window_safe"]
        and min(best["per_seed_clean_gains"]) >= 0
        and best["total_clean_gain"] >= 2
    )
    result = {
        "schema": "rsi_postcontact_phase_recovery_dev_gate_v1",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(PROTOCOL.read_bytes()),
        "frozen_actor_hash": actor["actor_hash"],
        "seed_rows": seed_rows,
        "scores": scores,
        "selected_recovery_frames": best["recovery_frames"] if pass_gate else None,
        "fresh_holdout_open_authorized": bool(pass_gate),
        "promotion_authorized": False,
    }
    result["verdict_hash"] = hash_json(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "seed_rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
