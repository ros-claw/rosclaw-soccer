"""Train a fail-closed late-swing intervention gate from audited paired physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import contact_time_phase_features as feature_module
from rosclaw_soccer.rsi import late_swing_memory as late_module
from rosclaw_soccer.rsi import taskspace_gate_memory as gate_module
from rosclaw_soccer.rsi.baseline_retention_phase import load_guarded_phase_actor
from rosclaw_soccer.rsi.contact_time_phase_features import (
    current_context,
    gait_phase_features,
    predict_contact_time,
)
from rosclaw_soccer.rsi.first_touch_snapshot_bank import audit_snapshot_bank
from rosclaw_soccer.rsi.snapshot_replay_evidence import audit_snapshot_replay
from rosclaw_soccer.rsi.taskspace_gate_memory import (
    BASELINE_CLEAN_CEILINGS,
    CONFIDENCE_MULTIPLIERS,
    NEIGHBOR_COUNTS,
    select_taskspace_gate,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_contextual_phase_train import lane_outcomes

PROTOCOL = Path("docs/rsi/protocols/first-touch-late-swing-memory-v10b.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-actor", required=True, type=Path)
    parser.add_argument("--set", dest="sets", nargs=3, action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--actor-output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.actor_output.exists() or len(args.sets) != 10:
        parser.error("ten independent paired development groups and fresh outputs required")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    phase_actor = load_guarded_phase_actor(args.phase_actor)
    if phase_actor["actor_hash"] != protocol["frozen_parent_actor_hash"]:
        raise ValueError("unregistered frozen SONIC phase parent")
    time_weights = np.asarray(phase_actor["contact_time_weights"], dtype=np.float64)
    features, clean, rewards, groups, evidence = [], [], [], [], []
    seeds = []
    for bank_raw, parent_raw, late_raw in args.sets:
        bank, parent, late = map(Path, (bank_raw, parent_raw, late_raw))
        bank_audit = audit_snapshot_bank(bank)
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        source = Path(manifest["snapshots"][0]["source_folder"])
        seed = json.loads((source / "report.json").read_text(encoding="utf-8")).get(
            "training_course_seed"
        )
        count = manifest["snapshot_count"]
        if (
            manifest.get("partition") != "SEALED_HOLDOUT"
            or count != (6 if seed == 20260953 else 8)
            or seed in seeds
            or seed not in protocol["paired_development_course_seeds"]
        ):
            raise ValueError("unregistered paired development source")
        seeds.append(seed)
        parent_audit = audit_snapshot_replay(
            parent, snapshot_bank=bank, local_phase_policy_path=args.phase_actor
        )
        late_audit = audit_snapshot_replay(
            late, snapshot_bank=bank, local_phase_policy_path=args.phase_actor
        )
        report = json.loads((late / "report.json").read_text(encoding="utf-8"))
        if (
            parent_audit["sample_count"] != count
            or late_audit["sample_count"] != count
            or parent_audit.get("taskspace_action_audited") is not None
            or not late_audit.get("taskspace_action_audited")
            or report.get("taskspace_forward_m") != 0.08
            or report.get("taskspace_lateral_cap_m") != 0.05
            or report.get("taskspace_vertical_offset_m") != 0.04
            or report.get("taskspace_acquisition_max_gap_m") != 0.55
        ):
            raise ValueError("unverified paired late-swing physics")
        with np.load(bank / "snapshots.npz", allow_pickle=False) as snapshot:
            raw = current_context(
                snapshot["root_pose_local_xyzw_m"],
                snapshot["root_velocity_world"],
                snapshot["ball_position_local_m"],
                snapshot["ball_linear_velocity_m_s"],
            )
        features.append(gait_phase_features(raw, predict_contact_time(raw, time_weights)))
        parent_reward, parent_clean = lane_outcomes(parent, count)
        late_reward, late_clean = lane_outcomes(late, count)
        rewards.append(np.column_stack((parent_reward, late_reward)))
        clean.append(np.column_stack((parent_clean, late_clean)).astype(np.float64))
        groups.extend([seed] * count)
        evidence.append(
            {
                "seed": seed,
                "count": count,
                "bank_hash": bank_audit["manifest_hash"],
                "parent_audit_hash": parent_audit["report_hash"],
                "late_audit_hash": late_audit["report_hash"],
            }
        )
    if set(seeds) != set(protocol["paired_development_course_seeds"]):
        raise ValueError("missing pre-registered development source")
    x = np.concatenate(features)
    clean_array = np.concatenate(clean)
    reward_array = np.concatenate(rewards)
    group = np.asarray(groups)
    candidates: list[dict[str, Any]] = []
    for neighbors in NEIGHBOR_COUNTS:
        for confidence in CONFIDENCE_MULTIPLIERS:
            for ceiling in BASELINE_CLEAN_CEILINGS:
                choice = np.zeros(len(x), dtype=np.bool_)
                for held_seed in seeds:
                    held = group == held_seed
                    choice[held] = select_taskspace_gate(
                        x[held],
                        x[~held],
                        clean_array[~held],
                        reward_array[~held],
                        group[~held],
                        neighbors=neighbors,
                        confidence=confidence,
                        baseline_clean_ceiling=ceiling,
                    )
                chosen_clean = clean_array[np.arange(len(x)), choice.astype(int)]
                chosen_reward = reward_array[np.arange(len(x)), choice.astype(int)]
                per_seed = [
                    {
                        "seed": seed,
                        "parent_clean": int(np.sum(clean_array[group == seed, 0])),
                        "candidate_clean": int(np.sum(chosen_clean[group == seed])),
                        "interventions": int(np.count_nonzero(choice[group == seed])),
                    }
                    for seed in seeds
                ]
                gate = protocol["development_gate"]
                candidate: dict[str, Any] = {
                    "neighbors": neighbors,
                    "confidence": confidence,
                    "baseline_clean_ceiling": ceiling,
                    "per_seed": per_seed,
                    "parent_clean": int(np.sum(clean_array[:, 0])),
                    "candidate_clean": int(np.sum(chosen_clean)),
                    "parent_success_regressions": int(
                        np.count_nonzero((clean_array[:, 0] == 1) & (chosen_clean == 0))
                    ),
                    "parent_mean_reward": float(np.mean(reward_array[:, 0])),
                    "candidate_mean_reward": float(np.mean(chosen_reward)),
                    "intervention_count": int(np.count_nonzero(choice)),
                }
                candidate["gate_passed"] = bool(
                    candidate["candidate_clean"] - candidate["parent_clean"]
                    >= gate["minimum_combined_clean_foot_gain_over_parent"]
                    and all(row["candidate_clean"] >= row["parent_clean"] for row in per_seed)
                    and candidate["parent_success_regressions"] == 0
                    and candidate["candidate_mean_reward"] > candidate["parent_mean_reward"]
                )
                candidates.append(candidate)
    qualified = [row for row in candidates if row["gate_passed"]]
    result = {
        "schema": "rsi_late_swing_memory_development_v10b",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_json(protocol),
        "phase_actor_hash": phase_actor["actor_hash"],
        "evidence": evidence,
        "candidates": candidates,
        "development_gate_passed": bool(qualified),
        "fresh_holdout_open_authorized": bool(qualified),
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not qualified:
        print(json.dumps({"gate": False, "report_hash": result["report_hash"]}))
        return
    chosen = max(
        qualified,
        key=lambda row: (
            row["candidate_clean"],
            row["candidate_mean_reward"],
            row["confidence"],
            -row["neighbors"],
        ),
    )
    if (
        gate_module.__file__ is None
        or feature_module.__file__ is None
        or late_module.__file__ is None
    ):
        raise ValueError("unresolved frozen policy source")
    actor = {
        "schema": "rsi_late_swing_memory_actor_v10b",
        "activation_ceiling": "SIM_ONLY",
        "action_forward_m": 0.08,
        "action_vertical_offset_m": 0.04,
        "action_lateral_cap_m": 0.05,
        "action_acquisition_max_gap_m": 0.55,
        "feature_names": list(feature_module.ACTION_FEATURE_NAMES),
        "frozen_phase_actor_hash": phase_actor["actor_hash"],
        "protocol_hash": hash_json(protocol),
        "development_report_hash": result["report_hash"],
        "policy_source_hash": hash_bytes(Path(gate_module.__file__).read_bytes()),
        "loader_source_hash": hash_bytes(Path(late_module.__file__).read_bytes()),
        "feature_source_hash": hash_bytes(Path(feature_module.__file__).read_bytes()),
        "neighbors": chosen["neighbors"],
        "confidence": chosen["confidence"],
        "baseline_clean_ceiling": chosen["baseline_clean_ceiling"],
        "memory_features": x.tolist(),
        "memory_clean": clean_array.tolist(),
        "memory_reward": reward_array.tolist(),
        "memory_groups": group.tolist(),
        "contact_time_weights": time_weights.tolist(),
        "evidence": evidence,
        "holdout_open_authorized": True,
        "promotion_authorized": False,
    }
    actor["actor_hash"] = hash_json(actor)
    args.actor_output.write_text(
        json.dumps(actor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"gate": True, "report_hash": result["report_hash"], "actor_hash": actor["actor_hash"]}
        )
    )


if __name__ == "__main__":
    main()
