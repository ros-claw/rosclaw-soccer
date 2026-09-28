"""Train a conservative SIM_ONLY foot-correction gate on paired physical episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import contact_time_phase_features as feature_module
from rosclaw_soccer.rsi import taskspace_gate_memory as policy_module
from rosclaw_soccer.rsi.baseline_retention_phase import load_guarded_phase_actor
from rosclaw_soccer.rsi.contact_time_phase_features import (
    ACTION_FEATURE_NAMES,
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

PROTOCOL = Path("docs/rsi/protocols/first-touch-taskspace-gate-v8.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-actor", required=True, type=Path)
    parser.add_argument("--set", dest="sets", nargs=3, action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or len(args.sets) not in (4, 6):
        parser.error("four or six paired independent development seeds and new output required")
    phase_actor = load_guarded_phase_actor(args.phase_actor)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if phase_actor["actor_hash"] != protocol["frozen_parent_actor_hash"]:
        raise ValueError("unregistered frozen phase parent actor")
    time_weights = np.asarray(phase_actor["contact_time_weights"], dtype=np.float64)
    features, rewards, cleans, groups, evidence = [], [], [], [], []
    seeds = []
    for bank_raw, parent_raw, candidate_raw in args.sets:
        bank, parent, candidate = map(Path, (bank_raw, parent_raw, candidate_raw))
        bank_audit = audit_snapshot_bank(bank)
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        source = Path(manifest["snapshots"][0]["source_folder"])
        source_report = json.loads((source / "report.json").read_text(encoding="utf-8"))
        seed = source_report.get("training_course_seed")
        if (
            manifest.get("partition") != "SEALED_HOLDOUT"
            or manifest["snapshot_count"] != 8
            or seed
            not in (
                protocol["paired_development_course_seeds"]
                + protocol["additional_development_course_seeds_if_gate_fails"]
            )
            or seed in seeds
        ):
            raise ValueError("unregistered or duplicated development physical bank")
        seeds.append(seed)
        parent_audit = audit_snapshot_replay(
            parent, snapshot_bank=bank, local_phase_policy_path=args.phase_actor
        )
        candidate_audit = audit_snapshot_replay(
            candidate, snapshot_bank=bank, local_phase_policy_path=args.phase_actor
        )
        report = json.loads((candidate / "report.json").read_text(encoding="utf-8"))
        if (
            parent_audit["sample_count"] != 8
            or candidate_audit["sample_count"] != 8
            or parent_audit.get("taskspace_action_audited") is not None
            or not candidate_audit.get("taskspace_action_audited")
            or report.get("taskspace_forward_m") != 0.08
        ):
            raise ValueError("unverified paired G1 task-space physics")
        with np.load(bank / "snapshots.npz", allow_pickle=False) as snapshot:
            raw = current_context(
                snapshot["root_pose_local_xyzw_m"],
                snapshot["root_velocity_world"],
                snapshot["ball_position_local_m"],
                snapshot["ball_linear_velocity_m_s"],
            )
        features.append(gait_phase_features(raw, predict_contact_time(raw, time_weights)))
        parent_reward, parent_clean = lane_outcomes(parent, 8)
        candidate_reward, candidate_clean = lane_outcomes(candidate, 8)
        rewards.append(np.column_stack((parent_reward, candidate_reward)))
        cleans.append(np.column_stack((parent_clean, candidate_clean)).astype(np.float64))
        groups.extend([seed] * 8)
        evidence.append(
            {
                "seed": seed,
                "bank_hash": bank_audit["manifest_hash"],
                "parent_audit_hash": parent_audit["report_hash"],
                "candidate_audit_hash": candidate_audit["report_hash"],
            }
        )
    expected = (
        protocol["paired_development_course_seeds"]
        if len(args.sets) == 4
        else protocol["paired_development_course_seeds"]
        + protocol["additional_development_course_seeds_if_gate_fails"]
    )
    if set(seeds) != set(expected):
        raise ValueError("missing or incorrect pre-registered development source")
    x = np.concatenate(features)
    reward = np.concatenate(rewards)
    clean = np.concatenate(cleans)
    group = np.asarray(groups)
    candidate_rows: list[dict[str, Any]] = []
    for neighbors in NEIGHBOR_COUNTS:
        for confidence in CONFIDENCE_MULTIPLIERS:
            for ceiling in BASELINE_CLEAN_CEILINGS:
                choice = np.zeros(len(x), dtype=np.bool_)
                for held_seed in seeds:
                    held = group == held_seed
                    training_groups = np.asarray(
                        [seeds.index(value) for value in group[~held]], dtype=np.int64
                    )
                    choice[held] = select_taskspace_gate(
                        x[held],
                        x[~held],
                        clean[~held],
                        reward[~held],
                        training_groups,
                        neighbors=neighbors,
                        confidence=confidence,
                        baseline_clean_ceiling=ceiling,
                    )
                selected_clean = clean[np.arange(len(x)), choice.astype(int)]
                selected_reward = reward[np.arange(len(x)), choice.astype(int)]
                gains = [
                    int(np.count_nonzero(selected_clean[group == seed]))
                    - int(np.count_nonzero(clean[group == seed, 0]))
                    for seed in seeds
                ]
                losses = int(np.count_nonzero((clean[:, 0] == 1) & (selected_clean == 0)))
                candidate_rows.append(
                    {
                        "neighbors": neighbors,
                        "confidence": confidence,
                        "baseline_clean_ceiling": ceiling,
                        "leave_seed_out_clean_count": int(np.count_nonzero(selected_clean)),
                        "leave_seed_out_group_clean_gains": gains,
                        "leave_seed_out_parent_success_losses": losses,
                        "leave_seed_out_mean_reward": float(np.mean(selected_reward)),
                        "leave_seed_out_intervention_count": int(np.count_nonzero(choice)),
                    }
                )
    best = max(
        candidate_rows,
        key=lambda row: (
            -row["leave_seed_out_parent_success_losses"],
            min(row["leave_seed_out_group_clean_gains"]),
            row["leave_seed_out_clean_count"],
            row["leave_seed_out_mean_reward"],
        ),
    )
    parent_clean_count = int(np.count_nonzero(clean[:, 0]))
    parent_reward_mean = float(np.mean(reward[:, 0]))
    open_holdout = bool(
        best["leave_seed_out_parent_success_losses"] == 0
        and min(best["leave_seed_out_group_clean_gains"]) >= 0
        and best["leave_seed_out_clean_count"] >= parent_clean_count + 3
        and best["leave_seed_out_mean_reward"] > parent_reward_mean
    )
    actor: dict[str, Any] = {
        "schema": "rsi_taskspace_gate_actor_v8",
        "activation_ceiling": "SIM_ONLY",
        "frozen_phase_actor_hash": phase_actor["actor_hash"],
        "frozen_taskspace_action_forward_m": 0.08,
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "development_seeds": seeds,
        "development_evidence": evidence,
        "candidate_rows": candidate_rows,
        "neighbors": best["neighbors"],
        "confidence": best["confidence"],
        "baseline_clean_ceiling": best["baseline_clean_ceiling"],
        "parent_development_clean_count": parent_clean_count,
        "parent_development_mean_reward": parent_reward_mean,
        "contact_time_weights": time_weights.tolist(),
        "memory_features": x.tolist(),
        "memory_clean": clean.tolist(),
        "memory_reward": reward.tolist(),
        "memory_groups": [seeds.index(seed) for seed in groups],
        "protocol_hash": hash_bytes(PROTOCOL.read_bytes()),
        "policy_source_hash": hash_bytes(Path(policy_module.__file__).read_bytes()),
        "feature_source_hash": hash_bytes(Path(feature_module.__file__).read_bytes()),
        "trainer_source_hash": hash_bytes(Path(__file__).read_bytes()),
        "holdout_open_authorized": open_holdout,
        "promotion_authorized": False,
    }
    actor["actor_hash"] = hash_json(actor)
    args.output.write_text(json.dumps(actor, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in actor.items()
                if key not in ("memory_features", "memory_clean", "memory_reward", "memory_groups")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
