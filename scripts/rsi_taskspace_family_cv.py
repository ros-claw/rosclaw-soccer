"""Audit paired G1 physics and leave an entire course seed out for action selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import contact_time_phase_features as feature_module
from rosclaw_soccer.rsi import taskspace_family_memory as policy_module
from rosclaw_soccer.rsi.baseline_retention_phase import load_guarded_phase_actor
from rosclaw_soccer.rsi.contact_time_phase_features import (
    current_context,
    gait_phase_features,
    predict_contact_time,
)
from rosclaw_soccer.rsi.first_touch_snapshot_bank import audit_snapshot_bank
from rosclaw_soccer.rsi.snapshot_replay_evidence import audit_snapshot_replay
from rosclaw_soccer.rsi.taskspace_family_memory import (
    ACTION_NAMES,
    CONFIDENCE,
    NEIGHBORS,
    select_taskspace_family,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_contextual_phase_train import lane_outcomes

PROTOCOL = Path("docs/rsi/protocols/first-touch-taskspace-action-family-expansion-v9b.json")
INITIAL_PROTOCOL = Path("docs/rsi/protocols/first-touch-taskspace-action-family-v9.json")


def evaluate_family(
    features: np.ndarray[Any, Any],
    clean: np.ndarray[Any, Any],
    reward: np.ndarray[Any, Any],
    group: np.ndarray[Any, Any],
    seeds: list[int],
    *,
    neighbors: int,
    confidence: float,
) -> dict[str, Any]:
    choice = np.zeros(len(features), dtype=np.int64)
    for seed in seeds:
        held = group == seed
        choice[held] = select_taskspace_family(
            features[held],
            features[~held],
            clean[~held],
            reward[~held],
            group[~held],
            neighbors=neighbors,
            confidence=confidence,
        )
    selected_clean = clean[np.arange(len(clean)), choice]
    selected_reward = reward[np.arange(len(reward)), choice]
    parent_clean = clean[:, 0]
    per_seed = [
        {
            "seed": seed,
            "parent_clean": int(np.sum(parent_clean[group == seed])),
            "selected_clean": int(np.sum(selected_clean[group == seed])),
            "selected_actions": choice[group == seed].tolist(),
        }
        for seed in seeds
    ]
    return {
        "neighbors": neighbors,
        "confidence": confidence,
        "per_seed": per_seed,
        "parent_clean": int(np.sum(parent_clean)),
        "selected_clean": int(np.sum(selected_clean)),
        "parent_success_regressions": int(np.sum((parent_clean == 1) & (selected_clean == 0))),
        "parent_mean_reward": float(np.mean(reward[:, 0])),
        "selected_mean_reward": float(np.mean(selected_reward)),
        "intervention_count": int(np.count_nonzero(choice)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-actor", required=True, type=Path)
    parser.add_argument("--set", dest="sets", nargs=4, action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--actor-output", type=Path)
    args = parser.parse_args()
    if (
        args.output.exists()
        or (args.actor_output is not None and args.actor_output.exists())
        or len(args.sets) != 8
    ):
        parser.error("eight independent paired development groups and fresh output required")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    initial = json.loads(INITIAL_PROTOCOL.read_text(encoding="utf-8"))
    actor = load_guarded_phase_actor(args.phase_actor)
    if actor["actor_hash"] != initial["parent_actor_hash"]:
        raise ValueError("unregistered frozen parent actor")
    time_weights = np.asarray(actor["contact_time_weights"], dtype=np.float64)
    seeds: list[int] = []
    features, clean, reward, groups, evidence = [], [], [], [], []
    for bank_raw, parent_raw, up_raw, wide_raw in args.sets:
        bank, parent, up, wide = map(Path, (bank_raw, parent_raw, up_raw, wide_raw))
        bank_audit = audit_snapshot_bank(bank)
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        source = Path(manifest["snapshots"][0]["source_folder"])
        seed = json.loads((source / "report.json").read_text(encoding="utf-8")).get(
            "training_course_seed"
        )
        if (
            manifest.get("partition") != "SEALED_HOLDOUT"
            or manifest.get("snapshot_count") != 8
            or seed in seeds
            or seed
            not in initial["paired_development_seeds"]
            + protocol["additional_consumed_development_seeds"]
        ):
            raise ValueError("unregistered paired G1 development group")
        seeds.append(seed)
        audit_hashes = []
        action_clean, action_reward = [], []
        for action, path in zip(ACTION_NAMES, (parent, up, wide), strict=True):
            audit = audit_snapshot_replay(
                path, snapshot_bank=bank, local_phase_policy_path=args.phase_actor
            )
            report = json.loads((path / "report.json").read_text(encoding="utf-8"))
            if (
                audit["sample_count"] != 8
                or (action == "parent" and report.get("taskspace_forward_m") is not None)
                or (
                    action != "parent"
                    and (
                        not audit.get("taskspace_action_audited")
                        or report.get("taskspace_forward_m") != 0.08
                        or report.get("taskspace_vertical_offset_m", 0.0)
                        != (0.04 if action == "up" else 0.0)
                        or report.get("taskspace_lateral_cap_m", 0.05)
                        != (0.10 if action == "wide" else 0.05)
                    )
                )
            ):
                raise ValueError("unverified paired action family physics")
            one_reward, one_clean = lane_outcomes(path, 8)
            action_reward.append(one_reward)
            action_clean.append(one_clean)
            audit_hashes.append(audit["report_hash"])
        with np.load(bank / "snapshots.npz", allow_pickle=False) as snapshot:
            raw = current_context(
                snapshot["root_pose_local_xyzw_m"],
                snapshot["root_velocity_world"],
                snapshot["ball_position_local_m"],
                snapshot["ball_linear_velocity_m_s"],
            )
        features.append(gait_phase_features(raw, predict_contact_time(raw, time_weights)))
        clean.append(np.column_stack(action_clean).astype(np.float64))
        reward.append(np.column_stack(action_reward))
        groups.extend([seed] * 8)
        evidence.append(
            {"seed": seed, "bank_hash": bank_audit["manifest_hash"], "audits": audit_hashes}
        )
    if set(seeds) != set(
        initial["paired_development_seeds"] + protocol["additional_consumed_development_seeds"]
    ):
        raise ValueError("missing preregistered development course")
    features_array = np.concatenate(features)
    clean_array = np.concatenate(clean)
    reward_array = np.concatenate(reward)
    group_array = np.asarray(groups)
    candidates = [
        evaluate_family(
            features_array,
            clean_array,
            reward_array,
            group_array,
            seeds,
            neighbors=neighbors,
            confidence=confidence,
        )
        for neighbors in NEIGHBORS
        for confidence in CONFIDENCE
    ]
    gate = protocol["development_gate"]
    for candidate in candidates:
        candidate["gate_passed"] = bool(
            candidate["selected_clean"] - candidate["parent_clean"]
            >= gate["minimum_combined_clean_foot_gain_over_parent"]
            and all(row["selected_clean"] >= row["parent_clean"] for row in candidate["per_seed"])
            and candidate["parent_success_regressions"] == 0
            and candidate["selected_mean_reward"] > candidate["parent_mean_reward"]
        )
    qualified = [row for row in candidates if row["gate_passed"]]
    result = {
        "schema": "rsi_taskspace_family_cross_validation_v9b",
        "activation_ceiling": "SIM_ONLY",
        "phase_actor_hash": actor["actor_hash"],
        "protocol_hash": hash_json(protocol),
        "action_names": ACTION_NAMES,
        "evidence": evidence,
        "candidates": candidates,
        "development_gate_passed": bool(qualified),
        "fresh_holdout_open_authorized": bool(qualified),
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    if args.actor_output is not None:
        if not qualified:
            raise ValueError("development gate failed; actor output forbidden")
        best = max(
            qualified,
            key=lambda row: (
                row["selected_clean"],
                row["selected_mean_reward"],
                row["confidence"],
                -row["neighbors"],
            ),
        )
        if policy_module.__file__ is None or feature_module.__file__ is None:
            raise ValueError("unresolved frozen policy source")
        frozen_actor = {
            "schema": "rsi_taskspace_family_actor_v9b",
            "activation_ceiling": "SIM_ONLY",
            "action_names": list(ACTION_NAMES),
            "feature_names": list(feature_module.ACTION_FEATURE_NAMES),
            "frozen_phase_actor_hash": actor["actor_hash"],
            "protocol_hash": hash_json(protocol),
            "cross_validation_report_hash": result["report_hash"],
            "policy_source_hash": hash_bytes(Path(policy_module.__file__).read_bytes()),
            "feature_source_hash": hash_bytes(Path(feature_module.__file__).read_bytes()),
            "neighbors": best["neighbors"],
            "confidence": best["confidence"],
            "memory_features": features_array.tolist(),
            "memory_clean": clean_array.tolist(),
            "memory_reward": reward_array.tolist(),
            "memory_groups": group_array.tolist(),
            "contact_time_weights": time_weights.tolist(),
            "evidence": evidence,
            "holdout_open_authorized": True,
            "promotion_authorized": False,
        }
        frozen_actor["actor_hash"] = hash_json(frozen_actor)
        args.actor_output.write_text(
            json.dumps(frozen_actor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"report_hash": result["report_hash"], "gate": result["development_gate_passed"]}
        )
    )


if __name__ == "__main__":
    main()
