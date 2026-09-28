"""Fit support-aware phase actor from paired Isaac physics across whole course seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import contextual_phase_policy_v2 as policy_module
from rosclaw_soccer.rsi.contextual_phase_policy_v2 import (
    FEATURE_NAMES,
    PHASE_ACTIONS,
    SUPPORT_THRESHOLDS,
    context_features,
    fit_action_values,
    support_aware_actions,
)
from rosclaw_soccer.rsi.first_touch_snapshot_bank import audit_snapshot_bank
from rosclaw_soccer.rsi.snapshot_replay_evidence import audit_snapshot_replay
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_contextual_phase_train import lane_outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        dest="sets",
        nargs=5,
        action="append",
        metavar=("BANK", "COUNT", "BASE", "PLUS3", "PLUS6"),
        required=True,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or len(args.sets) != 5:
        parser.error("five independent development sets and fresh immutable output required")
    feature_blocks: list[np.ndarray[Any, Any]] = []
    reward_blocks: list[np.ndarray[Any, Any]] = []
    clean_blocks: list[np.ndarray[Any, Any]] = []
    groups: list[str] = []
    evidence = []
    for bank_raw, count_raw, *replay_raw in args.sets:
        bank_path = Path(bank_raw)
        count = int(count_raw)
        bank_audit = audit_snapshot_bank(bank_path)
        manifest = json.loads((bank_path / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("partition") != "CONSUMED_DEV"
            or not 2 <= count <= manifest["snapshot_count"]
        ):
            raise ValueError("development sets may not consume sealed holdout")
        with np.load(bank_path / "snapshots.npz", allow_pickle=False) as bank:
            feature_blocks.append(
                context_features(
                    bank["root_pose_local_xyzw_m"][:count],
                    bank["ball_position_local_m"][:count],
                    bank["ball_linear_velocity_m_s"][:count],
                    bank["foot_geometry_position_local_m"][:count],
                )
            )
        groups.extend(row["source_folder"] for row in manifest["snapshots"][:count])
        rewards, clean, audits = [], [], []
        for action, raw_path in zip(PHASE_ACTIONS, replay_raw, strict=True):
            replay_path = Path(raw_path)
            audit = audit_snapshot_replay(replay_path, snapshot_bank=bank_path)
            report = json.loads((replay_path / "report.json").read_text(encoding="utf-8"))
            if (
                report["start_index"] != 0
                or report["sample_count"] < count
                or report.get("phase_target_frames") != (None if action == 0 else action)
                or not audit["closed_loop_sonic"]
                or (
                    action == 0
                    and (
                        audit["intervention_action_audited"]
                        or audit["contact_body_class_equal_count"] < report["sample_count"] - 1
                        or audit["first_contact_frame_equal_count"] < report["sample_count"] - 1
                    )
                )
                or (action != 0 and not audit["intervention_action_audited"])
            ):
                raise ValueError("unqualified paired physical phase replay")
            reward, is_clean = lane_outcomes(replay_path, count)
            rewards.append(reward)
            clean.append(is_clean)
            audits.append(audit["report_hash"])
        reward_blocks.append(np.column_stack(rewards))
        clean_blocks.append(np.column_stack(clean))
        evidence.append({"bank_hash": bank_audit["manifest_hash"], "replay_audit_hashes": audits})
    x = np.concatenate(feature_blocks)
    y = np.concatenate(reward_blocks)
    clean_array = np.concatenate(clean_blocks)
    group_array = np.asarray(groups)
    if len(x) < 35 or len(set(groups)) != 6:
        raise ValueError("insufficient independent incoming contact coverage")
    candidates: list[dict[str, Any]] = []
    for ridge in (0.1, 1.0, 10.0):
        for threshold in SUPPORT_THRESHOLDS:
            choices = np.zeros(len(x), dtype=np.int64)
            for group in sorted(set(groups)):
                held = group_array == group
                weights = fit_action_values(x[~held], y[~held], ridge)
                choices[held] = (
                    support_aware_actions(x[held], weights, x[~held], threshold) / 3
                ).astype(np.int64)
            selected_reward = y[np.arange(len(x)), choices]
            selected_clean = clean_array[np.arange(len(x)), choices]
            group_gains = [
                int(np.count_nonzero(selected_clean[group_array == group]))
                - int(np.count_nonzero(clean_array[group_array == group, 0]))
                for group in sorted(set(groups))
            ]
            candidates.append(
                {
                    "ridge": ridge,
                    "support_threshold": threshold,
                    "leave_source_out_mean_reward": float(np.mean(selected_reward)),
                    "leave_source_out_clean_count": int(np.count_nonzero(selected_clean)),
                    "leave_source_out_group_clean_gains": group_gains,
                    "leave_source_out_action_counts": np.bincount(choices, minlength=3).tolist(),
                }
            )
    best = max(
        candidates,
        key=lambda row: (
            min(row["leave_source_out_group_clean_gains"]),
            row["leave_source_out_clean_count"],
            row["leave_source_out_mean_reward"],
            -row["support_threshold"],
        ),
    )
    parent_clean = int(np.count_nonzero(clean_array[:, 0]))
    parent_reward = float(np.mean(y[:, 0]))
    holdout_open = bool(
        min(best["leave_source_out_group_clean_gains"]) >= 0
        and best["leave_source_out_clean_count"] >= parent_clean + 3
        and best["leave_source_out_mean_reward"] > parent_reward
    )
    weights = fit_action_values(x, y, best["ridge"])
    result = {
        "schema": "rsi_contextual_sonic_phase_actor_v2",
        "activation_ceiling": "SIM_ONLY",
        "feature_names": list(FEATURE_NAMES),
        "phase_actions_frames": list(PHASE_ACTIONS),
        "training_count": len(x),
        "development_evidence": evidence,
        "ridge_candidates": candidates,
        "selected_ridge": best["ridge"],
        "support_threshold": best["support_threshold"],
        "parent_development_mean_reward": parent_reward,
        "parent_development_clean_count": parent_clean,
        "weights": weights.tolist(),
        "support_contexts": x.tolist(),
        "trainer_source_hash": hash_bytes(Path(__file__).read_bytes()),
        "policy_source_hash": hash_bytes(Path(policy_module.__file__).read_bytes()),
        "holdout_open_authorized": holdout_open,
        "promotion_authorized": False,
    }
    result["actor_hash"] = hash_json(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("weights", "support_contexts")
            }
        )
    )


if __name__ == "__main__":
    main()
