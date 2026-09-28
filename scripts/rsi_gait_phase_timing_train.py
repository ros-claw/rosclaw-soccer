"""Fit contact-time-aware phase actor on paired SIM_ONLY Isaac development seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import contact_time_phase_features as feature_module
from rosclaw_soccer.rsi.contact_time_phase_features import (
    ACTION_FEATURE_NAMES,
    RAW_FEATURE_NAMES,
    current_context,
    fit_action_values,
    fit_contact_time,
    gait_phase_features,
    predict_contact_time,
    select_phase_actions,
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
        parser.error("five development sets and fresh immutable output required")
    raw_blocks: list[np.ndarray[Any, Any]] = []
    reward_blocks: list[np.ndarray[Any, Any]] = []
    clean_blocks: list[np.ndarray[Any, Any]] = []
    contact_blocks: list[np.ndarray[Any, Any]] = []
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
            raise ValueError("contact-time teacher cannot consume sealed holdout")
        with np.load(bank_path / "snapshots.npz", allow_pickle=False) as bank:
            raw_blocks.append(
                current_context(
                    bank["root_pose_local_xyzw_m"][:count],
                    bank["root_velocity_world"][:count],
                    bank["ball_position_local_m"][:count],
                    bank["ball_linear_velocity_m_s"][:count],
                )
            )
        contact_blocks.append(
            np.asarray([row["first_contact_offset"] for row in manifest["snapshots"][:count]])
        )
        groups.extend(row["source_folder"] for row in manifest["snapshots"][:count])
        rewards, clean, audits = [], [], []
        for action, raw_path in zip((0.0, 3.0, 6.0), replay_raw, strict=True):
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
                raise ValueError("unauthenticated paired physical action")
            reward, is_clean = lane_outcomes(replay_path, count)
            rewards.append(reward)
            clean.append(is_clean)
            audits.append(audit["report_hash"])
        reward_blocks.append(np.column_stack(rewards))
        clean_blocks.append(np.column_stack(clean))
        evidence.append({"bank_hash": bank_audit["manifest_hash"], "replay_audit_hashes": audits})
    raw = np.concatenate(raw_blocks)
    reward = np.concatenate(reward_blocks)
    clean_outcome = np.concatenate(clean_blocks)
    contact = np.concatenate(contact_blocks)
    group_array = np.asarray(groups)
    if len(raw) != 46 or len(set(groups)) != 6:
        raise ValueError("six independent development seeds and 46 paired contacts required")
    candidate_rows: list[dict[str, Any]] = []
    for ridge in (0.1, 1.0, 10.0):
        choices = np.zeros(len(raw), dtype=np.int64)
        contact_error = np.zeros(len(raw))
        for group in sorted(set(groups)):
            held = group_array == group
            time_weights = fit_contact_time(raw[~held], contact[~held])
            train_features = gait_phase_features(
                raw[~held], predict_contact_time(raw[~held], time_weights)
            )
            held_contact = predict_contact_time(raw[held], time_weights)
            contact_error[held] = np.abs(held_contact - contact[held])
            held_features = gait_phase_features(raw[held], held_contact)
            action_weights = fit_action_values(train_features, reward[~held], ridge)
            choices[held] = (select_phase_actions(held_features, action_weights) / 3).astype(
                np.int64
            )
        selected_reward = reward[np.arange(len(raw)), choices]
        selected_clean = clean_outcome[np.arange(len(raw)), choices]
        group_gains = [
            int(np.count_nonzero(selected_clean[group_array == group]))
            - int(np.count_nonzero(clean_outcome[group_array == group, 0]))
            for group in sorted(set(groups))
        ]
        candidate_rows.append(
            {
                "ridge": ridge,
                "leave_source_out_mean_reward": float(np.mean(selected_reward)),
                "leave_source_out_clean_count": int(np.count_nonzero(selected_clean)),
                "leave_source_out_group_clean_gains": group_gains,
                "leave_source_out_contact_time_mae_frames": float(np.mean(contact_error)),
                "leave_source_out_action_counts": np.bincount(choices, minlength=3).tolist(),
            }
        )
    best = max(
        candidate_rows,
        key=lambda row: (
            min(row["leave_source_out_group_clean_gains"]),
            row["leave_source_out_clean_count"],
            row["leave_source_out_mean_reward"],
        ),
    )
    parent_count = int(np.count_nonzero(clean_outcome[:, 0]))
    parent_reward = float(np.mean(reward[:, 0]))
    holdout_open = bool(
        min(best["leave_source_out_group_clean_gains"]) >= 0
        and best["leave_source_out_clean_count"] >= parent_count + 4
        and best["leave_source_out_mean_reward"] > parent_reward
        and best["leave_source_out_contact_time_mae_frames"] <= 6.0
    )
    final_time_weights = fit_contact_time(raw, contact)
    final_features = gait_phase_features(raw, predict_contact_time(raw, final_time_weights))
    final_action_weights = fit_action_values(final_features, reward, best["ridge"])
    result = {
        "schema": "rsi_gait_phase_timing_actor_v3",
        "activation_ceiling": "SIM_ONLY",
        "raw_feature_names": list(RAW_FEATURE_NAMES),
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "gait_period_frames": 40,
        "phase_actions_frames": [0.0, 3.0, 6.0],
        "training_count": len(raw),
        "development_evidence": evidence,
        "ridge_candidates": candidate_rows,
        "selected_ridge": best["ridge"],
        "parent_development_mean_reward": parent_reward,
        "parent_development_clean_count": parent_count,
        "contact_time_weights": final_time_weights.tolist(),
        "action_value_weights": final_action_weights.tolist(),
        "trainer_source_hash": hash_bytes(Path(__file__).read_bytes()),
        "feature_source_hash": hash_bytes(Path(feature_module.__file__).read_bytes()),
        "holdout_open_authorized": holdout_open,
        "promotion_authorized": False,
    }
    result["actor_hash"] = hash_json(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("_weights")}))


if __name__ == "__main__":
    main()
