"""Train seed-diverse local first-touch phase memory on paired Isaac physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import baseline_retention_phase as guard_module
from rosclaw_soccer.rsi import contact_time_phase_features as feature_module
from rosclaw_soccer.rsi import local_phase_memory as policy_module
from rosclaw_soccer.rsi.baseline_retention_phase import (
    BASELINE_CLEAN_CEILINGS,
    select_guarded_phase,
)
from rosclaw_soccer.rsi.contact_time_phase_features import (
    ACTION_FEATURE_NAMES,
    current_context,
    fit_contact_time,
    gait_phase_features,
    predict_contact_time,
)
from rosclaw_soccer.rsi.first_touch_snapshot_bank import audit_snapshot_bank
from rosclaw_soccer.rsi.local_phase_memory import (
    CONFIDENCE_MULTIPLIERS,
    NEIGHBOR_COUNTS,
    select_local_phase,
)
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
    parser.add_argument("--baseline-retention-v5", action="store_true")
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
            raise ValueError("local memory cannot consume sealed holdout")
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
    clean_outcome = np.concatenate(clean_blocks).astype(np.float64)
    contact = np.concatenate(contact_blocks)
    group_array = np.asarray(groups)
    group_names = sorted(set(groups))
    if len(raw) != 46 or len(group_names) != 6:
        raise ValueError("six independent development seeds and 46 paired contacts required")
    candidate_rows: list[dict[str, Any]] = []
    neighbor_grid: tuple[int, ...]
    confidence_grid: tuple[float, ...]
    if args.baseline_retention_v5:
        neighbor_grid = guard_module.NEIGHBOR_COUNTS
        confidence_grid = guard_module.CONFIDENCE_MULTIPLIERS
        ceiling_grid: tuple[float | None, ...] = BASELINE_CLEAN_CEILINGS
    else:
        neighbor_grid = NEIGHBOR_COUNTS
        confidence_grid = CONFIDENCE_MULTIPLIERS
        ceiling_grid = (None,)
    for neighbor_count in neighbor_grid:
        for confidence in confidence_grid:
            for ceiling in ceiling_grid:
                candidate_rows.append(
                    _score_candidate(
                        raw,
                        reward,
                        clean_outcome,
                        contact,
                        group_array,
                        group_names,
                        neighbor_count,
                        confidence,
                        ceiling,
                    )
                )
    best = max(
        candidate_rows,
        key=lambda row: (
            min(row["leave_source_out_group_clean_gains"]),
            row["leave_source_out_clean_count"],
            row["leave_source_out_mean_reward"],
            row["confidence"],
        ),
    )
    parent_count = int(np.count_nonzero(clean_outcome[:, 0]))
    parent_reward = float(np.mean(reward[:, 0]))
    holdout_open = bool(
        min(best["leave_source_out_group_clean_gains"]) >= 0
        and best["leave_source_out_clean_count"]
        >= parent_count + (5 if args.baseline_retention_v5 else 4)
        and best["leave_source_out_mean_reward"] > parent_reward
        and best["leave_source_out_contact_time_mae_frames"] <= 6.0
    )
    final_time_weights = fit_contact_time(raw, contact)
    final_features = gait_phase_features(raw, predict_contact_time(raw, final_time_weights))
    result = {
        "schema": (
            "rsi_baseline_retention_phase_actor_v5"
            if args.baseline_retention_v5
            else "rsi_local_contact_phase_actor_v4"
        ),
        "activation_ceiling": "SIM_ONLY",
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "phase_actions_frames": [0.0, 3.0, 6.0],
        "training_count": len(raw),
        "development_evidence": evidence,
        "candidate_rows": candidate_rows,
        "neighbors": best["neighbors"],
        "confidence": best["confidence"],
        "parent_development_mean_reward": parent_reward,
        "parent_development_clean_count": parent_count,
        "contact_time_weights": final_time_weights.tolist(),
        "memory_features": final_features.tolist(),
        "memory_clean": clean_outcome.tolist(),
        "memory_reward": reward.tolist(),
        "memory_groups": [group_names.index(name) for name in groups],
        "trainer_source_hash": hash_bytes(Path(__file__).read_bytes()),
        "policy_source_hash": hash_bytes(
            Path(
                guard_module.__file__ if args.baseline_retention_v5 else policy_module.__file__
            ).read_bytes()
        ),
        "feature_source_hash": hash_bytes(Path(feature_module.__file__).read_bytes()),
        "holdout_open_authorized": holdout_open,
        "promotion_authorized": False,
    }
    if args.baseline_retention_v5:
        result["baseline_clean_ceiling"] = best["baseline_clean_ceiling"]
        result["protocol_hash"] = hash_bytes(
            Path("docs/rsi/protocols/first-touch-baseline-retention-v5.json").read_bytes()
        )
    result["actor_hash"] = hash_json(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("memory_features", "memory_clean", "memory_reward", "memory_groups")
            }
        )
    )


def _score_candidate(
    raw: np.ndarray[Any, Any],
    reward: np.ndarray[Any, Any],
    clean_outcome: np.ndarray[Any, Any],
    contact: np.ndarray[Any, Any],
    group_array: np.ndarray[Any, Any],
    group_names: list[str],
    neighbor_count: int,
    confidence: float,
    ceiling: float | None,
) -> dict[str, Any]:
    choices = np.zeros(len(raw), dtype=np.int64)
    contact_error = np.zeros(len(raw))
    for group in group_names:
        held = group_array == group
        time_weights = fit_contact_time(raw[~held], contact[~held])
        train_features = gait_phase_features(
            raw[~held], predict_contact_time(raw[~held], time_weights)
        )
        held_contact = predict_contact_time(raw[held], time_weights)
        contact_error[held] = np.abs(held_contact - contact[held])
        held_features = gait_phase_features(raw[held], held_contact)
        training_groups = np.asarray(
            [group_names.index(name) for name in group_array[~held]], dtype=np.int64
        )
        select = select_local_phase if ceiling is None else select_guarded_phase
        parameters = {} if ceiling is None else {"baseline_clean_ceiling": ceiling}
        choices[held] = (
            select(
                held_features,
                train_features,
                clean_outcome[~held],
                reward[~held],
                training_groups,
                neighbors=neighbor_count,
                confidence=confidence,
                **parameters,
            )
            / 3
        ).astype(np.int64)
    selected_reward = reward[np.arange(len(raw)), choices]
    selected_clean = clean_outcome[np.arange(len(raw)), choices]
    group_gains = [
        int(np.count_nonzero(selected_clean[group_array == group]))
        - int(np.count_nonzero(clean_outcome[group_array == group, 0]))
        for group in group_names
    ]
    return {
        "neighbors": neighbor_count,
        "confidence": confidence,
        "baseline_clean_ceiling": ceiling,
        "leave_source_out_mean_reward": float(np.mean(selected_reward)),
        "leave_source_out_clean_count": int(np.count_nonzero(selected_clean)),
        "leave_source_out_group_clean_gains": group_gains,
        "leave_source_out_contact_time_mae_frames": float(np.mean(contact_error)),
        "leave_source_out_action_counts": np.bincount(choices, minlength=3).tolist(),
    }


if __name__ == "__main__":
    main()
