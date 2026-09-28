"""Train a shared phase actor on audited, paired SIM_ONLY Isaac contacts.

The three actions for every lane are observed in physical replay, so leave-seed-
out action selection is an offline policy evaluation, not a world-model claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contextual_phase_policy import (
    FEATURE_NAMES,
    PHASE_ACTIONS,
    context_features,
    fit_action_values,
    select_actions,
)
from rosclaw_soccer.rsi.first_touch_snapshot_bank import audit_snapshot_bank
from rosclaw_soccer.rsi.snapshot_replay_evidence import audit_snapshot_replay
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def lane_outcomes(path: Path, count: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    with np.load(path / "replay.npz", allow_pickle=False) as trace:
        force = trace["observed_ball_body_contact_force_peak_n"][:, :count]
        ball = trace["observed_ball_position_local_m"][:, :count]
        root = trace["observed_root_pose_local_xyzw_m"][:, :count]
    if (
        force.ndim != 3
        or force.shape[1:] != (count, 6)
        or ball.shape != (len(force), count, 3)
        or root.shape != (len(force), count, 7)
        or not np.isfinite(force).all()
        or not np.isfinite(ball).all()
        or not np.isfinite(root).all()
        or np.any(force < 0)
        or float(np.min(root[:, :, 2])) < 0.65
    ):
        raise ValueError("unsafe or invalid paired physical replay")
    reward = np.full(count, -2.0)
    clean = np.zeros(count, dtype=np.bool_)
    for lane in range(count):
        active = np.flatnonzero(np.max(force[:, lane], axis=1) > 1.0)
        if len(active) == 0:
            continue
        first = int(active[0])
        first_bodies = set(np.flatnonzero(force[first, lane] > 1.0).tolist())
        all_bodies = set(np.flatnonzero(np.max(force[:, lane], axis=0) > 1.0).tolist())
        first_foot = bool(first_bodies and first_bodies <= {0, 1})
        clean[lane] = bool(all_bodies and all_bodies <= {0, 1})
        later = min(first + 6, len(force) - 1)
        if later == first:
            continue
        speed = (ball[later, lane, 0] - ball[first, lane, 0]) / ((later - first) * 0.02)
        reward[lane] = (
            float(clean[lane])
            + (0.5 if first_foot else -0.5)
            + 0.2 * float(np.clip(speed, -1.0, 3.0)) / 3.0
        )
    return reward, clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-bank", required=True, type=Path)
    parser.add_argument("--new-bank", required=True, type=Path)
    parser.add_argument("--old-base", required=True, type=Path)
    parser.add_argument("--old-plus3", required=True, type=Path)
    parser.add_argument("--old-plus6", required=True, type=Path)
    parser.add_argument("--new-base", required=True, type=Path)
    parser.add_argument("--new-plus3", required=True, type=Path)
    parser.add_argument("--new-plus6", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("immutable contextual actor output already exists")
    banks = ((args.old_bank, 15), (args.new_bank, 7))
    replays = (
        (args.old_base, args.old_plus3, args.old_plus6),
        (args.new_base, args.new_plus3, args.new_plus6),
    )
    features, rewards, clean, groups = [], [], [], []
    evidence = []
    for (bank_path, count), action_replays in zip(banks, replays, strict=True):
        bank_audit = audit_snapshot_bank(bank_path)
        manifest = json.loads((bank_path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("partition") != "CONSUMED_DEV":
            raise ValueError("training cannot consume sealed holdout")
        with np.load(bank_path / "snapshots.npz", allow_pickle=False) as snapshot:
            features.append(
                context_features(
                    snapshot["root_pose_local_xyzw_m"][:count],
                    snapshot["ball_position_local_m"][:count],
                    snapshot["ball_linear_velocity_m_s"][:count],
                    snapshot["foot_geometry_position_local_m"][:count],
                )
            )
        group_names = [row["source_folder"] for row in manifest["snapshots"][:count]]
        groups.extend(group_names)
        action_rewards, action_clean = [], []
        audits = []
        for action, replay_path in zip(PHASE_ACTIONS, action_replays, strict=True):
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
                        or report.get("warmup_max_target_error_rad", 1.0) > 1e-3
                        or report.get("max_parent_target_error_at_snapshot_rad", 1.0) > 1e-3
                        or audit["max_initial_state_error"] > 1e-4
                    )
                )
                or (action != 0 and not audit["intervention_action_audited"])
            ):
                raise ValueError("unauthenticated paired phase actions")
            reward, is_clean = lane_outcomes(replay_path, count)
            action_rewards.append(reward)
            action_clean.append(is_clean)
            audits.append(audit["report_hash"])
        rewards.append(np.column_stack(action_rewards))
        clean.append(np.column_stack(action_clean))
        evidence.append({"bank_hash": bank_audit["manifest_hash"], "replay_audit_hashes": audits})
    x = np.concatenate(features)
    y = np.concatenate(rewards)
    clean_array = np.concatenate(clean)
    group_array = np.asarray(groups)
    if len(set(groups)) != 3 or len(x) != 22:
        raise ValueError("three independent development sources and 22 paired lanes required")
    candidates = []
    for ridge in (0.1, 1.0, 10.0):
        choices = np.zeros(len(x), dtype=np.int64)
        for group in sorted(set(groups)):
            held = group_array == group
            weights = fit_action_values(x[~held], y[~held], ridge)
            choices[held] = (select_actions(x[held], weights) / 3).astype(np.int64)
        selected_reward = y[np.arange(len(x)), choices]
        selected_clean = clean_array[np.arange(len(x)), choices]
        candidates.append(
            {
                "ridge": ridge,
                "leave_source_out_mean_reward": float(np.mean(selected_reward)),
                "leave_source_out_clean_count": int(np.count_nonzero(selected_clean)),
                "leave_source_out_action_counts": np.bincount(choices, minlength=3).tolist(),
            }
        )
    best = max(candidates, key=lambda row: (row["leave_source_out_mean_reward"], -row["ridge"]))
    weights = fit_action_values(x, y, best["ridge"])
    parent_mean = float(np.mean(y[:, 0]))
    parent_clean = int(np.count_nonzero(clean_array[:, 0]))
    holdout_open = bool(
        best["leave_source_out_mean_reward"] > parent_mean
        and best["leave_source_out_clean_count"] >= parent_clean + 2
    )
    result = {
        "schema": "rsi_contextual_sonic_phase_actor_v1",
        "activation_ceiling": "SIM_ONLY",
        "feature_names": list(FEATURE_NAMES),
        "phase_actions_frames": list(PHASE_ACTIONS),
        "training_count": len(x),
        "development_evidence": evidence,
        "ridge_candidates": candidates,
        "selected_ridge": best["ridge"],
        "parent_development_mean_reward": parent_mean,
        "parent_development_clean_count": parent_clean,
        "weights": weights.tolist(),
        "trainer_source_hash": hash_bytes(Path(__file__).read_bytes()),
        "policy_source_hash": hash_bytes(
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("src/rosclaw_soccer/rsi/contextual_phase_policy.py")
            .read_bytes()
        ),
        "holdout_open_authorized": holdout_open,
        "promotion_authorized": False,
    }
    result["actor_hash"] = hash_json(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "weights"}))


if __name__ == "__main__":
    main()
