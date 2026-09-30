"""Cross-validate a two-window first-touch selector on audited Isaac pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contextual_first_touch_option import (
    first_touch_reward,
    fit_contextual_option,
)
from rosclaw_soccer.rsi.independent_first_touch_bank import post_contact_displacement
from rosclaw_soccer.rsi.taskspace_swing_evidence import audit_taskspace_swing_trace
from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch
from rosclaw_soccer.sim.contracts import hash_json

ARMS = ("acquisition_055", "acquisition_095")


def load_verified_bank(
    root: Path,
) -> tuple[dict[str, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    bank: dict[str, Any] = json.loads((root / "bank_summary.json").read_text(encoding="utf-8"))
    if (
        bank.get("schema") != "rsi_independent_early_acquisition_bank_v1"
        or bank.get("activation_ceiling") != "SIM_ONLY"
        or bank.get("promotion_authorized") is not False
        or bank.get("failures") != []
        or bank.get("report_hash")
        != hash_json({k: v for k, v in bank.items() if k != "report_hash"})
        or not isinstance(bank.get("episodes"), list)
        or [(row["seed"], row["lane"]) for row in bank["episodes"]]
        not in (
            [(seed, lane) for seed in (20260953, 20260954) for lane in range(0, 16, 2)],
            [(seed, lane) for seed in (20260955, 20260956) for lane in range(0, 16, 2)],
        )
    ):
        raise ValueError("incomplete or unauthenticated early-acquisition bank")
    features = []
    rewards = []
    for row in bank["episodes"]:
        if set(row["arms"]) != set(ARMS):
            raise ValueError("missing counterfactual acquisition arm")
        pair_features = None
        pair_rewards = []
        for arm in ARMS:
            folder = root / f"seed{row['seed']}-lane{row['lane']}-{arm}"
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            physical = audit_vector_first_touch(folder)
            summary = row["arms"][arm]
            if (
                report.get("source_hash") != bank["source_hash"]
                or report.get("asset_hash") != bank["asset_hash"]
                or report.get("late_swing_actor_hash") != bank["actor_hash"]
                or report["report_hash"] != summary["report_hash"]
                or physical["report_hash"] != summary["physical_audit_hash"]
                or report["environments"][0]["course"] != summary["course"]
            ):
                raise ValueError(f"physical bank drift: {folder}")
            entry = report["environments"][0]
            with np.load(folder / "trace.npz", allow_pickle=False) as physics:
                displacement = post_contact_displacement(
                    physics["ball_position_m"][:, 0], entry["first_contact_frame"]
                )
            bodies = entry["contact_body_indices"]
            if (
                summary["first_contact_frame"] != entry["first_contact_frame"]
                or summary["contact_body_indices"] != bodies
                or summary["clean_foot_only"] != bool(bodies and set(bodies) <= {0, 1})
                or summary["minimum_pelvis_z_m"] != entry["minimum_pelvis_z_m"]
                or summary["maximum_lateral_excursion_m"]
                != report["single_instance_max_lateral_excursion_m"]
                or any(summary[k] != value for k, value in displacement.items())
            ):
                raise ValueError(f"physical reward label drift: {folder}")
            with np.load(folder / "late_swing_action_trace.npz", allow_pickle=False) as replay:
                action = audit_taskspace_swing_trace(replay, report, frames=300, count=1)
                current = np.asarray(replay["frame30_gate_features"][0], dtype=float)
                if action != summary["action_audit"]:
                    raise ValueError(f"action trace drift: {folder}")
                if pair_features is None:
                    pair_features = current
                elif not np.allclose(pair_features, current, atol=1e-10, rtol=0):
                    raise ValueError("paired precontact features differ")
            pair_rewards.append(
                first_touch_reward(
                    {**summary, "max_lateral_excursion_m": summary["maximum_lateral_excursion_m"]}
                )
            )
        if pair_features is None or pair_features.shape != (9,):
            raise ValueError("missing causal precontact features")
        features.append(pair_features)
        rewards.append(pair_rewards)
    return bank, np.asarray(features), np.asarray(rewards)


def cross_validate(
    bank: dict[str, Any], features: np.ndarray[Any, Any], rewards: np.ndarray[Any, Any]
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for train_seed, test_seed in ((20260953, 20260954), (20260954, 20260953)):
        train = np.asarray([row["seed"] == train_seed for row in bank["episodes"]])
        test = ~train
        selector = fit_contextual_option(features[train], rewards[train], action_count=2)
        fixed = int(np.argmax(np.mean(rewards[train], axis=0)))
        indices = np.flatnonzero(test)
        choices = np.asarray([selector.choose(features[i]) for i in indices])
        selected = rewards[indices, choices]
        fixed_rewards = rewards[indices, fixed]
        folds.append(
            {
                "train_seed": train_seed,
                "test_seed": test_seed,
                "fixed_training_arm": ARMS[fixed],
                "choices": [ARMS[i] for i in choices],
                "mean_selected_reward": float(np.mean(selected)),
                "mean_fixed_reward": float(np.mean(fixed_rewards)),
                "mean_055_reward": float(np.mean(rewards[test, 0])),
                "mean_095_reward": float(np.mean(rewards[test, 1])),
                "mean_oracle_reward": float(np.mean(np.max(rewards[test], axis=1))),
            }
        )
    passed = bool(all(fold["mean_selected_reward"] > fold["mean_fixed_reward"] for fold in folds))
    result: dict[str, Any] = {
        "schema": "rsi_early_acquisition_selector_cv_v1",
        "activation_ceiling": "SIM_ONLY",
        "bank_report_hash": bank["report_hash"],
        "folds": folds,
        "development_gate_passed": passed,
        "fresh_holdout_authorized": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.bank_root.is_dir() or args.output.exists():
        parser.error("existing audited bank and new output file required")
    bank, features, rewards = load_verified_bank(args.bank_root)
    result = cross_validate(bank, features, rewards)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"EARLY_SELECTOR_CV={result['report_hash']}", flush=True)
    print(f"DEVELOPMENT_GATE_PASSED={result['development_gate_passed']}", flush=True)


if __name__ == "__main__":
    main()
