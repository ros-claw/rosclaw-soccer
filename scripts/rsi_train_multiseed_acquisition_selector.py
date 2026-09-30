"""Leave-one-seed-out growth gate for audited early-acquisition courses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_train_early_acquisition_selector import ARMS, load_verified_bank

from rosclaw_soccer.rsi.contextual_first_touch_option import fit_contextual_option
from rosclaw_soccer.sim.contracts import hash_json

SEEDS = (20260953, 20260954, 20260955, 20260956)


def cross_validate(
    banks: list[dict[str, Any]], features: np.ndarray[Any, Any], rewards: np.ndarray[Any, Any]
) -> dict[str, Any]:
    episodes = [row for bank in banks for row in bank["episodes"]]
    if (
        len(banks) != 2
        or len(episodes) != 32
        or [(row["seed"], row["lane"]) for row in episodes]
        != [(seed, lane) for seed in SEEDS for lane in range(0, 16, 2)]
        or features.shape != (32, 9)
        or rewards.shape != (32, 2)
    ):
        raise ValueError("four complete disjoint development seeds required")
    folds = []
    for test_seed in SEEDS:
        test = np.asarray([row["seed"] == test_seed for row in episodes])
        train = ~test
        selector = fit_contextual_option(features[train], rewards[train], action_count=2)
        fixed = int(np.argmax(np.mean(rewards[train], axis=0)))
        indices = np.flatnonzero(test)
        choices = np.asarray([selector.choose(features[i]) for i in indices])
        selected = rewards[indices, choices]
        fixed_rewards = rewards[indices, fixed]
        selected_arms = [
            episodes[int(i)]["arms"][ARMS[choice]]
            for i, choice in zip(indices, choices, strict=True)
        ]
        fixed_arms = [episodes[int(i)]["arms"][ARMS[fixed]] for i in indices]
        out_selected = sum(arm["maximum_lateral_excursion_m"] > 4.0 for arm in selected_arms)
        out_fixed = sum(arm["maximum_lateral_excursion_m"] > 4.0 for arm in fixed_arms)
        clean_selected = sum(arm["clean_foot_only"] for arm in selected_arms)
        clean_fixed = sum(arm["clean_foot_only"] for arm in fixed_arms)
        folds.append(
            {
                "test_seed": test_seed,
                "fixed_training_arm": ARMS[fixed],
                "choices": [ARMS[i] for i in choices],
                "mean_selected_reward": float(np.mean(selected)),
                "mean_fixed_reward": float(np.mean(fixed_rewards)),
                "mean_oracle_reward": float(np.mean(np.max(rewards[test], axis=1))),
                "out_of_play_selected": out_selected,
                "out_of_play_fixed": out_fixed,
                "clean_selected": clean_selected,
                "clean_fixed": clean_fixed,
            }
        )
    passed = bool(
        all(
            fold["mean_selected_reward"] > fold["mean_fixed_reward"]
            and fold["out_of_play_selected"] <= fold["out_of_play_fixed"]
            and fold["clean_selected"] >= fold["clean_fixed"]
            for fold in folds
        )
    )
    result: dict[str, Any] = {
        "schema": "rsi_multiseed_acquisition_selector_cv_v1",
        "activation_ceiling": "SIM_ONLY",
        "bank_report_hashes": [bank["report_hash"] for bank in banks],
        "folds": folds,
        "development_gate_passed": passed,
        "fresh_holdout_authorized": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-bank-root", required=True, type=Path)
    parser.add_argument("--second-bank-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.first_bank_root == args.second_bank_root:
        parser.error("new output and two distinct audited banks required")
    first, first_features, first_rewards = load_verified_bank(args.first_bank_root)
    second, second_features, second_rewards = load_verified_bank(args.second_bank_root)
    if any(first[key] != second[key] for key in ("source_hash", "asset_hash", "actor_hash")):
        raise ValueError("banks do not share frozen simulator source, G1 asset and actor")
    result = cross_validate(
        [first, second],
        np.concatenate((first_features, second_features)),
        np.concatenate((first_rewards, second_rewards)),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"MULTISEED_ACQUISITION_CV={result['report_hash']}", flush=True)
    print(f"DEVELOPMENT_GATE_PASSED={result['development_gate_passed']}", flush=True)


if __name__ == "__main__":
    main()
