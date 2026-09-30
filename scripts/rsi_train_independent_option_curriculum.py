"""Cross-validate a conservative SIM_ONLY G1 first-touch option-value learner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.conservative_first_touch_option import (
    GAIN_MARGIN,
    MAX_NEAREST_DISTANCE,
    RIDGE_STRENGTH,
    fit_conservative_option_value,
)
from rosclaw_soccer.rsi.contextual_first_touch_option import first_touch_reward
from rosclaw_soccer.rsi.independent_first_touch_bank import post_contact_displacement
from rosclaw_soccer.rsi.taskspace_swing_evidence import audit_taskspace_swing_trace
from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch
from rosclaw_soccer.sim.contracts import hash_json

SEEDS = tuple(range(20260958, 20260966))
ARMS = ("acquisition_055", "acquisition_095", "acquisition_095_leadn004")


def load_bank(root: Path) -> tuple[dict[str, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    bank: dict[str, Any] = json.loads((root / "bank_summary.json").read_text(encoding="utf-8"))
    if (
        bank.get("schema") != "rsi_independent_first_touch_option_bank_v2"
        or bank.get("activation_ceiling") != "SIM_ONLY"
        or bank.get("promotion_authorized") is not False
        or bank.get("failures") != []
        or bank.get("report_hash")
        != hash_json({key: value for key, value in bank.items() if key != "report_hash"})
        or [(row["seed"], row["lane"]) for row in bank.get("episodes", [])]
        != [(seed, lane) for seed in SEEDS for lane in range(0, 16, 2)]
    ):
        raise ValueError("incomplete or uncommitted independent option curriculum")
    features = []
    rewards = []
    for episode in bank["episodes"]:
        if set(episode["arms"]) != set(ARMS):
            raise ValueError("missing causal counterfactual option")
        pair_features = None
        option_rewards = []
        for arm in ARMS:
            folder = root / f"seed{episode['seed']}-lane{episode['lane']}-{arm}"
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            physical = audit_vector_first_touch(folder)
            summary = episode["arms"][arm]
            expected_gap = 0.55 if arm == "acquisition_055" else 0.95
            expected_lead = -0.04 if arm == "acquisition_095_leadn004" else 0.0
            if (
                report.get("source_hash") != bank["source_hash"]
                or report.get("asset_hash") != bank["asset_hash"]
                or report.get("late_swing_actor_hash") != bank["actor_hash"]
                or report.get("late_swing_side_acquisition_gap_m") != expected_gap
                or report.get("taskspace_lateral_lead_m") != expected_lead
                or report["report_hash"] != summary["report_hash"]
                or physical["report_hash"] != summary["physical_audit_hash"]
                or report["environments"][0]["course"] != summary["course"]
            ):
                raise ValueError(f"bank physics receipt drift: {folder}")
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
                or any(summary[key] != value for key, value in displacement.items())
            ):
                raise ValueError(f"bank reward labels drift: {folder}")
            with np.load(folder / "late_swing_action_trace.npz", allow_pickle=False) as replay:
                action = audit_taskspace_swing_trace(replay, report, frames=300, count=1)
                current = np.asarray(replay["frame30_gate_features"][0], dtype=float)
                if action != summary["action_audit"]:
                    raise ValueError(f"bank action receipt drift: {folder}")
                if pair_features is None:
                    pair_features = current
                elif not np.allclose(pair_features, current, atol=1e-10, rtol=0):
                    raise ValueError("action arms differed before first decision")
            option_rewards.append(
                first_touch_reward(
                    {**summary, "max_lateral_excursion_m": summary["maximum_lateral_excursion_m"]}
                )
            )
        if pair_features is None or pair_features.shape != (9,):
            raise ValueError("causal frame-30 feature missing")
        features.append(pair_features)
        rewards.append(option_rewards)
    return bank, np.asarray(features), np.asarray(rewards)


def cross_validate(
    bank: dict[str, Any], features: np.ndarray[Any, Any], rewards: np.ndarray[Any, Any]
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for test_seed in SEEDS:
        test = np.asarray([episode["seed"] == test_seed for episode in bank["episodes"]])
        train = ~test
        model = fit_conservative_option_value(features[train], rewards[train])
        indices = np.flatnonzero(test)
        choices = [model.choose(features[i]) for i in indices]
        selected = [
            bank["episodes"][int(i)]["arms"][ARMS[choice]]
            for i, choice in zip(indices, choices, strict=True)
        ]
        fixed = [bank["episodes"][int(i)]["arms"][ARMS[model.fixed]] for i in indices]
        selected_rewards = [rewards[i, choice] for i, choice in zip(indices, choices, strict=True)]
        fixed_rewards = rewards[test, model.fixed]
        folds.append(
            {
                "test_seed": test_seed,
                "fixed_training_arm": ARMS[model.fixed],
                "choices": [ARMS[choice] for choice in choices],
                "mean_selected_reward": float(np.mean(selected_rewards)),
                "mean_fixed_reward": float(np.mean(fixed_rewards)),
                "clean_selected": sum(row["clean_foot_only"] for row in selected),
                "clean_fixed": sum(row["clean_foot_only"] for row in fixed),
                "out_of_play_selected": sum(
                    row["maximum_lateral_excursion_m"] > 4 for row in selected
                ),
                "out_of_play_fixed": sum(row["maximum_lateral_excursion_m"] > 4 for row in fixed),
            }
        )
    wins = sum(fold["mean_selected_reward"] > fold["mean_fixed_reward"] for fold in folds)
    safe = all(
        fold["clean_selected"] >= fold["clean_fixed"]
        and fold["out_of_play_selected"] <= fold["out_of_play_fixed"]
        for fold in folds
    )
    passed = bool(wins >= 6 and safe)
    result: dict[str, Any] = {
        "schema": "rsi_independent_option_curriculum_cv_v1",
        "activation_ceiling": "SIM_ONLY",
        "bank_report_hash": bank["report_hash"],
        "model": "regularized_linear_option_value_with_conservative_fallback",
        "ridge_strength": RIDGE_STRENGTH,
        "gain_margin": GAIN_MARGIN,
        "maximum_nearest_distance": MAX_NEAREST_DISTANCE,
        "folds": folds,
        "winning_seed_folds": wins,
        "all_folds_safe": safe,
        "development_gate_passed": passed,
        "fresh_seed_20260966_authorized": passed,
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
        parser.error("complete audited bank and new output required")
    bank, features, rewards = load_bank(args.bank_root)
    result = cross_validate(bank, features, rewards)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"OPTION_CURRICULUM_CV={result['report_hash']}", flush=True)
    print(f"DEVELOPMENT_GATE_PASSED={result['development_gate_passed']}", flush=True)


if __name__ == "__main__":
    main()
