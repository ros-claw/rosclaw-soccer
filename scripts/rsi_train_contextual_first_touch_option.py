"""Cross-validate a SIM_ONLY option selector on audited independent Isaac courses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contextual_first_touch_option import (
    ACTION_NAMES,
    first_touch_reward,
    fit_contextual_option,
)
from rosclaw_soccer.rsi.independent_first_touch_bank import post_contact_displacement
from rosclaw_soccer.rsi.taskspace_swing_evidence import audit_taskspace_swing_trace
from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch
from rosclaw_soccer.sim.contracts import hash_json


def load_verified_bank(
    root: Path,
) -> tuple[dict[str, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Recheck raw physics and action receipts before using an episode as a label."""
    path = root / "bank_summary.json"
    bank: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if (
        bank.get("schema") != "rsi_independent_first_touch_bank_v1"
        or bank.get("activation_ceiling") != "SIM_ONLY"
        or bank.get("promotion_authorized") is not False
        or bank.get("failures") != []
        or not isinstance(bank.get("source_hash"), str)
        or not bank["source_hash"].startswith("sha256:")
        or bank.get("report_hash")
        != hash_json({key: value for key, value in bank.items() if key != "report_hash"})
        or not isinstance(bank.get("episodes"), list)
        or len(bank["episodes"]) != 16
        or [(row["seed"], row["lane"]) for row in bank["episodes"]]
        != [(seed, lane) for seed in (20260953, 20260954) for lane in range(0, 16, 2)]
    ):
        raise ValueError("incomplete or uncommitted independent bank")
    features = []
    rewards = []
    for row in bank["episodes"]:
        seed, lane = row["seed"], row["lane"]
        if set(row["arms"]) != set(ACTION_NAMES):
            raise ValueError("missing counterfactual first-touch arm")
        course_features = None
        course_rewards = []
        for arm in ACTION_NAMES:
            folder = root / f"seed{seed}-lane{lane}-{arm}"
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            physical = audit_vector_first_touch(folder)
            summary_arm = row["arms"][arm]
            if (
                physical["report_hash"] != summary_arm["audit_hash"]
                or report["report_hash"] != summary_arm["report_hash"]
                or report["source_hash"] != bank["source_hash"]
                or report["environments"][0]["course"] != summary_arm["course"]
            ):
                raise ValueError(f"bank physics receipt drifted: {folder}")
            entry = report["environments"][0]
            with np.load(folder / "trace.npz", allow_pickle=False) as physics:
                displacement = post_contact_displacement(
                    physics["ball_position_m"][:, 0], entry["first_contact_frame"]
                )
            bodies = entry["contact_body_indices"]
            if (
                summary_arm["first_contact_frame"] != entry["first_contact_frame"]
                or summary_arm["contact_body_indices"] != bodies
                or summary_arm["clean_foot_only"] != bool(bodies and set(bodies) <= {0, 1})
                or summary_arm["minimum_pelvis_z_m"] != entry["minimum_pelvis_z_m"]
                or summary_arm["gate_selected"] != report.get("selected_taskspace_mask", [False])[0]
                or any(summary_arm[key] != value for key, value in displacement.items())
            ):
                raise ValueError(f"bank reward labels drifted from physics: {folder}")
            if arm != "parent":
                with np.load(folder / "late_swing_action_trace.npz", allow_pickle=False) as replay:
                    action = audit_taskspace_swing_trace(replay, report, frames=300, count=1)
                    current_features = np.asarray(replay["frame30_gate_features"][0], dtype=float)
                    if action != summary_arm["action_audit"]:
                        raise ValueError(f"bank action receipt drifted: {folder}")
                    if course_features is None:
                        course_features = current_features
                    elif not np.allclose(course_features, current_features, atol=1e-10, rtol=0):
                        raise ValueError("counterfactuals used different precontact body state")
            course_rewards.append(first_touch_reward(summary_arm))
        if course_features is None or course_features.shape != (9,):
            raise ValueError("missing causal precontact features")
        features.append(course_features)
        rewards.append(course_rewards)
    return bank, np.asarray(features), np.asarray(rewards)


def cross_validate(
    bank: dict[str, Any], features: np.ndarray[Any, Any], rewards: np.ndarray[Any, Any]
) -> dict[str, Any]:
    """Two seed-held-out folds; never train on that fold's physical outcomes."""
    folds: list[dict[str, Any]] = []
    total_rescues = 0
    all_safe = True
    for train_seed, test_seed in ((20260953, 20260954), (20260954, 20260953)):
        train = np.asarray([row["seed"] == train_seed for row in bank["episodes"]])
        test = ~train
        selector = fit_contextual_option(features[train], rewards[train])
        fixed = int(np.argmax(np.mean(rewards[train], axis=0)))
        test_indices = np.flatnonzero(test)
        choices = [selector.choose(features[i]) for i in test_indices]
        selected = np.asarray(
            [rewards[i, choice] for i, choice in zip(test_indices, choices, strict=True)]
        )
        parent = rewards[test, 0]
        fixed_reward = rewards[test, fixed]
        clean_parent = 0
        clean_selected = 0
        rescues = 0
        decisions = []
        for i, choice in zip(test_indices, choices, strict=True):
            episode = bank["episodes"][int(i)]
            parent_arm = episode["arms"]["parent"]
            selected_arm = episode["arms"][ACTION_NAMES[choice]]
            parent_clean = bool(parent_arm["clean_foot_only"])
            selected_clean = bool(selected_arm["clean_foot_only"])
            clean_parent += parent_clean
            clean_selected += selected_clean
            rescues += not parent_clean and selected_clean
            all_safe &= selected_arm["minimum_pelvis_z_m"] >= 0.65
            decisions.append(
                {
                    "seed": episode["seed"],
                    "lane": episode["lane"],
                    "choice": ACTION_NAMES[choice],
                    "selected_reward": float(rewards[i, choice]),
                    "parent_reward": float(rewards[i, 0]),
                }
            )
        total_rescues += rescues
        folds.append(
            {
                "train_seed": train_seed,
                "test_seed": test_seed,
                "fixed_training_arm": ACTION_NAMES[fixed],
                "mean_selected_reward": float(np.mean(selected)),
                "mean_parent_reward": float(np.mean(parent)),
                "mean_fixed_reward": float(np.mean(fixed_reward)),
                "clean_parent": clean_parent,
                "clean_selected": clean_selected,
                "rescues": rescues,
                "decisions": decisions,
            }
        )
    passed = bool(
        all_safe
        and total_rescues >= 2
        and all(
            fold["mean_selected_reward"] > fold["mean_parent_reward"]
            and fold["mean_selected_reward"] > fold["mean_fixed_reward"]
            and fold["clean_selected"] >= fold["clean_parent"]
            for fold in folds
        )
    )
    result: dict[str, Any] = {
        "schema": "rsi_contextual_first_touch_option_crossvalidation_v1",
        "activation_ceiling": "SIM_ONLY",
        "bank_report_hash": bank["report_hash"],
        "folds": folds,
        "total_rescues": total_rescues,
        "all_selected_safe": all_safe,
        "development_gate_passed": passed,
        "fresh_holdout_authorized": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.bank_root.is_dir() or args.output.exists():
        parser.error("audited bank and new output file required")
    bank, features, rewards = load_verified_bank(args.bank_root)
    result = cross_validate(bank, features, rewards)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CONTEXTUAL_OPTION_CV={result['report_hash']}", flush=True)
    print(f"DEVELOPMENT_GATE_PASSED={result['development_gate_passed']}", flush=True)


if __name__ == "__main__":
    main()
