"""Audit and learn a cautious offline chooser from physical paired 3v3 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def causal_features(values: list[float]) -> np.ndarray[Any, Any]:
    raw = np.asarray(values, dtype=np.float64)
    if raw.shape != (24,) or not np.all(np.isfinite(raw)):
        raise ValueError("complete finite same-frame entry state required")
    ball = raw[6:9]
    feet = raw[12:18].reshape(2, 3)
    foot_velocity = raw[18:24].reshape(2, 3)
    if feet[0, 2] - feet[1, 2] >= 0.02:
        side = 0
    elif feet[1, 2] - feet[0, 2] >= 0.02:
        side = 1
    else:
        side = int(np.argmin(np.abs(feet[:, 1] - ball[1])))
    selected = feet[side]
    return np.asarray(
        (
            ball[0] - selected[0],
            ball[1] - selected[1],
            selected[2] - feet[1 - side, 2],
            raw[9],
            raw[3],
            raw[4],
            foot_velocity[side, 0],
            foot_velocity[side, 1],
        ),
        dtype=np.float64,
    )


def outcome_reward(row: dict[str, Any]) -> float:
    if not row["safe"]:
        return -5.0
    if row["useful_pass"]:
        return 2.0
    if row["foot_contact_frames"]:
        return 1.0
    return 0.0


def choose_arm(
    features: np.ndarray[Any, Any],
    train_features: np.ndarray[Any, Any],
    train_outcomes: dict[str, list[dict[str, Any]]],
    train_indices: list[int],
    arm_names: list[str],
    *,
    neighbors: int = 3,
) -> dict[str, Any]:
    if len(train_indices) < neighbors or features.shape != (8,):
        raise ValueError("insufficient causal training observations")
    scale = np.maximum(np.std(train_features[train_indices], axis=0), 0.05)
    distance = np.linalg.norm((train_features[train_indices] - features) / scale, axis=1)
    nearby = np.argsort(distance, kind="stable")[:neighbors]
    selected_indices = [train_indices[int(index)] for index in nearby]
    weights = 1.0 / np.maximum(distance[nearby], 0.1)
    weights /= weights.sum()
    estimates = {}
    for arm in arm_names:
        rewards = np.asarray(
            [outcome_reward(train_outcomes[arm][index]) for index in selected_indices]
        )
        estimates[arm] = float(weights @ rewards)
    ranked = sorted(arm_names, key=lambda arm: (-estimates[arm], arm))
    best = ranked[0]
    # A reward average cannot certify safety. Reject unless every nearest
    # physical episode for the selected action was safe.
    eligible = [
        arm
        for arm in ranked
        if estimates[arm] > 0
        and all(train_outcomes[arm][index]["safe"] for index in selected_indices)
    ]
    chosen = eligible[0] if eligible else None
    return {
        "arm": chosen,
        "highest_average_arm": best,
        "estimates": estimates,
        "neighbor_indices": selected_indices,
        "neighbor_distances": distance[nearby].tolist(),
        "abstained": chosen is None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_contextual_action_learner_protocol_v24"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("training_indices") != list(range(18))
        or protocol.get("internal_extrapolation_indices") != list(range(18, 24))
        or protocol.get("neighbors") != 3
        or protocol.get("unsafe_reward") != -5
        or protocol.get("useful_reward") != 2
        or protocol.get("safe_contact_reward") != 1
    ):
        raise ValueError("invalid frozen offline development protocol")
    reports: dict[str, dict[str, Any]] = {}
    for dataset in protocol["datasets"]:
        for name, expected_hash in dataset["arm_report_hashes"].items():
            if name in reports:
                raise ValueError("duplicate physical arm")
            report = json.loads((Path(dataset["root"]) / name / "report.json").read_text())
            if (
                report["report_hash"] != expected_hash
                or hash_json({key: value for key, value in report.items() if key != "report_hash"})
                != expected_hash
            ):
                raise ValueError("physical evidence hash mismatch")
            reports[name] = report
    if (
        len(reports) != 11
        or len({str(report["source_hashes"]) for report in reports.values()}) != 1
    ):
        raise ValueError("mixed physics source or incomplete action library")
    names = sorted(name for name in reports if name != "parent")
    parent_rows = reports["parent"]["rows"]
    if len(parent_rows) != 24 or any(len(report["rows"]) != 24 for report in reports.values()):
        raise ValueError("incomplete paired physical dataset")
    for index in range(24):
        if len({report["rows"][index]["entry"]["hash"] for report in reports.values()}) != 1:
            raise ValueError("actions do not share the same pre-intervention entry state")
        if len({report["rows"][index]["course"] for report in reports.values()}) != 1:
            raise ValueError("paired scenario identities differ")
    features = np.stack([causal_features(row["entry"]["values"]) for row in parent_rows])
    outcomes = {name: report["rows"] for name, report in reports.items()}
    trials = []
    for index in protocol["internal_extrapolation_indices"]:
        choice = choose_arm(
            features[index],
            features,
            outcomes,
            protocol["training_indices"],
            names,
            neighbors=protocol["neighbors"],
        )
        arm = choice["arm"]
        chosen_outcome = outcomes[arm][index] if arm is not None else None
        trials.append(
            {
                "course": parent_rows[index]["course"],
                "choice": choice,
                "selected_outcome": None
                if chosen_outcome is None
                else {
                    "safe": chosen_outcome["safe"],
                    "foot_contact": bool(chosen_outcome["foot_contact_frames"]),
                    "useful_pass": chosen_outcome["useful_pass"],
                    "outgoing_ball_vx_mps": chosen_outcome["outgoing_ball_vx_mps"],
                },
                "oracle_useful_pass_diagnostic_only": any(
                    outcomes[name][index]["useful_pass"] for name in names
                ),
            }
        )
    args.output_dir.mkdir(parents=True)
    report = {
        "schema": "rsi_team_contextual_action_learner_report_v24",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "source_hashes": reports["parent"]["source_hashes"],
        "entry_observation_parity": True,
        "development_scene_count": 24,
        "training_scene_count": 18,
        "internal_extrapolation_scene_count": 6,
        "overall_oracle_safe_contact_diagnostic_only": sum(
            any(
                outcomes[name][index]["safe"] and outcomes[name][index]["foot_contact_frames"]
                for name in names
            )
            for index in range(24)
        ),
        "overall_oracle_useful_pass_diagnostic_only": sum(
            any(outcomes[name][index]["useful_pass"] for name in names) for index in range(24)
        ),
        "internal_extrapolation_oracle_useful_pass_diagnostic_only": sum(
            trial["oracle_useful_pass_diagnostic_only"] for trial in trials
        ),
        "internal_extrapolation_selected_count": sum(
            not trial["choice"]["abstained"] for trial in trials
        ),
        "internal_extrapolation_safe_selected_count": sum(
            trial["selected_outcome"] is not None and trial["selected_outcome"]["safe"]
            for trial in trials
        ),
        "internal_extrapolation_useful_pass_count": sum(
            trial["selected_outcome"] is not None and trial["selected_outcome"]["useful_pass"]
            for trial in trials
        ),
        "trials": trials,
        "fresh_holdout": False,
        "policy_intervention_authorized": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_CONTEXTUAL_LEARNER="
        + json.dumps(
            {
                key: report[key]
                for key in (
                    "report_hash",
                    "overall_oracle_safe_contact_diagnostic_only",
                    "overall_oracle_useful_pass_diagnostic_only",
                    "internal_extrapolation_selected_count",
                    "internal_extrapolation_safe_selected_count",
                    "internal_extrapolation_useful_pass_count",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
