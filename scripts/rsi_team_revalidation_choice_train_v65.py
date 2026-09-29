"""Train a SIM_ONLY old-versus-revalidated swing-foot risk chooser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_paired_risk_chooser_train_v57 import _choose, _load_pair, _scores, _tally


def _newly_unsafe(selected: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> int:
    if selected.shape != (len(labels),) or labels.shape != (len(selected), 3, 3):
        raise ValueError("invalid paired safety comparison")
    return int((labels[:, 1, 0] & ~labels[np.arange(len(labels)), selected, 0]).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("revalidation choice output exists")
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("schema") != "rsi_team_revalidation_choice_protocol_v65"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("pair_names") != ["parent", "gate22_cap10", "gate22_revalidate"]
        or protocol.get("feature_sets") != ["relative_feet_16", "full_proprio_89"]
        or protocol.get("max_depths") != [6, 10]
        or protocol.get("min_samples_leaf") != [3, 8]
        or protocol.get("harm_thresholds") != [0.02, 0.05, 0.10, 0.20]
        or protocol.get("gain_thresholds") != [0.0, 0.10]
        or protocol.get("estimators_per_forest") != 120
        or protocol.get("ensemble_seeds") != [11, 19]
        or protocol.get("train_scenes") != 384
        or protocol.get("consumed_internal_validation_scenes") != 128
        or protocol.get("consumed_external_stress_scenes") != 64
        or protocol.get("internal_gate")
        != {"minimum_safe": 119, "minimum_safe_contact": 94, "minimum_useful": 65}
        or protocol.get("stress_gate")
        != {"minimum_safe": 59, "minimum_useful": 33, "maximum_newly_unsafe": 0}
    ):
        raise ValueError("invalid frozen v65 revalidation-choice protocol")
    pair = tuple(protocol["pair_names"])
    training, labels = _load_pair(protocol, "training", 512, pair_names=pair)
    stress, stress_labels = _load_pair(protocol, "stress", 64, pair_names=pair)
    if set(training["scenario_hashes"].tolist()) & set(stress["scenario_hashes"].tolist()):
        raise ValueError("training and stress scenarios overlap")
    rows: list[dict[str, Any]] = []
    score_cache: dict[tuple[str, int, int], tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = {}
    for name in protocol["feature_sets"]:
        for depth in protocol["max_depths"]:
            for leaf in protocol["min_samples_leaf"]:
                validation_scores, stress_scores = _scores(
                    training[name][:384],
                    labels[:384],
                    training[name][384:],
                    stress[name],
                    depth=depth,
                    leaf=leaf,
                    estimators=120,
                    seeds=[11, 19],
                )
                score_cache[name, depth, leaf] = validation_scores, stress_scores
                for harm_threshold in protocol["harm_thresholds"]:
                    for gain_threshold in protocol["gain_thresholds"]:
                        selected = _choose(validation_scores, harm_threshold, gain_threshold)
                        internal = _tally(selected, labels[384:])
                        gate = protocol["internal_gate"]
                        eligible = bool(
                            internal["safe"] >= gate["minimum_safe"]
                            and internal["safe_contact"] >= gate["minimum_safe_contact"]
                            and internal["useful"] >= gate["minimum_useful"]
                        )
                        rows.append(
                            {
                                "index": len(rows),
                                "feature_set": name,
                                "max_depth": depth,
                                "min_samples_leaf": leaf,
                                "harm_threshold": harm_threshold,
                                "gain_threshold": gain_threshold,
                                "internal_validation": internal,
                                "internal_newly_unsafe": _newly_unsafe(selected, labels[384:]),
                                "internal_gate_passed": eligible,
                            }
                        )

    def rank(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
        score = row["internal_validation"]
        return (
            row["internal_newly_unsafe"],
            -score["safe"],
            -score["useful"],
            -score["safe_contact"],
            row["index"],
        )

    eligible_rows = sorted((row for row in rows if row["internal_gate_passed"]), key=rank)
    chosen = (eligible_rows or sorted(rows, key=rank))[0]
    _, stress_scores = score_cache[
        chosen["feature_set"], chosen["max_depth"], chosen["min_samples_leaf"]
    ]
    stress_selected = _choose(stress_scores, chosen["harm_threshold"], chosen["gain_threshold"])
    stress_metrics = _tally(stress_selected, stress_labels)
    newly_unsafe = _newly_unsafe(stress_selected, stress_labels)
    gate = protocol["stress_gate"]
    stress_pass = bool(
        stress_metrics["safe"] >= gate["minimum_safe"]
        and stress_metrics["useful"] >= gate["minimum_useful"]
        and newly_unsafe <= gate["maximum_newly_unsafe"]
    )
    report: dict[str, Any] = {
        "schema": "rsi_team_revalidation_choice_train_report_v65",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "training_dataset_hash": protocol["training_dataset_hash"],
        "stress_dataset_hash": protocol["stress_dataset_hash"],
        "rows": rows,
        "chosen_index": chosen["index"],
        "internal_gate_passed": bool(eligible_rows),
        "consumed_stress": stress_metrics,
        "consumed_stress_newly_unsafe": newly_unsafe,
        "consumed_stress_gate_passed": stress_pass,
        "fresh_exam_authorized": bool(eligible_rows) and stress_pass,
    }
    report["report_hash"] = hash_json(report)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        "RSI_TEAM_REVALIDATION_CHOICE="
        + json.dumps(
            {
                "report_hash": report["report_hash"],
                "chosen": chosen,
                "consumed_stress": stress_metrics,
                "consumed_stress_newly_unsafe": newly_unsafe,
                "fresh_exam_authorized": report["fresh_exam_authorized"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
