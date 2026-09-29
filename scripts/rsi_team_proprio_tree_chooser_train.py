"""Compare optional offline tree ensembles on sealed G1 action outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import ExtraTreesClassifier

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_contextual_neural_train import tally
from scripts.rsi_team_full_proprio_chooser_train import _physical_features
from scripts.rsi_team_large_context_chooser_train import feature_table


def _positive_probability(
    classifier: ExtraTreesClassifier,
    feature: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    outputs = classifier.predict_proba(feature)
    classes = classifier.classes_
    probability = np.zeros((len(feature), 6), dtype=np.float64)
    for index, (values, labels) in enumerate(zip(outputs, classes, strict=True)):
        positive = np.flatnonzero(labels == 1)
        if len(positive) == 1:
            probability[:, index] = values[:, positive[0]]
        elif len(positive) != 0:
            raise ValueError("invalid tree class label")
    return probability


def _select(
    *,
    train_feature: np.ndarray[Any, Any],
    train_label: np.ndarray[Any, Any],
    validation_feature: np.ndarray[Any, Any],
    depth: int,
    leaf: int,
    threshold: float,
    estimators: int,
    seeds: list[int],
) -> np.ndarray[Any, Any]:
    scores = []
    for label_index in range(3):
        predictions = []
        for seed in seeds:
            model = ExtraTreesClassifier(
                n_estimators=estimators,
                max_depth=depth,
                min_samples_leaf=leaf,
                max_features=0.8,
                random_state=seed,
                n_jobs=1,
            )
            model.fit(train_feature, train_label[:, :, label_index].astype(int))
            predictions.append(_positive_probability(model, validation_feature))
        scores.append(
            np.quantile(predictions, 0.2, axis=0)
            if label_index == 0
            else np.mean(predictions, axis=0)
        )
    safe, contact, useful = scores
    eligible = safe >= threshold
    value = np.where(eligible, useful + 0.25 * contact, -np.inf)
    return np.where(eligible.any(axis=1), np.argmax(value, axis=1), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("tree development output exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_proprio_tree_chooser_protocol_v47"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("feature_sets") != ["relative_feet_16", "legs_48", "full_proprio_89"]
        or protocol.get("max_depths") != [6, 10]
        or protocol.get("min_samples_leaf") != [3, 8]
        or protocol.get("safety_thresholds") != [0.65, 0.75]
        or protocol.get("estimators_per_forest") != 120
        or protocol.get("ensemble_seeds") != [11, 19]
        or protocol.get("internal_gate")
        != {
            "minimum_useful": 28,
            "minimum_safe_contact": 50,
            "maximum_unsafe": 14,
            "maximum_abstained": 32,
        }
    ):
        raise ValueError("invalid frozen tree chooser protocol")
    previous = json.loads(Path(protocol["prior_train_report"]).read_text())
    if (
        previous.get("report_hash") != protocol["prior_train_report_hash"]
        or hash_json({key: value for key, value in previous.items() if key != "report_hash"})
        != protocol["prior_train_report_hash"]
        or previous.get("internal_gate_passed") is not False
    ):
        raise ValueError("previous physical training evidence missing or changed")
    root = Path(__file__).parents[1]
    v46 = json.loads((root / protocol["prior_protocol"]).read_text())
    dataset_path = Path(v46["dataset_path"])
    audit_path = Path(v46["audit_report_path"])
    if (
        hash_bytes(dataset_path.read_bytes()) != v46["dataset_hash"]
        or hash_bytes(audit_path.read_bytes()) != v46["audit_report_file_hash"]
    ):
        raise ValueError("audited training data changed")
    audit = json.loads(audit_path.read_text())
    if audit.get("report_hash") != v46["audit_report_hash"]:
        raise ValueError("physical audit changed")
    with np.load(dataset_path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_entry"], dtype=np.float64)
        hashes = data["entry_hashes"].tolist()
        names = data["arm_names"].tolist()
        labels = np.stack(
            [np.asarray(data[key]) for key in ("safe", "safe_contact", "useful_pass")],
            axis=2,
        ).astype(np.float64)
    if raw.shape != (512, 24) or labels.shape != (512, 6, 3) or len(names) != 6:
        raise ValueError("invalid six-arm training labels")
    features = {
        "relative_feet_16": feature_table(raw)["relative_feet_16"],
        **_physical_features(
            root=Path(v46["physical_root"]), report=audit, raw=raw, entry_hashes=hashes
        ),
    }
    rows = []
    for name in protocol["feature_sets"]:
        table = features[name]
        for depth in protocol["max_depths"]:
            for leaf in protocol["min_samples_leaf"]:
                for threshold in protocol["safety_thresholds"]:
                    selected = _select(
                        train_feature=table[:384],
                        train_label=labels[:384],
                        validation_feature=table[384:],
                        depth=depth,
                        leaf=leaf,
                        threshold=threshold,
                        estimators=protocol["estimators_per_forest"],
                        seeds=protocol["ensemble_seeds"],
                    )
                    metrics = tally(selected, labels[384:])
                    gate = protocol["internal_gate"]
                    eligible = bool(
                        metrics["useful"] >= gate["minimum_useful"]
                        and metrics["safe_contact"] >= gate["minimum_safe_contact"]
                        and metrics["unsafe"] <= gate["maximum_unsafe"]
                        and metrics["abstained"] <= gate["maximum_abstained"]
                    )
                    rows.append(
                        {
                            "index": len(rows),
                            "feature_set": name,
                            "max_depth": depth,
                            "min_samples_leaf": leaf,
                            "safety_threshold": threshold,
                            "internal_validation": metrics,
                            "eligible_for_fresh_exam": eligible,
                            "arm_counts": {
                                arm: int((selected == index).sum())
                                for index, arm in enumerate(names)
                            },
                        }
                    )
                    print(
                        f"{name} depth={depth} leaf={leaf} threshold={threshold} "
                        f"useful={metrics['useful']} unsafe={metrics['unsafe']}",
                        flush=True,
                    )

    def rank(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
        score = row["internal_validation"]
        return (
            -score["useful"],
            score["unsafe"],
            -score["safe_contact"],
            score["abstained"],
            row["index"],
        )

    eligible_rows = sorted((row for row in rows if row["eligible_for_fresh_exam"]), key=rank)
    chosen = (eligible_rows or sorted(rows, key=rank))[0]
    result = {
        "schema": "rsi_team_proprio_tree_chooser_report_v47",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "prior_train_report_hash": previous["report_hash"],
        "dataset_hash": v46["dataset_hash"],
        "sklearn_version": sklearn_version,
        "internal_validation_previously_consumed": True,
        "rows": rows,
        "chosen_index": chosen["index"],
        "internal_gate_passed": bool(eligible_rows),
        "fresh_exam_authorized": bool(eligible_rows),
    }
    result["report_hash"] = hash_json(result)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_PROPRIO_TREE="
        + json.dumps(
            {
                "report_hash": result["report_hash"],
                "chosen": chosen,
                "internal_gate_passed": result["internal_gate_passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
