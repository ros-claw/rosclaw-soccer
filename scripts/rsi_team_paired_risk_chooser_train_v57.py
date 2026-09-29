"""Train a SIM_ONLY paired-action risk chooser on sealed six-G1 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import ExtraTreesClassifier

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_full_proprio_chooser_train import _physical_features
from scripts.rsi_team_large_context_chooser_train import feature_table


def _load_pair(
    protocol: dict[str, Any],
    prefix: str,
    count: int,
    *,
    pair_names: tuple[str, str, str] = ("parent", "baseline", "gate22_cap10"),
) -> tuple[dict[str, np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    audit_path = Path(protocol[f"{prefix}_audit_report_path"])
    dataset_path = Path(protocol[f"{prefix}_dataset_path"])
    if (
        hash_bytes(audit_path.read_bytes()) != protocol[f"{prefix}_audit_report_file_hash"]
        or hash_bytes(dataset_path.read_bytes()) != protocol[f"{prefix}_dataset_hash"]
    ):
        raise ValueError(f"{prefix} physical audit or dataset changed")
    report = json.loads(audit_path.read_text())
    if (
        report.get("report_hash")
        != hash_json({k: v for k, v in report.items() if k != "report_hash"})
        or report["report_hash"] != protocol[f"{prefix}_audit_report_hash"]
        or report.get("dataset_hash") != protocol[f"{prefix}_dataset_hash"]
        or report.get("scene_count") != count
    ):
        raise ValueError(f"{prefix} physical audit commitment invalid")
    with np.load(dataset_path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_entry"], dtype=np.float64)
        hashes: list[str] = data["entry_hashes"].tolist()
        names: list[str] = data["arm_names"].tolist()
        scenarios: list[str] = data["scenario_hashes"].tolist()
        labels = np.stack(
            [
                np.asarray(data[key], dtype=np.bool_)
                for key in ("safe", "safe_contact", "useful_pass")
            ],
            axis=2,
        )
    if (
        raw.shape != (count, 24)
        or labels.shape != (count, len(names), 3)
        or pair_names[0] != "parent"
        or not set(pair_names).issubset(names)
        or len(set(scenarios)) != count
        or len(hashes) != count
    ):
        raise ValueError(f"{prefix} paired physical table invalid")
    labels = labels[:, [names.index(name) for name in pair_names], :]
    physical = _physical_features(
        root=Path(protocol[f"{prefix}_physical_root"]), report=report, raw=raw, entry_hashes=hashes
    )
    return {
        "relative_feet_16": feature_table(raw)["relative_feet_16"],
        "full_proprio_89": physical["full_proprio_89"],
        "scenario_hashes": np.asarray(scenarios),
    }, labels


def _positive_probability(
    model: ExtraTreesClassifier, feature: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    classes = np.asarray(model.classes_)
    if classes.ndim != 1 or not np.isin(classes, (0, 1)).all():
        raise ValueError("invalid binary classifier classes")
    positive = np.flatnonzero(classes == 1)
    if len(positive) == 0:
        return np.zeros(len(feature), dtype=np.float64)
    return np.asarray(model.predict_proba(feature)[:, positive[0]], dtype=np.float64)


def _scores(
    train: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    validation: np.ndarray[Any, Any],
    stress: np.ndarray[Any, Any],
    *,
    depth: int,
    leaf: int,
    estimators: int,
    seeds: list[int],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    harm = labels[:, 1, 0] & ~labels[:, 2, 0]
    gain = labels[:, 2, 2] & ~labels[:, 1, 2]
    output: list[np.ndarray[Any, Any]] = []
    for feature in (validation, stress):
        output.append(np.zeros((len(feature), 2), dtype=np.float64))
    for label_index, target in enumerate((harm, gain)):
        predicted: list[list[np.ndarray[Any, Any]]] = [[], []]
        for seed in seeds:
            model = ExtraTreesClassifier(
                n_estimators=estimators,
                max_depth=depth,
                min_samples_leaf=leaf,
                max_features=0.8,
                random_state=seed,
                n_jobs=1,
            )
            model.fit(train, target.astype(np.int8))
            for output_index, feature in enumerate((validation, stress)):
                predicted[output_index].append(_positive_probability(model, feature))
        for output_index in range(2):
            ensemble = np.asarray(predicted[output_index])
            output[output_index][:, label_index] = (
                ensemble.max(axis=0) if label_index == 0 else ensemble.mean(axis=0)
            )
    return output[0], output[1]


def _tally(selected: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> dict[str, int]:
    if (
        selected.shape != (len(labels),)
        or not np.isin(selected, (1, 2)).all()
        or labels.shape != (len(selected), 3, 3)
    ):
        raise ValueError("invalid paired selection; baseline is not an abstention")
    chosen = labels[np.arange(len(labels)), selected]
    return {
        "safe": int(chosen[:, 0].sum()),
        "safe_contact": int(chosen[:, 1].sum()),
        "useful": int(chosen[:, 2].sum()),
        "unsafe": int((~chosen[:, 0]).sum()),
        "gate_count": int((selected == 2).sum()),
    }


def _choose(
    scores: np.ndarray[Any, Any], harm_threshold: float, gain_threshold: float
) -> np.ndarray[Any, Any]:
    if scores.ndim != 2 or scores.shape[1] != 2 or not np.isfinite(scores).all():
        raise ValueError("invalid risk/gain scores")
    return np.where((scores[:, 0] <= harm_threshold) & (scores[:, 1] >= gain_threshold), 2, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("paired-risk training output exists")
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("schema") != "rsi_team_paired_risk_chooser_protocol_v57"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("feature_sets") != ["relative_feet_16", "full_proprio_89"]
        or protocol.get("max_depths") != [6, 10]
        or protocol.get("min_samples_leaf") != [3, 8]
        or protocol.get("harm_thresholds") != [0.04, 0.08, 0.15, 0.25]
        or protocol.get("gain_thresholds") != [0.0, 0.10]
        or protocol.get("estimators_per_forest") != 120
        or protocol.get("ensemble_seeds") != [11, 19]
        or protocol.get("train_scenes") != 384
        or protocol.get("consumed_internal_validation_scenes") != 128
        or protocol.get("consumed_external_stress_scenes") != 64
        or protocol.get("internal_gate")
        != {"minimum_safe": 112, "minimum_safe_contact": 70, "minimum_useful": 36}
        or protocol.get("stress_gate") != {"minimum_safe": 58, "minimum_useful": 15}
    ):
        raise ValueError("invalid frozen v57 paired-risk protocol")
    training, labels = _load_pair(protocol, "training", 512)
    stress, stress_labels = _load_pair(protocol, "stress", 64)
    if set(training["scenario_hashes"].tolist()) & set(stress["scenario_hashes"].tolist()):
        raise ValueError("training/stress scenarios overlap")
    rows: list[dict[str, Any]] = []
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
                for harm_threshold in protocol["harm_thresholds"]:
                    for gain_threshold in protocol["gain_thresholds"]:
                        internal = _tally(
                            _choose(validation_scores, harm_threshold, gain_threshold), labels[384:]
                        )
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
                                "internal_gate_passed": eligible,
                                "stress_diagnostic": _tally(
                                    _choose(stress_scores, harm_threshold, gain_threshold),
                                    stress_labels,
                                ),
                            }
                        )

    def rank(row: dict[str, Any]) -> tuple[int, int, int, int]:
        score = row["internal_validation"]
        return (-score["safe"], -score["useful"], -score["safe_contact"], row["index"])

    eligible_rows = sorted((r for r in rows if r["internal_gate_passed"]), key=rank)
    chosen = (eligible_rows or sorted(rows, key=rank))[0]
    stress_gate = protocol["stress_gate"]
    external_pass = bool(
        chosen["stress_diagnostic"]["safe"] >= stress_gate["minimum_safe"]
        and chosen["stress_diagnostic"]["useful"] >= stress_gate["minimum_useful"]
    )
    report: dict[str, Any] = {
        "schema": "rsi_team_paired_risk_chooser_train_report_v57",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "training_dataset_hash": protocol["training_dataset_hash"],
        "stress_dataset_hash": protocol["stress_dataset_hash"],
        "sklearn_version": sklearn_version,
        "training_scenes": 384,
        "consumed_internal_validation_scenes": 128,
        "consumed_external_stress_scenes": 64,
        "rows": rows,
        "chosen_index": chosen["index"],
        "internal_gate_passed": bool(eligible_rows),
        "consumed_stress_gate_passed": external_pass,
        "fresh_exam_authorized": bool(eligible_rows) and external_pass,
    }
    report["report_hash"] = hash_json(report)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        "RSI_TEAM_PAIRED_RISK="
        + json.dumps(
            {
                "report_hash": report["report_hash"],
                "chosen": chosen,
                "fresh_exam_authorized": report["fresh_exam_authorized"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
