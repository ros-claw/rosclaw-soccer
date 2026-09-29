"""Retrain the frozen v47 winner and export bounded NumPy-only tree inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import ExtraTreesClassifier

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.rsi.team_proprio_tree_chooser import TeamProprioTreeChooser
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_contextual_neural_train import tally
from scripts.rsi_team_full_proprio_chooser_train import _physical_features
from scripts.rsi_team_proprio_tree_chooser_train import _positive_probability, _select


def _forest_arrays(
    features: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    *,
    depth: int,
    leaf: int,
    estimators: int,
    seeds: list[int],
) -> dict[str, np.ndarray[Any, Any]]:
    tree_offsets = [0]
    forest_offsets = [0]
    fields: list[np.ndarray[Any, Any]] = []
    thresholds: list[np.ndarray[Any, Any]] = []
    lefts: list[np.ndarray[Any, Any]] = []
    rights: list[np.ndarray[Any, Any]] = []
    positives: list[np.ndarray[Any, Any]] = []
    for label_index in range(3):
        for seed in seeds:
            model = ExtraTreesClassifier(
                n_estimators=estimators,
                max_depth=depth,
                min_samples_leaf=leaf,
                max_features=0.8,
                random_state=seed,
                n_jobs=1,
            )
            model.fit(features, labels[:, :, label_index].astype(int))
            # Export must retain the exact forest probabilities, not labels.
            if _positive_probability(model, features[:2]).shape != (2, 6):
                raise ValueError("invalid fitted multi-output tree probabilities")
            for estimator in model.estimators_:
                tree = estimator.tree_
                start = tree_offsets[-1]
                count = tree.node_count
                if tree.max_depth > depth:
                    raise ValueError("forest exceeded frozen maximum depth")
                fields.append(np.asarray(tree.feature, dtype=np.int16))
                thresholds.append(np.asarray(tree.threshold, dtype=np.float64))
                lefts.append(
                    np.where(tree.children_left < 0, -1, tree.children_left + start).astype(
                        np.int64
                    )
                )
                rights.append(
                    np.where(tree.children_right < 0, -1, tree.children_right + start).astype(
                        np.int64
                    )
                )
                value = np.zeros((count, 6), dtype=np.float64)
                for arm, classes in enumerate(model.classes_):
                    positive = np.flatnonzero(classes == 1)
                    if len(positive) == 1:
                        value[:, arm] = tree.value[:, arm, positive[0]]
                    elif len(positive) != 0:
                        raise ValueError("invalid fitted class labels")
                positives.append(value)
                tree_offsets.append(start + count)
            forest_offsets.append(len(tree_offsets) - 1)
    return {
        "tree_offsets": np.asarray(tree_offsets, dtype=np.int64),
        "forest_offsets": np.asarray(forest_offsets, dtype=np.int64),
        "feature": np.concatenate(fields),
        "threshold": np.concatenate(thresholds),
        "left": np.concatenate(lefts),
        "right": np.concatenate(rights),
        "positive_probability": np.concatenate(positives),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-report", required=True, type=Path)
    parser.add_argument("--train-protocol", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("tree export output exists")
    report = json.loads(args.train_report.read_text())
    protocol = json.loads(args.train_protocol.read_text())
    expected_report = "sha256:1f07bb338b86e5adbcc421282543475c8f79595fdf5331d5fcc64959381a1587"
    if (
        report.get("report_hash") != expected_report
        or hash_json({key: value for key, value in report.items() if key != "report_hash"})
        != expected_report
        or report.get("internal_gate_passed") is not True
        or report.get("fresh_exam_authorized") is not True
        or report.get("chosen_index") != 21
        or report.get("sklearn_version") != sklearn_version
        or report.get("protocol_hash") != hash_bytes(args.train_protocol.read_bytes())
    ):
        raise ValueError("unsealed or incompatible v47 trained winner")
    chosen = report["rows"][21]
    if (
        chosen.get("feature_set") != "full_proprio_89"
        or chosen.get("max_depth") != 10
        or chosen.get("min_samples_leaf") != 3
        or chosen.get("safety_threshold") != 0.75
        or chosen.get("eligible_for_fresh_exam") is not True
    ):
        raise ValueError("v47 selected a different model")
    root = Path(__file__).parents[1]
    v46 = json.loads((root / protocol["prior_protocol"]).read_text())
    dataset_path = Path(v46["dataset_path"])
    audit_path = Path(v46["audit_report_path"])
    if (
        hash_bytes(dataset_path.read_bytes()) != v46["dataset_hash"]
        or hash_bytes(audit_path.read_bytes()) != v46["audit_report_file_hash"]
    ):
        raise ValueError("physical training data changed")
    audit = json.loads(audit_path.read_text())
    if audit.get("report_hash") != v46["audit_report_hash"]:
        raise ValueError("physical audit changed")
    qualification = qualify_g1_assets(args.asset_root)
    qualification.require_eligible()
    if qualification.body_hash != audit["body_hash"]:
        raise ValueError("physical G1 body does not match audited training data")
    with np.load(dataset_path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_entry"], dtype=np.float64)
        hashes = data["entry_hashes"].tolist()
        labels = np.stack(
            [np.asarray(data[key]) for key in ("safe", "safe_contact", "useful_pass")],
            axis=2,
        ).astype(np.float64)
        names = data["arm_names"].tolist()
    feature = _physical_features(
        root=Path(v46["physical_root"]), report=audit, raw=raw, entry_hashes=hashes
    )["full_proprio_89"]
    selected = _select(
        train_feature=feature[:384],
        train_label=labels[:384],
        validation_feature=feature[384:],
        depth=10,
        leaf=3,
        threshold=0.75,
        estimators=120,
        seeds=[11, 19],
    )
    if tally(selected, labels[384:]) != chosen["internal_validation"]:
        raise ValueError("v47 training outcome did not reproduce")
    arrays = _forest_arrays(
        feature[:384], labels[:384], depth=10, leaf=3, estimators=120, seeds=[11, 19]
    )
    args.output_dir.mkdir(parents=True)
    forest_path = args.output_dir / "forest.npz"
    np.savez_compressed(forest_path, **arrays)
    curriculum = json.loads(
        (root / "docs/rsi/protocols/team-diverse-curriculum-v38.json").read_text()
    )
    arms = {arm["name"]: arm for arm in curriculum["arms"] if arm["name"] in names}
    adaptive = json.loads(Path(curriculum["adaptive_model_path"]).read_text())
    if (
        set(arms) != set(names)
        or adaptive.get("model_hash") != curriculum["adaptive_model_hash"]
        or hash_json({key: value for key, value in adaptive.items() if key != "model_hash"})
        != curriculum["adaptive_model_hash"]
    ):
        raise ValueError("frozen navigation arm family changed")
    manifest = {
        "schema": "rsi_team_proprio_tree_chooser_model_v47",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "train_report_hash": expected_report,
        "training_dataset_hash": v46["dataset_hash"],
        "body_hash": audit["body_hash"],
        "feature_set": "full_proprio_89",
        "feature_count": 89,
        "forest_file": "forest.npz",
        "forest_hash": hash_bytes(forest_path.read_bytes()),
        "estimators_per_forest": 120,
        "ensemble_seeds": [11, 19],
        "safety_threshold": 0.75,
        "safety_quantile": 0.2,
        "arm_names": names,
        "arm_parameters": arms,
        "adaptive_model_hash": curriculum["adaptive_model_hash"],
        "adaptive_model_parameters": adaptive["parameters"],
    }
    manifest["model_hash"] = hash_json(manifest)
    model_path = args.output_dir / "model.json"
    model_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    runtime = TeamProprioTreeChooser(
        agent_id="red.playmaker",
        foundation_hash=hash_bytes(
            (args.asset_root / "policy/loco_mode/model/policy_29dof.pt").read_bytes()
        ),
        foundation_config_hash=hash_bytes(
            (args.asset_root / "policy/loco_mode/config/LocoMode.yaml").read_bytes()
        ),
        model_path=model_path,
    )
    runtime_arms = np.asarray(
        [
            -1 if (name := runtime.choose_feature(row)) is None else names.index(name)
            for row in feature[384:]
        ],
        dtype=np.int64,
    )
    if not np.array_equal(runtime_arms, selected):
        raise ValueError("pure NumPy runtime disagrees with training decisions")
    result = {
        "schema": "rsi_team_proprio_tree_export_report_v47",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "train_report_hash": expected_report,
        "model_hash": manifest["model_hash"],
        "forest_hash": manifest["forest_hash"],
        "validation_decisions_matched": 128,
        "validation_metrics": tally(runtime_arms, labels[384:]),
    }
    result["report_hash"] = hash_json(result)
    (args.output_dir / "export_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("RSI_TEAM_PROPRIO_TREE_EXPORT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
