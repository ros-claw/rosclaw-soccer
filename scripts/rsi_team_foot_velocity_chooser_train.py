"""Freeze a measured-foot velocity chooser from sealed SIM_ONLY physical episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_contextual_action_dataset import entry_features
from scripts.rsi_team_contextual_neural_retrain import train_candidate
from scripts.rsi_team_contextual_neural_train import tally


def measured_features(
    *,
    root: Path,
    version: str,
    dataset: Any,
    indices: range,
    feature_indices: list[int],
) -> np.ndarray:
    values = []
    for index in indices:
        path = (
            root
            / f"rsi-team-diverse-curriculum-{version}"
            / "parent"
            / f"t{index:03d}"
            / "parent"
            / "taskspace_trace.npz"
        )
        entry = entry_features(path, 30)
        if entry["hash"] != str(dataset["entry_hashes"][index]):
            raise ValueError("unsealed measured-foot entry features")
        values.append(np.asarray(entry["values"])[feature_indices])
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("foot-velocity chooser output exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    architecture = protocol.get("architecture", {})
    if (
        protocol.get("schema") != "rsi_team_foot_velocity_chooser_protocol_v33"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("feature_indices_in_raw_entry") != list(range(12, 24))
        or architecture.get("width") != 24
        or architecture.get("epochs") != 50
        or architecture.get("ensemble_seeds") != [11, 19, 29]
        or protocol.get("internal_gate")
        != {"maximum_unsafe_equal_to_parent": 3, "minimum_useful": 5}
    ):
        raise ValueError("invalid committed measured-foot chooser protocol")
    roots = Path(protocol["physical_result_root"])
    dataset = []
    for version in ("v29", "v31"):
        path = Path(protocol[f"{version}_dataset_path"])
        if hash_bytes(path.read_bytes()) != protocol[f"{version}_dataset_hash"]:
            raise ValueError("physical dataset hash mismatch")
        dataset.append(np.load(path, allow_pickle=False))
    a, b = dataset
    arms = protocol["eligible_arms"]
    columns_a = [a["arm_names"].tolist().index(name) for name in arms]
    columns_b = [b["arm_names"].tolist().index(name) for name in arms]
    features_a = measured_features(
        root=roots,
        version="v29",
        dataset=a,
        indices=range(64),
        feature_indices=protocol["feature_indices_in_raw_entry"],
    )
    features_b = measured_features(
        root=roots,
        version="v31",
        dataset=b,
        indices=range(128),
        feature_indices=protocol["feature_indices_in_raw_entry"],
    )
    labels_a = np.stack(
        [a[key][:, columns_a] for key in ("safe", "safe_contact", "useful_pass")], axis=2
    ).astype(np.float64)
    labels_b = np.stack(
        [b[key][:, columns_b] for key in ("safe", "safe_contact", "useful_pass")], axis=2
    ).astype(np.float64)
    train_feature = np.concatenate((features_a[:48], features_b))
    train_label = np.concatenate((labels_a[:48], labels_b))
    validation_feature = features_a[48:]
    validation_label = labels_a[48:]
    torch.set_num_threads(1)
    specification = {
        "feature_count": 12,
        "epochs": architecture["epochs"],
        "safety_threshold": architecture["safety_threshold"],
    }
    model, chosen = train_candidate(
        train_feature=train_feature,
        train_label=train_label,
        validation_feature=validation_feature,
        candidate=specification,
        seeds=architecture["ensemble_seeds"],
        width=architecture["width"],
    )
    validation = tally(chosen, validation_label)
    parent_report = json.loads(
        (roots / "rsi-team-diverse-curriculum-v29/parent/report.json").read_text(encoding="utf-8")
    )
    parent_unsafe = sum(not row["safe"] for row in parent_report["rows"][48:])
    passed = bool(
        validation["unsafe"] <= parent_unsafe
        and validation["useful"] >= protocol["internal_gate"]["minimum_useful"]
    )
    course_path = Path(__file__).parents[1] / "docs/rsi/protocols/team-diverse-curriculum-v31.json"
    course = json.loads(course_path.read_text(encoding="utf-8"))
    selected_arms = {arm["name"]: arm for arm in course["arms"] if arm["name"] in arms}
    if set(selected_arms) != set(arms):
        raise ValueError("incomplete arm contract")
    sealed = {
        "schema": "rsi_team_foot_velocity_chooser_model_v33",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "training_dataset_hashes": [
            protocol["v29_dataset_hash"],
            protocol["v31_dataset_hash"],
        ],
        "feature_order": [
            "left_foot_x",
            "left_foot_y",
            "left_foot_z",
            "right_foot_x",
            "right_foot_y",
            "right_foot_z",
            "left_foot_vx",
            "left_foot_vy",
            "left_foot_vz",
            "right_foot_vx",
            "right_foot_vy",
            "right_foot_vz",
        ],
        "arm_names": arms,
        "arm_parameters": selected_arms,
        "adaptive_model_hash": course["adaptive_model_hash"],
        "adaptive_model_parameters": json.loads(Path(course["adaptive_model_path"]).read_text())[
            "parameters"
        ],
        **model,
    }
    sealed["model_hash"] = hash_json(sealed)
    report = {
        "schema": "rsi_team_foot_velocity_chooser_train_report_v33",
        "activation_ceiling": "SIM_ONLY",
        "training_count": 176,
        "internal_validation_count": 16,
        "internal_validation_previously_consumed": True,
        "internal_validation": validation,
        "parent_unsafe_internal_validation": parent_unsafe,
        "selected_arms": [None if arm < 0 else arms[arm] for arm in chosen],
        "fresh_exam_authorized": passed,
        "model_hash": sealed["model_hash"],
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "model.json").write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RSI_TEAM_FOOT_VELOCITY_CHOOSER=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
