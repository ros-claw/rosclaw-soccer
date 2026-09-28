"""Develop a small navigation chooser on 176 sealed paired physical contexts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_contextual_neural_train import ChooserNet, tally


def read_dataset(item: dict[str, Any], arms: list[str]) -> tuple[np.ndarray, np.ndarray]:
    path = Path(item["path"])
    if hash_bytes(path.read_bytes()) != item["hash"]:
        raise ValueError("physical dataset hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        names = data["arm_names"].tolist()
        indices = [names.index(name) for name in arms]
        feature = np.asarray(data["feature"], dtype=np.float64)
        labels = np.stack(
            [
                np.asarray(data["safe"])[:, indices],
                np.asarray(data["safe_contact"])[:, indices],
                np.asarray(data["useful_pass"])[:, indices],
            ],
            axis=2,
        ).astype(np.float64)
    selected = item["indices"]
    if selected == "all_128":
        selected = list(range(128))
    if max(selected) >= len(feature):
        raise ValueError("invalid split index")
    return feature[selected], labels[selected]


def train_candidate(
    *,
    train_feature: np.ndarray,
    train_label: np.ndarray,
    validation_feature: np.ndarray,
    candidate: dict[str, Any],
    seeds: list[int],
    width: int,
) -> tuple[dict[str, Any], np.ndarray]:
    count = candidate["feature_count"]
    mean = train_feature[:, :count].mean(axis=0)
    scale = np.maximum(train_feature[:, :count].std(axis=0), 0.03)
    x = torch.as_tensor((train_feature[:, :count] - mean) / scale, dtype=torch.float32)
    y = torch.as_tensor(train_label, dtype=torch.float32)
    xv = torch.as_tensor((validation_feature[:, :count] - mean) / scale, dtype=torch.float32)
    predictions = []
    network_weights = []
    for seed in seeds:
        torch.manual_seed(seed)
        net = ChooserNet(count, width, train_label.shape[1])
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.004, weight_decay=0.04)
        for _ in range(candidate["epochs"]):
            logits = net(x).reshape(*train_label.shape)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            predictions.append(torch.sigmoid(net(xv).reshape(-1, train_label.shape[1], 3)).numpy())
        network_weights.append(
            {
                "seed": seed,
                "layers": [
                    {"weight": layer.weight.detach().tolist(), "bias": layer.bias.detach().tolist()}
                    for layer in net.layers
                    if isinstance(layer, nn.Linear)
                ],
            }
        )
    stacked = np.stack(predictions)
    score = stacked.mean(axis=0)
    score[:, :, 0] = np.quantile(stacked[:, :, :, 0], 0.2, axis=0)
    safe = score[:, :, 0] >= candidate["safety_threshold"]
    value = np.where(safe, score[:, :, 2] + 0.25 * score[:, :, 1], -np.inf)
    chosen = np.where(safe.any(axis=1), np.argmax(value, axis=1), -1)
    model = {
        "feature_count": count,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "networks": network_weights,
        "safety_threshold": candidate["safety_threshold"],
        "safety_quantile": 0.2,
    }
    return model, chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("retraining output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_contextual_neural_chooser_protocol_v32"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or len(protocol.get("training_datasets", [])) != 2
        or len(protocol.get("candidates", [])) != 12
        or protocol.get("internal_gate") != {"max_unsafe": 0, "minimum_useful": 4}
    ):
        raise ValueError("invalid committed retraining protocol")
    arms = protocol["eligible_arms"]
    parts = [read_dataset(item, arms) for item in protocol["training_datasets"]]
    feature = np.concatenate([part[0] for part in parts])
    labels = np.concatenate([part[1] for part in parts])
    validation_feature, validation_label = read_dataset(
        protocol["internal_validation_dataset"], arms
    )
    if feature.shape != (176, 8) or labels.shape != (176, 6, 3):
        raise ValueError("incorrect physical training split")
    if validation_feature.shape != (16, 8):
        raise ValueError("incorrect internal validation split")
    torch.set_num_threads(1)
    candidates = []
    models = []
    for index, candidate in enumerate(protocol["candidates"]):
        model, chosen = train_candidate(
            train_feature=feature,
            train_label=labels,
            validation_feature=validation_feature,
            candidate=candidate,
            seeds=protocol["ensemble_seeds"],
            width=protocol["mlp_width"],
        )
        metrics = tally(chosen, validation_label)
        candidates.append(
            {
                "index": index,
                "specification": candidate,
                "internal_validation": metrics,
                "selected_arms": [None if arm < 0 else arms[arm] for arm in chosen],
            }
        )
        models.append(model)
    ranking = sorted(
        candidates,
        key=lambda row: (
            row["internal_validation"]["unsafe"],
            -row["internal_validation"]["useful"],
            -row["internal_validation"]["safe_contact"],
            row["internal_validation"]["abstained"],
            row["index"],
        ),
    )
    best = ranking[0]
    selected_model = {
        "schema": "rsi_team_contextual_neural_chooser_model_v32",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "training_dataset_hashes": [item["hash"] for item in protocol["training_datasets"]],
        "arm_names": arms,
        "selected_candidate_index": best["index"],
        **models[best["index"]],
    }
    selected_model["model_hash"] = hash_json(selected_model)
    gate = bool(
        best["internal_validation"]["unsafe"] == 0 and best["internal_validation"]["useful"] >= 4
    )
    report = {
        "schema": "rsi_team_contextual_neural_retrain_report_v32",
        "activation_ceiling": "SIM_ONLY",
        "training_context_count": 176,
        "validation_context_count": 16,
        "validation_previously_consumed": True,
        "fixed_arm_internal_validation": {
            arm: tally(np.full(16, index, dtype=np.int64), validation_label)
            for index, arm in enumerate(arms)
        },
        "candidates": candidates,
        "selected_candidate_index": best["index"],
        "selected_model_hash": selected_model["model_hash"],
        "fresh_physical_exam_authorized": gate,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "model.json").write_text(
        json.dumps(selected_model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RSI_TEAM_NEURAL_RETRAIN="
        + json.dumps(
            {
                "report_hash": report["report_hash"],
                "model_hash": selected_model["model_hash"],
                "selected_candidate": best,
                "fresh_physical_exam_authorized": gate,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
