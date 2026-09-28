"""Train a tiny causal, SIM_ONLY team navigation chooser on paired physical labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


class ChooserNet(nn.Module):
    def __init__(self, input_count: int, width: int, arm_count: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_count, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, arm_count * 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def choose(scores: np.ndarray) -> np.ndarray:
    """Pessimistic safety guard; -1 means leave the parent unchanged."""
    safe = scores[:, :, 0]
    value = scores[:, :, 2] + 0.25 * scores[:, :, 1]
    eligible = safe >= 0.70
    rank = np.where(eligible, value, -np.inf)
    selection = np.argmax(rank, axis=1)
    return np.where(eligible.any(axis=1), selection, -1)


def tally(selection: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    selected = selection >= 0
    row = np.arange(selection.size)[selected]
    col = selection[selected]
    picked = labels[row, col]
    return {
        "scenes": int(selection.size),
        "abstained": int((~selected).sum()),
        "safe": int(picked[:, 0].sum()),
        "safe_contact": int(picked[:, 1].sum()),
        "useful": int(picked[:, 2].sum()),
        "unsafe": int((picked[:, 0] == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("chooser output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_contextual_neural_chooser_protocol_v30"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("training_indices") != list(range(48))
        or protocol.get("internal_validation_indices") != list(range(48, 64))
        or protocol.get("selection_gate") != {"minimum_safe": 14, "minimum_useful": 4}
    ):
        raise ValueError("invalid committed chooser protocol")
    dataset_path = Path(protocol["dataset_path"])
    if hash_bytes(dataset_path.read_bytes()) != protocol["dataset_hash"]:
        raise ValueError("training dataset commitment mismatch")
    with np.load(dataset_path, allow_pickle=False) as dataset:
        names = dataset["arm_names"].tolist()
        columns = [names.index(name) for name in protocol["eligible_arms"]]
        features = np.asarray(dataset["feature"], dtype=np.float64)
        labels = np.stack(
            [
                np.asarray(dataset["safe"])[:, columns],
                np.asarray(dataset["safe_contact"])[:, columns],
                np.asarray(dataset["useful_pass"])[:, columns],
            ],
            axis=2,
        ).astype(np.float64)
    if features.shape != (64, 8) or labels.shape != (64, 6, 3):
        raise ValueError("unexpected causal feature or arm-label shape")
    mean = features[:48].mean(axis=0)
    scale = np.maximum(features[:48].std(axis=0), 0.03)
    normalized = (features - mean) / scale
    x_train = torch.as_tensor(normalized[:48], dtype=torch.float32)
    y_train = torch.as_tensor(labels[:48], dtype=torch.float32)
    x_all = torch.as_tensor(normalized, dtype=torch.float32)
    nets: list[dict[str, Any]] = []
    predictions = []
    torch.set_num_threads(1)
    for seed in protocol["ensemble_seeds"]:
        torch.manual_seed(seed)
        net = ChooserNet(8, protocol["mlp_width"], 6)
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=protocol["learning_rate"],
            weight_decay=protocol["weight_decay"],
        )
        for _ in range(protocol["epochs"]):
            logits = net(x_train).reshape(48, 6, 3)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            predictions.append(torch.sigmoid(net(x_all).reshape(64, 6, 3)).numpy())
        nets.append(
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
    # A low-tail estimate penalizes ensemble disagreements on safety.
    scores = stacked.mean(axis=0)
    scores[:, :, 0] = np.quantile(stacked[:, :, :, 0], 0.2, axis=0)
    selected = choose(scores)
    train = tally(selected[:48], labels[:48])
    validation = tally(selected[48:], labels[48:])
    model = {
        "schema": "rsi_team_contextual_neural_chooser_model_v30",
        "activation_ceiling": "SIM_ONLY",
        "dataset_hash": protocol["dataset_hash"],
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "arm_names": protocol["eligible_arms"],
        "input_features": [
            "ball_minus_selected_foot_x",
            "ball_minus_selected_foot_y",
            "selected_minus_other_foot_z",
            "ball_vx",
            "body_x",
            "body_y",
            "body_velocity_x",
            "body_velocity_y",
        ],
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "networks": nets,
        "safety_threshold": 0.70,
        "safety_quantile": 0.2,
        "promotion_authorized": False,
    }
    model["model_hash"] = hash_json(model)
    passed = bool(
        validation["safe"] >= protocol["selection_gate"]["minimum_safe"]
        and validation["useful"] >= protocol["selection_gate"]["minimum_useful"]
        and validation["unsafe"] == 0
    )
    report = {
        "schema": "rsi_team_contextual_neural_chooser_train_v30",
        "activation_ceiling": "SIM_ONLY",
        "model_hash": model["model_hash"],
        "dataset_hash": protocol["dataset_hash"],
        "train": train,
        "internal_validation": validation,
        "internal_validation_selected_arms": [
            None if item < 0 else model["arm_names"][item] for item in selected[48:]
        ],
        "internal_validation_only": True,
        "fresh_physical_exam_authorized": passed,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RSI_TEAM_NEURAL_CHOOSER=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
