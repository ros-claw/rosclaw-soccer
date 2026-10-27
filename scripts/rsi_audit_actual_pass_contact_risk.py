"""Replay held-out contact classifier checkpoints; no simulator or control."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.rsi.actual_pass_contact_risk import (
    body_relative_pass_features,
    contact_risk_data,
    context_shard,
)
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "dataset", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("source-hash", "dataset-hash", "expected-head"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--scene-hash")
    args = parser.parse_args()
    clean(args.expected_head)
    source, receipt = args.source / "complete.json", args.dataset / "complete.json"
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or source.is_symlink()
        or receipt.is_symlink()
        or digest(source) != args.source_hash
        or digest(receipt) != args.dataset_hash
    ):
        raise ValueError("exact source, dataset and unused external audit output required")
    training, dataset = (json.loads(p.read_text()) for p in (source, receipt))
    path = args.dataset / "dataset.npz"
    if (
        training["status"] != "ACTUAL_PASS_CONTACT_RISK_SUPERVISED_CONTEXT_HOLDOUT_NOT_CONTROL"
        or training["dataset_hash"] != args.dataset_hash
        or dataset["status"] != "ACTUAL_PASS_CAUSAL_TRANSITIONS_EXPORTED_NOT_TRAINED"
        or path.is_symlink()
        or digest(path) != dataset["dataset_hash"]
    ):
        raise ValueError("complete unpromoted model and bound dataset required")
    with np.load(path, allow_pickle=False) as f:
        recorded = {k: f[k].copy() for k in f.files}
    x, y, groups = contact_risk_data(recorded, dataset["courses"])
    if training.get("feature_frame", "world") == "body":
        if (
            args.scene is None
            or args.scene_hash is None
            or args.scene.is_symlink()
            or digest(args.scene) != args.scene_hash
            or training.get("scene_hash") != args.scene_hash
        ):
            raise ValueError("exact body-relative scene required")
        import mujoco

        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

        model = mujoco.MjModel.from_xml_path(str(args.scene))
        layout = NativeVelocityLayout(
            model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
        )
        layout_contract = dict(
            robot_q=layout.rq, robot_v=layout.rv, ball_q=layout.bq, ball_v=layout.bv
        )
        if training.get("layout") != layout_contract:
            raise ValueError("exact simulator-derived layout required")
        x = body_relative_pass_features(recorded, **layout_contract)
    elif (
        training.get("feature_frame", "world") != "world"
        or args.scene is not None
        or args.scene_hash is not None
    ):
        raise ValueError("explicit valid feature-frame and scene pairing required")
    if training.get("input_dimensions", x.shape[1]) != x.shape[1]:
        raise ValueError("trained feature dimension must match reconstructed input")
    contexts = {int(v) for v in set(groups)}
    all_contexts = tuple(sorted(contexts))
    shard_count = training.get("shard_count", 1)
    shard_index = training.get("shard_index", 0)
    selected = context_shard(all_contexts, index=shard_index, count=shard_count)
    if training.get("selected_contexts", list(all_contexts)) != list(selected) or len(
        training["rows"]
    ) != 3 * len(selected):
        raise ValueError("all held-out context/seed checkpoints required")
    torch.set_num_threads(1)
    rows = []
    seen = set()
    for original in training["rows"]:
        holdout, seed = original["heldout_context"], original["seed"]
        if holdout not in selected or seed not in (17, 43, 71) or (holdout, seed) in seen:
            raise ValueError("all unique held-out context/seed checkpoints required")
        seen.add((holdout, seed))
        checkpoint = args.source / original["checkpoint"]
        if (
            checkpoint.is_symlink()
            or checkpoint.parent != args.source
            or digest(checkpoint) != original["checkpoint_hash"]
        ):
            raise ValueError("exact confined numeric checkpoint required")
        with np.load(checkpoint, allow_pickle=False) as f:
            weights = {k: f[k].copy() for k in f.files}
        expected_shapes = {
            "center": (x.shape[1],),
            "scale": (x.shape[1],),
            "weight1": (64, x.shape[1]),
            "bias1": (64,),
            "weight2": (1, 64),
            "bias2": (1,),
        }
        if set(weights) != set(expected_shapes) or any(
            weights[k].shape != s or not np.isfinite(weights[k]).all()
            for k, s in expected_shapes.items()
        ):
            raise ValueError("finite numeric model checkpoint required")
        train, test = groups != holdout, groups == holdout
        center, scale = x[train].mean(axis=0), np.maximum(x[train].std(axis=0), 0.01)
        if not np.array_equal(weights["center"], center) or not np.array_equal(
            weights["scale"], scale
        ):
            raise ValueError("normalization must use training contexts only")
        z = np.clip((x[test] - center) / scale, -10, 10).astype(np.float32)
        network = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 1)
        )
        with torch.no_grad():
            network[0].weight.copy_(torch.from_numpy(weights["weight1"]))
            network[0].bias.copy_(torch.from_numpy(weights["bias1"]))
            network[2].weight.copy_(torch.from_numpy(weights["weight2"]))
            network[2].bias.copy_(torch.from_numpy(weights["bias2"]))
            predicted = torch.sigmoid(network(torch.from_numpy(z))).numpy().reshape(-1) >= 0.5
        truth = y[test].astype(bool)
        counts = dict(
            tp=int((predicted & truth).sum()),
            fn=int((~predicted & truth).sum()),
            fp=int((predicted & ~truth).sum()),
            tn=int((~predicted & ~truth).sum()),
        )
        if any(original[k] != v for k, v in counts.items()):
            raise ValueError("replayed holdout confusion counts differ")
        rows.append(
            dict(
                heldout_context=holdout,
                seed=seed,
                checkpoint_hash=original["checkpoint_hash"],
                **counts,
            )
        )
    if (
        digest(source) != args.source_hash
        or digest(receipt) != args.dataset_hash
        or digest(path) != dataset["dataset_hash"]
        or args.scene is not None
        and digest(args.scene) != args.scene_hash
    ):
        raise ValueError("source changed during readback")
    args.output.mkdir()
    write_once(
        args.output / "complete.json",
        dict(
            status="ACTUAL_PASS_CONTACT_RISK_HOLDOUT_PREDICTIONS_REPLAYED_NOT_CERTIFIED",
            source_hash=args.source_hash,
            dataset_hash=args.dataset_hash,
            source_head=args.expected_head,
            rows=rows,
            checkpoint_predictions_replayed=True,
            optimizer_retrained=False,
            same_feature_implementation=True,
            feature_frame=training.get("feature_frame", "world"),
            scene_hash=training.get("scene_hash"),
            safety_certificate=False,
            controller_interface=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
