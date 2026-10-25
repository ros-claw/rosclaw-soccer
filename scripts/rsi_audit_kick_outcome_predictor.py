"""Read-only checkpoint replay and grouped forecast failure diagnosis, no training."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--training-hash", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    training_path = a.training / "complete.json"
    bank_path = a.bank / "complete.json"
    if (
        a.output.exists()
        or training_path.is_symlink()
        or (digest(training_path) != a.training_hash)
    ):
        raise ValueError("new audit output and exact completed training required")
    training = json.loads(training_path.read_text())
    if training["status"] != "THREE_SUPERVISED_OUTCOME_PREDICTORS_TRAINED_NOT_MOTOR_POLICIES":
        raise ValueError("complete three-seed predictor study required")
    if digest(bank_path) != training["bank_hash"]:
        raise ValueError("original causal data bank required")
    bank = json.loads(bank_path.read_text())
    pins = dict(training["input_file_hashes"])
    pins[str(training_path)] = a.training_hash
    for result in training["results"]:
        for key in ("checkpoint", "predictions"):
            artifact = result[key]
            path = a.training / artifact["path"]
            if path.parent.resolve() != a.training.resolve() or path.is_symlink():
                raise ValueError("confined ordinary model artifacts required")
            pins[str(path)] = artifact["hash"]
    source_pins(pins)
    check_pins(pins)
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.kick_outcome_predictor import (
        group_weights,
        grouped_metrics,
        grouped_partition,
        make_predictor,
    )

    torch.set_num_threads(1)
    with np.load(a.bank / bank["artifact"]["path"]) as saved:
        x, y, groups, strata = (saved[k] for k in ("features", "labels", "groups", "strata"))
    split = grouped_partition(groups, strata)
    if training["split"] != split:
        raise ValueError("training split changed")
    train = np.isin(groups, split["train_groups"])
    val = np.isin(groups, split["validation_groups"])
    w = group_weights(groups[train])
    mean64 = (w[:, None] * x[train]).sum(axis=0)
    mean = mean64.astype(np.float32)
    scale = np.maximum(np.sqrt((w[:, None] * (x[train] - mean64) ** 2).sum(axis=0)), 0.1)
    scale = scale.astype(np.float32)
    reports = []
    for result in training["results"]:
        checkpoint = torch.load(
            a.training / result["checkpoint"]["path"], map_location="cpu", weights_only=True
        )
        if any(
            checkpoint[k] != expected
            for k, expected in (
                ("actual_optimizer_updates", 512),
                ("actor_updates", 0),
                ("motor_authority", False),
                ("context_hash", training["context_hash"]),
                ("bank_hash", training["bank_hash"]),
                ("seed", result["seed"]),
                ("split", split),
            )
        ):
            raise ValueError("checkpoint identity/authority mismatch")
        model = make_predictor(mean, scale)
        state = checkpoint["state_dict"]
        if not np.array_equal(state["mean"].numpy(), mean) or not np.array_equal(
            state["scale"].numpy(), scale
        ):
            raise ValueError("normalization is not exact TRAIN-only group normalization")
        model.load_state_dict(state, strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if len(optimizer.state) != 6 or any(
            int(s["step"]) != 512
            or not torch.isfinite(s["exp_avg_sq"]).all()
            or not bool(s["exp_avg_sq"].sum() > 0)
            for s in optimizer.state.values()
        ):
            raise ValueError("six finite nonempty Adam moment states at512 required")
        model.eval()
        with torch.no_grad():
            prediction = model.probabilities(torch.from_numpy(x)).numpy()
        with np.load(a.training / result["predictions"]["path"]) as saved:
            for actual, expected in (
                (saved["prediction"], prediction[val]),
                (saved["targets"], y[val]),
                (saved["groups"], groups[val]),
                (saved["strata"], strata[val]),
            ):
                if not np.array_equal(actual, expected):
                    raise ValueError("safe-loaded checkpoint failed exact validation replay")
        metrics = grouped_metrics(prediction[val], y[val], groups[val])
        if metrics != result["validation"]:
            raise ValueError("published metrics differ from independent recomputation")
        class_rates = np.stack(
            [
                np.stack(
                    [
                        y[(groups == g)].mean(axis=0)
                        for g in split["train_groups"]
                        if strata[np.flatnonzero(groups == g)[0]] == bucket
                    ]
                ).mean(axis=0)
                for bucket in range(4)
            ]
        )
        baseline = grouped_metrics(class_rates[strata[val]], y[val], groups[val])
        if baseline != training["baseline"]["class_frequency"]:
            raise ValueError("TRAIN-only class baseline failed replay")
        reports.append(
            dict(
                seed=result["seed"],
                exact_checkpoint_validation_replay=True,
                optimizer_six_tensors_verified_at_step512=True,
                train=grouped_metrics(prediction[train], y[train], groups[train]),
                validation=metrics,
                per_validation_class=[
                    grouped_metrics(
                        prediction[val & (strata == k)],
                        y[val & (strata == k)],
                        groups[val & (strata == k)],
                    )
                    for k in range(4)
                ],
                baseline=baseline,
                rejected_for_physical_ranking=not result["eligible_for_physical_ranking_review"],
            )
        )
    check_pins(pins)
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "complete.json",
        dict(
            status="THREE_OUTCOME_CHECKPOINTS_EXACTLY_REPLAYED_AND_GROUP_METRICS_VERIFIED",
            input_file_hashes=pins,
            source_head=a.expected_head,
            training_hash=a.training_hash,
            results=reports,
            optimizer_updates_added=0,
            actor_updates=0,
            physical_steps_added=0,
            physical_football_gain_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
