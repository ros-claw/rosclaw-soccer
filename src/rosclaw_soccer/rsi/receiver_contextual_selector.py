"""SIM-only tiny selector trained on past measured counterfactual outcomes.

Only a receiver's 98-D current velocity-policy observation is consumed at
inference. Classes select zero, positive or negative bounded leg corrections.
This is a developmental classifier, not a torque policy or a safety gate.
"""

import numpy as np
import torch


def train_context_selector(
    observations: np.ndarray, labels: np.ndarray, *, seed: int, steps: int
) -> tuple[dict[str, np.ndarray], dict]:
    if (
        type(observations) is not np.ndarray
        or observations.dtype != np.float32
        or observations.ndim != 2
        or observations.shape[1] != 98
        or not 2 <= len(observations) <= 10000
        or not np.isfinite(observations).all()
        or type(labels) is not np.ndarray
        or labels.dtype != np.int64
        or labels.shape != (len(observations),)
        or (labels < 0).any()
        or (labels > 2).any()
        or type(seed) is not int
        or not 0 <= seed < 2**31
        or type(steps) is not int
        or not 1 <= steps <= 500
    ):
        raise ValueError("finite own 98-D observations and bounded three-class labels required")
    mean = observations.mean(axis=0, dtype=np.float32)
    std = np.maximum(observations.std(axis=0, dtype=np.float32), np.float32(0.05))
    x = torch.from_numpy(((observations - mean) / std).astype(np.float32))
    y = torch.from_numpy(labels.copy())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(98, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite selector loss")
            loss.backward()
            optimizer.step()
        arrays = {
            "mean": mean.copy(),
            "std": std.copy(),
            "w0": model[0].weight.detach().numpy().copy(),
            "b0": model[0].bias.detach().numpy().copy(),
            "w1": model[2].weight.detach().numpy().copy(),
            "b1": model[2].bias.detach().numpy().copy(),
        }
    predictions = np.array([selector_class(row, arrays) for row in observations], np.int64)
    return arrays, dict(
        rows=len(labels),
        training_accuracy=float(np.mean(predictions == labels)),
        final_training_loss=float(loss.detach()),
        optimizer_steps=steps,
        neural_weight_updates=steps,
        seed=seed,
        independent_fresh=False,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
    )


def selector_class(observation: np.ndarray, weights: dict[str, np.ndarray]) -> int:
    shapes = {"mean": (98,), "std": (98,), "w0": (16, 98), "b0": (16,), "w1": (3, 16), "b1": (3,)}
    if (
        type(observation) is not np.ndarray
        or observation.shape != (98,)
        or observation.dtype != np.float32
        or not np.isfinite(observation).all()
        or type(weights) is not dict
        or set(weights) != set(shapes)
        or any(
            type(weights[k]) is not np.ndarray
            or weights[k].dtype != np.float32
            or weights[k].shape != shape
            or not np.isfinite(weights[k]).all()
            for k, shape in shapes.items()
        )
        or np.any(weights["std"] < np.float32(0.05))
    ):
        raise ValueError("finite own observation and exact selector parameter schema required")
    normalized = (observation - weights["mean"]) / weights["std"]
    logits = weights["w1"] @ np.tanh(weights["w0"] @ normalized + weights["b0"]) + weights["b1"]
    if not np.isfinite(logits).all():
        raise ValueError("finite selector logits required")
    return int(np.argmax(logits))


def group_holdout_validation(
    observations: np.ndarray, labels: np.ndarray, groups: np.ndarray, *, steps: int
) -> dict:
    """Refit preprocessing and network on other groups for each held-out group."""
    if (
        type(observations) is not np.ndarray
        or observations.ndim != 2
        or observations.shape[1] != 98
        or observations.dtype != np.float32
        or not np.isfinite(observations).all()
        or type(labels) is not np.ndarray
        or labels.dtype != np.int64
        or labels.shape != (len(observations),)
        or (labels < 0).any()
        or (labels > 2).any()
        or type(groups) is not np.ndarray
        or groups.dtype != np.int64
        or groups.ndim != 1
        or groups.shape != labels.shape
        or len(np.unique(groups)) < 2
        or len(np.unique(groups)) > 32
    ):
        raise ValueError("explicit multiple held-out context groups required")
    folds = []
    for group in np.unique(groups):
        train, test = groups != group, groups == group
        if train.sum() < 2 or test.sum() < 1:
            raise ValueError("nonempty context-separated training and evaluation required")
        weights, _ = train_context_selector(
            observations[train], labels[train], seed=1234, steps=steps
        )
        prediction = [selector_class(row, weights) for row in observations[test]]
        truth = labels[test].tolist()
        folds.append(
            dict(
                group=int(group),
                train_rows=int(train.sum()),
                test_rows=int(test.sum()),
                predictions=prediction,
                labels=truth,
                correct=sum(a == b for a, b in zip(prediction, truth, strict=True)),
            )
        )
    total = sum(f["test_rows"] for f in folds)
    return dict(
        status="GROUPED_SELECTOR_OFFLINE_VALIDATION_NOT_PHYSICAL_SUCCESS",
        folds=folds,
        accuracy=sum(f["correct"] for f in folds) / total,
        always_neutral_accuracy=float(np.mean(labels == 0)),
        held_out_contexts_excluded_from_fold_training=True,
        physical_rollout=False,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
    )
