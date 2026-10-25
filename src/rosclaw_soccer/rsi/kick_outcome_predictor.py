"""Learned SIM outcome forecasts, not an RL Actor or motion authorization.

This supervised model has no actuator/trajectory interface. Its input is the
causal entry observation plus the reference/task clocks, never future results.
"""

from typing import Any

import numpy as np


def grouped_partition(groups: np.ndarray, strata: np.ndarray) -> dict[str, Any]:
    if (
        groups.ndim != 1
        or strata.shape != groups.shape
        or groups.dtype.kind != "U"
        or (strata.dtype.kind not in "iu" or not np.isin(strata, [0, 1, 2, 3]).all())
    ):
        raise ValueError("typed initial-state group keys and four strata required")
    rng = np.random.default_rng(2026102527)
    train: list[str] = []
    validation: list[str] = []
    for bucket in range(4):
        ids = sorted(set(groups[strata == bucket].tolist()))
        if len(ids) != 14 or any(len(set(strata[groups == g].tolist())) != 1 for g in ids):
            raise ValueError("fourteen distinct original initial states per class required")
        order = rng.permutation(len(ids))
        validation.extend(ids[i] for i in order[:4])
        train.extend(ids[i] for i in order[4:])
    if len(set(train) | set(validation)) != 56 or set(train) & set(validation):
        raise ValueError("complete disjoint56 initial-group partition required")
    return dict(
        train_groups=sorted(train),
        validation_groups=sorted(validation),
        split_seed=2026102527,
        validation_is_sealed_fresh=False,
        initial_groups_per_class=dict(train=10, validation=4),
    )


def group_weights(groups: np.ndarray) -> np.ndarray:
    if groups.ndim != 1 or not len(groups) or groups.dtype.kind != "U":
        raise ValueError("nonempty group-key vector required")
    ids, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / (len(ids) * counts[inverse])


def make_predictor(mean: np.ndarray, scale: np.ndarray) -> Any:
    """Lazy optional Torch; original control/runtime modules do not import it."""
    import torch
    from torch import nn

    if (
        any(
            x.shape != (549,) or x.dtype != np.float32 or not np.isfinite(x).all()
            for x in (mean, scale)
        )
        or (scale < 0.1).any()
    ):
        raise ValueError("TRAIN-only finite549 normalization with .1 floor required")

    class KickOutcomePredictor(nn.Module):
        simulation_only = True
        motor_authority = False

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("mean", torch.from_numpy(mean.copy()))
            self.register_buffer("scale", torch.from_numpy(scale.copy()))
            self.net = nn.Sequential(
                nn.Linear(549, 128, dtype=torch.float32, device="cpu"),
                nn.ELU(),
                nn.Linear(128, 64, dtype=torch.float32, device="cpu"),
                nn.ELU(),
                nn.Linear(64, 3, dtype=torch.float32, device="cpu"),
            )

        def forward(self, x: Any) -> Any:
            if (
                x.ndim != 2
                or x.shape[1] != 549
                or not 1 <= len(x) <= 8192
                or (
                    x.dtype != torch.float32
                    or x.device != self.mean.device
                    or not torch.isfinite(x).all()
                )
            ):
                raise ValueError("finite causal549 batch on the predictor device required")
            return self.net((x - self.mean) / self.scale)

        def probabilities(self, x: Any) -> Any:
            values = torch.sigmoid(self(x))
            return torch.stack((values[:, 0], values[:, 1], values.prod(dim=1)), dim=1)

    return KickOutcomePredictor()


def grouped_metrics(
    prediction: np.ndarray, target: np.ndarray, groups: np.ndarray
) -> dict[str, Any]:
    if (
        prediction.shape != target.shape
        or prediction.shape != (len(groups), 3)
        or (
            not np.isfinite(prediction).all()
            or not np.isfinite(target).all()
            or (prediction < 0).any()
            or (prediction > 1).any()
            or not np.isin(target, [0, 1]).all()
        )
    ):
        raise ValueError(
            "aligned finite probabilities, binary physical outcomes and groups required"
        )
    w = group_weights(groups)
    squared = (prediction - target) ** 2
    high_safe = prediction[:, 0] >= 0.9
    coverage = float(w @ high_safe)
    bad = high_safe & (target[:, 0] == 0)
    ece = 0.0
    for i in range(10):
        mask = (prediction[:, 2] >= i / 10) & (prediction[:, 2] < (i + 1) / 10)
        if i == 9:
            mask |= prediction[:, 2] == 1
        mass = float(w @ mask)
        if mass:
            ece += abs(float(w[mask] @ (prediction[mask, 2] - target[mask, 2])))
    return dict(
        independent_initial_groups=len(set(groups.tolist())),
        rows=len(groups),
        group_weighted_brier=(w[:, None] * squared).sum(axis=0).tolist(),
        safe_high_confidence_coverage=coverage,
        safe_high_confidence_error=float(w @ bad) / coverage if coverage else None,
        safe_high_confidence_raw_count=int(high_safe.sum()),
        dangerous_false_safe_raw_count=int(bad.sum()),
        task_probability_ece10=ece,
        quality_probability_is_bounded_by_safe_and_foot=bool(
            (prediction[:, 2] <= prediction[:, :2].min(axis=1) + 1e-7).all()
        ),
        prediction_is_motion_authorization=False,
    )
