"""Fit one simulation-research model; it is not a safety gate or motor policy."""

import numpy as np
import torch

from rosclaw_soccer.rsi.receiver_risk_cv import risk_probability


def fit_single_receiver_risk(
    features: np.ndarray,
    base_failed: np.ndarray,
    candidate_failed: np.ndarray,
    complete_pair: np.ndarray,
    *,
    steps: int = 200,
) -> tuple[dict, dict]:
    """Use complete pair labels plus every directly observed candidate failure.

    A candidate failure is a negative safety example even when the baseline
    was censored; it is not claimed to be a paired regression. A positive
    requires a complete paired failure-to-survival outcome.
    """
    n = len(features)
    if (
        type(features) is not np.ndarray
        or features.dtype != np.float32
        or features.shape != (n, 104)
        or not np.isfinite(features).all()
        or any(
            type(x) is not np.ndarray or x.dtype != np.bool_ or x.shape != (n,)
            for x in (base_failed, candidate_failed, complete_pair)
        )
        or type(steps) is not int
        or not 1 <= steps <= 500
    ):
        raise ValueError("finite causal features and aligned physical outcomes required")
    admitted = complete_pair | candidate_failed
    target = complete_pair & base_failed & ~candidate_failed
    if admitted.sum() < 2 or len(np.unique(target[admitted])) != 2:
        raise ValueError("both observed benefit and failure-protection examples required")
    mean = features[admitted].mean(axis=0)
    scale = np.maximum(features[admitted].std(axis=0), np.float32(0.05))
    x = torch.from_numpy(((features[admitted] - mean) / scale).copy())
    y = torch.from_numpy(target[admitted].astype(np.float32))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        model = torch.nn.Linear(104, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.05)
        weight = torch.where(y == 0, 3.0, 1.0)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x).squeeze(-1)
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
                * weight
            ).mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            optimizer.step()
        result = dict(
            kind="linear",
            mean=mean.tolist(),
            scale=scale.tolist(),
            weight=model.weight.detach().numpy()[0].tolist(),
            bias=float(model.bias.detach().numpy()[0]),
        )
    scores = [risk_probability(row, result) for row in features]
    return result, dict(
        admitted=int(admitted.sum()),
        complete_pair=int(complete_pair.sum()),
        added_candidate_failures=int((candidate_failed & ~complete_pair).sum()),
        positives=int(target.sum()),
        optimizer_updates=steps,
        probabilities=scores,
        predictions=[int(p >= 0.8) for p in scores],
        physical_rollout=False,
        independent_fresh=False,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
    )
