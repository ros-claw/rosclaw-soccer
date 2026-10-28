"""Group-separated research selection from paired SIM failure outcomes.

This module does not control a player, authorize a policy, or certify a catch.
Labels are retrospective paired outcomes; inputs must be pre-intervention state.
"""

import numpy as np
import torch


def grouped_risk_selection(
    features: np.ndarray,
    groups: np.ndarray,
    base_failed: np.ndarray,
    candidate_failed: np.ndarray,
    observed: np.ndarray,
    *,
    steps: int = 200,
    threshold: float = 0.8,
    predict_unobserved: bool = False,
) -> dict:
    """Refit normalization and binary model on each training fold only.

    Positive means a measured failure-to-survival change, not football success.
    Ties and regressions are negative; unobserved cases are never fit or scored.
    Optionally report their predictions separately, without asserting outcomes.
    The higher weight on negative examples penalizes unsafe intervention.
    """
    n = len(features)
    if (
        type(features) is not np.ndarray
        or features.dtype != np.float32
        or features.ndim != 2
        or features.shape[1] not in (98, 104)
        or not np.isfinite(features).all()
        or type(groups) is not np.ndarray
        or groups.dtype != np.int64
        or groups.shape != (n,)
        or len(np.unique(groups)) < 2
        or any(
            type(a) is not np.ndarray or a.dtype != np.bool_ or a.shape != (n,)
            for a in (base_failed, candidate_failed, observed)
        )
        or type(steps) is not int
        or not 1 <= steps <= 500
        or not 0.5 < threshold < 1.0
        or type(predict_unobserved) is not bool
    ):
        raise ValueError("aligned causal features, groups and observed paired outcomes required")
    target = base_failed & ~candidate_failed
    predictions = np.full(n, -1, dtype=np.int64)
    probabilities = np.full(n, np.nan, dtype=np.float64)
    folds = []
    for group in np.unique(groups):
        train = (groups != group) & observed
        test = (groups == group) & (observed | predict_unobserved)
        if not np.any(test):
            folds.append(dict(group=int(group), train=int(train.sum()), test=0, updates=0))
            continue
        # Without both classes, abstain instead of inventing an intervention.
        if train.sum() < 2 or len(np.unique(target[train])) < 2:
            predictions[test] = 0
            probabilities[test] = 0
            folds.append(
                dict(group=int(group), train=int(train.sum()), test=int(test.sum()), updates=0)
            )
            continue
        mean = features[train].mean(axis=0)
        scale = np.maximum(features[train].std(axis=0), np.float32(0.05))
        x = torch.from_numpy(((features[train] - mean) / scale).copy())
        y = torch.from_numpy(target[train].astype(np.float32))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            model = torch.nn.Linear(features.shape[1], 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.05)
            weight = torch.where(y == 0, 3.0, 1.0)
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                logits = model(x).squeeze(-1)
                loss = (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, y, reduction="none"
                    )
                    * weight
                ).mean()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                p = torch.sigmoid(
                    model(torch.from_numpy(((features[test] - mean) / scale).copy())).squeeze(-1)
                ).numpy()
        probabilities[test] = p
        predictions[test] = (p >= threshold).astype(np.int64)
        folds.append(
            dict(group=int(group), train=int(train.sum()), test=int(test.sum()), updates=steps)
        )
    chosen_failed = np.where(predictions == 1, candidate_failed, base_failed)
    return dict(
        folds=folds,
        predictions=predictions.tolist(),
        probabilities=[None if not np.isfinite(x) else float(x) for x in probabilities],
        labeled=int(observed.sum()),
        censored=int((~observed).sum()),
        baseline_failures=int(base_failed[observed].sum()),
        always_candidate_failures=int(candidate_failed[observed].sum()),
        selected_counterfactual_failures=int(chosen_failed[observed].sum()),
        selected_interventions=int((predictions[observed] == 1).sum()),
        selected_regressions=int(
            ((predictions == 1) & ~base_failed & candidate_failed & observed).sum()
        ),
        selected_rescues=int(((predictions == 1) & target & observed).sum()),
        unobserved_predictions=int(((predictions >= 0) & ~observed).sum()),
        unobserved_interventions=int(((predictions == 1) & ~observed).sum()),
        unobserved_interventions_with_candidate_failure=int(
            ((predictions == 1) & ~observed & candidate_failed).sum()
        ),
        optimizer_updates=sum(f["updates"] for f in folds),
        independent_fresh=False,
        physical_rollout=False,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
    )
