"""Consumed football credit diagnostics; no fitting, selection or promotion.

Episode frames are correlated. Report both frame-weighted and equally weighted
context errors, and compare a context-excluded time-only baseline. This does not
establish that changing a critic will improve physical control.
"""

from typing import Any

import numpy as np


def diagnose_credit(
    *,
    context_ids: Any,
    targets: Any,
    predictions: Any,
    advantages: Any,
    high_quality: Any,
    safety_passed: Any,
) -> dict[str, Any]:
    ids = np.asarray(context_ids)
    values = [np.asarray(v) for v in (targets, predictions, advantages)]
    labels = [np.asarray(v) for v in (high_quality, safety_passed)]
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or len(ids) < 5
        or np.any(ids < 0)
        or any(a.shape != (len(ids), 270) or a.dtype != np.float64 for a in values)
        or not all(np.isfinite(a).all() for a in values)
        or any(a.shape != ids.shape or a.dtype != np.bool_ for a in labels)
        or set((ids % 5).tolist()) != set(range(5))
    ):
        raise ValueError("complete finite270-frame credit and all five context folds required")
    y, pred, adv = values
    hq, safe = labels
    baseline = np.empty_like(y)
    time_baseline = np.empty_like(y)
    folds = []
    for fold in range(5):
        test = ids % 5 == fold
        train = ~test
        baseline[test] = y[train].mean()
        time_baseline[test] = y[train].mean(axis=0)
        folds.append(
            dict(
                fold=fold,
                train_context_ids=np.unique(ids[train]).tolist(),
                held_out_context_ids=np.unique(ids[test]).tolist(),
                held_out_episodes=int(test.sum()),
            )
        )
    errors = dict(
        critic=(pred - y) ** 2,
        excluded_global_mean=(baseline - y) ** 2,
        excluded_time_only_mean=(time_baseline - y) ** 2,
    )
    contexts: list[dict[str, Any]] = [
        dict(
            context_id=int(context),
            episodes=int((ids == context).sum()),
            high_quality_episodes=int(hq[ids == context].sum()),
            safe_episodes=int(safe[ids == context].sum()),
            mse={name: float(error[ids == context].mean()) for name, error in errors.items()},
        )
        for context in np.unique(ids)
    ]
    groups = []
    for name, mask in (
        ("high_quality", hq),
        ("not_high_quality", ~hq),
        ("safe", safe),
        ("unsafe", ~safe),
    ):
        groups.append(
            dict(
                group=name,
                episodes=int(mask.sum()),
                independent_contexts=int(len(np.unique(ids[mask]))),
                mean_advantage=float(adv[mask].mean()) if mask.any() else None,
                positive_advantage_fraction=float((adv[mask] > 0).mean()) if mask.any() else None,
            )
        )
    return dict(
        episodes=len(ids),
        independent_consumed_contexts=len(contexts),
        frame_rows=int(y.size),
        folds=folds,
        contexts=contexts,
        outcome_groups=groups,
        frame_weighted_mse={name: float(error.mean()) for name, error in errors.items()},
        equally_weighted_context_mse={
            name: float(np.mean([c["mse"][name] for c in contexts])) for name in errors
        },
        temporal_windows=[
            dict(
                decision_frames_inclusive=[30 + start, 30 + start + 89],
                mse={
                    name: float(error[:, start : start + 90].mean())
                    for name, error in errors.items()
                },
            )
            for start in (0, 90, 180)
        ],
        context_excluded_time_baseline_is_diagnostic_only=True,
        outcomes_used_only_for_reporting=True,
        new_physical_executions=0,
        optimizer_steps=0,
        physical_gain_verified=False,
        private_fresh_accessed=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
