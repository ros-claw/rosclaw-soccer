"""Development-only out-of-fold probe selection for actual SIM replay.

The context ID selects the held-out evaluator model and is not a model input.
This is not a deployable policy or a hardware safety mechanism.
"""

import numpy as np

from rosclaw_soccer.rsi.receiver_ball_observation import receiver_ball_observation
from rosclaw_soccer.rsi.receiver_risk_cv import risk_probability


def select_oof_probe(
    receipt: dict,
    *,
    snapshot_hash: str,
    index: int,
    context: int,
    observation: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    variant: str,
) -> tuple[int, float]:
    if (
        type(receipt) is not dict
        or receipt.get("status") != "MULTIBLOCK_GROUPED_RISK_RESEARCH_NOT_PHYSICALLY_VALIDATED"
        or receipt.get("source_head") != "c0a2b4cd2a9c15bc6f2f5097bedd923808a1af6a"
        or receipt.get("exported_fold_models") is not True
        or receipt.get("predicted_unobserved") is not True
        or receipt.get("threshold") != 0.8
        or variant not in ("proprioception_only", "ball_augmented")
        or any(
            receipt.get(k) is not False
            for k in (
                "independent_fresh",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
            )
        )
    ):
        raise ValueError("exact unpromoted exported out-of-fold study required")
    if (
        type(observation) is not np.ndarray
        or observation.dtype != np.float32
        or observation.shape != (98,)
        or not np.isfinite(observation).all()
    ):
        raise ValueError("finite actual current receiver observation required")
    result = receipt[variant]
    models = {m["group"]: m for m in result["fold_models"]}
    if len(result["fold_models"]) != 12 or len(models) != 12 or context not in models:
        raise ValueError("exact held-out fold model required")
    keys = [(r["snapshot_hash"], r["index"], r["context"]) for r in receipt["identities"]]
    key = (snapshot_hash, index, context)
    if len(keys) != 72 or len(set(keys)) != 72 or key not in keys:
        raise ValueError("unique exact development snapshot identity required")
    x = (
        observation
        if variant == "proprioception_only"
        else np.concatenate((observation, receiver_ball_observation(q, v).astype(np.float32)))
    )
    probability = risk_probability(x, models[context])
    decision = int(probability >= receipt["threshold"])
    # Stored scores are checked only after computation; never used as action input.
    i = keys.index(key)
    if abs(probability - result["probabilities"][i]) > 1e-6 or decision != result["predictions"][i]:
        raise ValueError("computed current-state decision differs from original held-out audit")
    return decision, probability
