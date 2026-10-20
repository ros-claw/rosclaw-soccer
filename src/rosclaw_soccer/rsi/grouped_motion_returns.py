"""Learner-only leave-one-out reward-to-go, grouped by shared initial context.

This baseline does not use a learned critic or enter an actor observation.
It is not a claim that a multi-step clipped surrogate is an unbiased full-path
policy-gradient estimator. Repeated noise samples are not new task contexts.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.motion_value_calibration import task_returns


def leave_one_out_returns(
    rewards: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    if (
        rewards.ndim != 3
        or not 1 <= rewards.shape[0] <= 128
        or not 2 <= rewards.shape[1] <= 8
        or rewards.shape[2] != 300
        or rewards.dtype != np.float32
        or not np.isfinite(rewards).all()
    ):
        raise ValueError("finite context, independent-noise-group, H300 float32 rewards required")
    n, k, _ = rewards.shape
    returns = task_returns(rewards.reshape(n * k, 300)).reshape(n, k, 300)
    if not np.isfinite(returns).all():
        raise ValueError("returns outside representable training domain")
    full = returns.astype(np.float64)
    baseline = (full.sum(axis=1, keepdims=True) - full) / (k - 1)
    advantage = (full - baseline).astype(np.float32)
    advantage[:, :, :20] = 0.0
    if not np.isfinite(advantage).all():
        raise ValueError("nonfinite grouped reward-to-go advantage")
    return advantage, dict(
        independent_contexts=n,
        noise_samples_per_context=k,
        episode_executions=n * k,
        prefix_frames_excluded=20,
        critic_predictions_used=False,
        baseline_excludes_own_trajectory=True,
        zero_variation_contexts=int(np.sum(np.max(np.abs(advantage[:, :, 20:]), axis=(1, 2)) == 0)),
        global_suffix_advantage_std=float(advantage[:, :, 20:].astype(np.float64).std()),
        max_absolute_group_advantage_sum=float(
            np.max(np.abs(advantage.sum(axis=1, dtype=np.float64)))
        ),
        optimizer_steps=0,
        new_native_steps=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
