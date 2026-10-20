"""Fixed-terminal discounted targets and diagnostics, never actor observations."""

from typing import Any

import numpy as np


def task_returns(rewards: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    if (
        rewards.ndim != 2
        or len(rewards) < 1
        or rewards.shape[1] != 300
        or rewards.dtype != np.float32
        or not np.isfinite(rewards).all()
    ):
        raise ValueError("finite full H300 float32 causal rewards required")
    result = np.zeros_like(rewards, dtype=np.float64)
    carry = np.zeros(len(rewards), dtype=np.float64)
    for t in range(299, -1, -1):
        carry = rewards[:, t].astype(np.float64) + 0.99 * carry
        result[:, t] = carry
    if not np.isfinite(result).all():
        raise ValueError("nonfinite complete discounted return")
    return result.astype(np.float32)


def value_metrics(predicted: np.ndarray[Any, Any], target: np.ndarray[Any, Any]) -> dict[str, Any]:
    if (
        predicted.shape != target.shape
        or target.ndim != 2
        or target.shape[1] != 300
        or len(target) < 1
        or not np.isfinite(predicted).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("finite full per-context value targets required")
    p, y = predicted[:, 20:].astype(np.float64), target[:, 20:].astype(np.float64)
    variance = float(y.var())
    return dict(
        contexts=len(y),
        suffix_frames_per_context=280,
        mse=float(np.mean((p - y) ** 2)),
        explained_variance=None if variance == 0 else 1 - float((y - p).var()) / variance,
        independent_sample_count_is_contexts_not_frames=True,
    )


def calibration_split(course_indices: list[int]) -> tuple[list[int], list[int]]:
    if (
        len(course_indices) != 32
        or len(set(course_indices)) != 32
        or course_indices != sorted(course_indices)
        or any(type(i) is not int or i not in range(40) for i in course_indices)
    ):
        raise ValueError("sorted32 unique original TRAIN course indices required")
    return [i for i in range(32) if i % 4 != 3], [i for i in range(32) if i % 4 == 3]
