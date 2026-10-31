"""Bounded sampled contact-response shaping, not physical pass evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sampled_pass_impulse_reward(
    *,
    ball_xy: NDArray[np.float64],
    target_xy: NDArray[np.float64],
    velocity_before_xy: NDArray[np.float64],
    velocity_after_xy: NDArray[np.float64],
    valid_contact: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Reward forward velocity change and penalize lateral change on contact.

    This deliberately conservative 50 Hz shaping is not a contact impulse: the
    measured velocity change can include ground and other within-frame forces.
    It is not a completed pass, safety signal, motor authorization or proof of
    causality. Callers must separately construct the sampled commitment/foot
    contact mask and preserve a physical evaluation contract.
    """
    n = len(valid_contact)
    vectors = (ball_xy, target_xy, velocity_before_xy, velocity_after_xy)
    if (
        valid_contact.ndim != 1
        or valid_contact.dtype != np.bool_
        or not 1 <= n <= 10000
        or any(x.shape != (n, 2) or not np.isfinite(x).all() for x in vectors)
    ):
        raise ValueError("finite aligned vectors and an explicit boolean contact mask required")
    ray = target_xy - ball_xy
    norm = np.linalg.norm(ray, axis=1)
    usable = valid_contact & (norm > 1e-6)
    direction = ray / np.maximum(norm[:, None], 1e-6)
    delta = velocity_after_xy - velocity_before_xy
    forward = np.sum(delta * direction, axis=1)
    lateral = delta[:, 0] * -direction[:, 1] + delta[:, 1] * direction[:, 0]
    reward = 0.04 * (np.clip(forward, -2.0, 2.0) - np.clip(np.abs(lateral), 0.0, 2.0))
    if not np.isfinite(reward).all():
        raise ValueError("nonfinite contact shaping")
    return np.asarray(np.where(usable, reward, 0.0), dtype=np.float64)
