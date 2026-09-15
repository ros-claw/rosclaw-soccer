"""Frozen reference distributions on the first on-policy training batch."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.role_behavior_anchor import RoleBehaviorAnchor


def validate_retention_coefficient(value: float) -> None:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not (value == 0 or 0.01 <= value <= 1000)
    ):
        raise ValueError("bounded explicit retention coefficient required")


def first_batch_anchor(
    reference: NearBallResidualPolicy, traces: list[dict[str, Any]], *, coefficient: float
) -> RoleBehaviorAnchor | None:
    validate_retention_coefficient(coefficient)
    if coefficient == 0:
        return None
    if not traces:
        raise ValueError("retention requires its declared first training batch")
    observations = np.concatenate([t["residual_observations"] for t in traces])
    active = np.concatenate([t["residual_active"] for t in traces])
    return RoleBehaviorAnchor(
        reference,
        observations,
        active,
        str(
            hash_json(
                {
                    "source": "first_training_batch_v1",
                    "rollouts": [trajectory_digest(t) for t in traces],
                }
            )
        ),
        coefficient,
    )
