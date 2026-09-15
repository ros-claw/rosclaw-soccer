"""Off-policy imitation of a qualified local motor window, never promotion.

Only one private actor's output layer is plastic. Its encoder, value head,
exploration scale and all seven other players remain byte-identical. Callers
own physical demonstration qualification, Core leases and fresh world exams.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy


def fit_private_motor_window(
    parent: NearBallResidualPolicy,
    *,
    agent_id: str,
    observations: np.ndarray,
    raw_actions: np.ndarray,
    anchor_observations: np.ndarray,
    steps: int = 64,
    learning_rate: float = 0.001,
    anchor_coefficient: float = 0.5,
) -> tuple[NearBallResidualPolicy, dict[str, Any]]:
    """Deterministic output-head regression; replay is explicitly not on-policy."""
    if (
        not isinstance(parent, NearBallResidualPolicy)
        or agent_id not in parent.agent_ids
        or type(steps) is not int
        or not 1 <= steps <= 128
        or any(
            type(v) not in (int, float) or not math.isfinite(v)
            for v in (learning_rate, anchor_coefficient)
        )
        or not 1e-5 <= learning_rate <= 0.003
        or not 0.1 <= anchor_coefficient <= 2.0
    ):
        raise ValueError("bounded single-player off-policy imitation contract required")
    for array, width, limit in (
        (observations, parent.observation_dim, 10.0),
        (raw_actions, 12, 20.0),
        (anchor_observations, parent.observation_dim, 10.0),
    ):
        if (
            not isinstance(array, np.ndarray)
            or array.ndim != 2
            or array.shape[1] != width
            or not 1 <= len(array) <= 4096
            or array.dtype.kind != "f"
            or not np.isfinite(array).all()
            or np.any(np.abs(array) > limit)
        ):
            raise ValueError("finite nonempty bounded motor demonstration and anchor required")
    if len(raw_actions) != len(observations):
        raise ValueError("motor actions must align with observed demonstration states")
    import torch

    index = parent.agent_ids.index(agent_id)
    weights = {k: v.copy() for k, v in parent.weights.items()}
    encoder = torch.tensor(weights["w1"][index], dtype=torch.float64)
    bias = torch.tensor(weights["b1"][index], dtype=torch.float64)
    hidden = torch.tanh(torch.tensor(observations, dtype=torch.float64) @ encoder + bias)
    anchor = torch.tanh(torch.tensor(anchor_observations, dtype=torch.float64) @ encoder + bias)
    target = torch.tensor(raw_actions, dtype=torch.float64)
    w = torch.nn.Parameter(torch.tensor(weights["w2"][index], dtype=torch.float64))
    b = torch.nn.Parameter(torch.tensor(weights["b2"][index], dtype=torch.float64))
    anchor_mean = (anchor @ w + b).detach().clone()
    before = float((hidden @ w + b - target).square().mean().detach())
    optimizer = torch.optim.Adam((w, b), lr=learning_rate)
    for _ in range(steps):
        loss = (hidden @ w + b - target).square().mean()
        loss = loss + anchor_coefficient * (anchor @ w + b - anchor_mean).square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite private motor imitation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        norm = torch.nn.utils.clip_grad_norm_((w, b), 0.5)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("nonfinite private motor imitation gradient")
        optimizer.step()
    weights["w2"][index] = w.detach().numpy()
    weights["b2"][index] = b.detach().numpy()
    child = NearBallResidualPolicy(
        parent.agent_ids,
        parent.body_hash,
        parent.generation + 1,
        parent.policy_hash,
        weights,
        parent.observation_contract,
    )
    return child, {
        "schema": "soccer.private_motor_window_imitation.v1",
        "agent_id": agent_id,
        "optimizer_steps": steps,
        "demonstration_samples": len(observations),
        "anchor_samples": len(anchor_observations),
        "learning_rate": learning_rate,
        "anchor_coefficient": anchor_coefficient,
        "demonstration_mse_before": before,
        "demonstration_mse_after": float((hidden @ w + b - target).square().mean().detach()),
        "anchor_mean_mse_after": float((anchor @ w + b - anchor_mean).square().mean().detach()),
        "plastic_keys": ["w2", "b2"],
        "on_policy": False,
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
    }
