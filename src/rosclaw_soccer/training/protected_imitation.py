"""Offline action distillation with an immutable pre-event actor.

Callers must qualify demonstrations using physical evidence, bind the dataset
and weight update, and independently test the resulting closed-loop policy.
Fitting loss is neither a skill-success certificate nor activation authority.
Models are trusted in-process objects; this module never deserializes models.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np


def fit_protected_demonstration(
    seed: Any,
    observation: np.ndarray,
    raw_action: np.ndarray,
    *,
    steps: int = 512,
    learning_rate: float = 1e-4,
    retention_coefficient: float = 0.1,
    rehearsal_observation: np.ndarray | None = None,
    rehearsal_coefficient: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
    """Fit only ``plastic.actor`` on explicitly executed post-event actions.

    The seed follows ``build_event_protected_actor_critic``'s module layout.
    The final observation feature must be one; ignored pre-event exploration
    must never be presented as an executed demonstration. All non-actor state,
    including the critic and exploration scale, is preserved exactly. Input
    arrays and the seed remain untouched, including their gradient flags.
    Optional rehearsal states are a separate, caller-qualified set of visited
    post-event states. Their targets are the seed's deterministic means, not
    recorded but unexecuted noise. This is a soft retention loss, never proof
    of closed-loop retention. The default preserves the original update math.
    """
    import torch

    if (
        type(steps) is not int
        or not 1 <= steps <= 1024
        or any(
            type(v) not in (int, float) or not math.isfinite(v)
            for v in (learning_rate, retention_coefficient, rehearsal_coefficient)
        )
        or not 1e-6 <= learning_rate <= 1e-3
        or not 0 <= retention_coefficient <= 10
        or not 0 <= rehearsal_coefficient <= 100
        or (rehearsal_observation is None) != (rehearsal_coefficient == 0)
        or not isinstance(seed, torch.nn.Module)
        or not isinstance(getattr(seed, "anchor", None), torch.nn.Module)
        or not isinstance(getattr(seed, "plastic", None), torch.nn.Module)
        or not isinstance(getattr(seed.plastic, "actor", None), torch.nn.Module)
        or not isinstance(getattr(seed.plastic, "critic", None), torch.nn.Module)
    ):
        raise ValueError("bounded fit configuration and an explicit protected actor required")
    if (
        not isinstance(observation, np.ndarray)
        or not isinstance(raw_action, np.ndarray)
        or observation.dtype != np.float32
        or raw_action.dtype != np.float32
        or observation.ndim != 2
        or raw_action.ndim != 2
        or not 1 <= len(observation) <= 4096
        or len(observation) != len(raw_action)
        or not 2 <= observation.shape[1] <= 1025
        or not 1 <= raw_action.shape[1] <= 256
        or not np.isfinite(observation).all()
        or not np.isfinite(raw_action).all()
        or np.any(np.abs(observation) > 10)
        or np.any(np.abs(raw_action) > 20)
        or not np.all(observation[:, -1] == 1)
    ):
        raise ValueError("finite float32 executed post-event observation/action pairs required")
    if rehearsal_observation is not None and (
        not isinstance(rehearsal_observation, np.ndarray)
        or rehearsal_observation.dtype != np.float32
        or rehearsal_observation.ndim != 2
        or not 1 <= len(rehearsal_observation) <= 4096
        or rehearsal_observation.shape[1] != observation.shape[1]
        or not np.isfinite(rehearsal_observation).all()
        or np.any(np.abs(rehearsal_observation) > 10)
        or not np.all(rehearsal_observation[:, -1] == 1)
    ):
        raise ValueError("finite float32 visited post-event rehearsal observations required")
    state = seed.state_dict()
    if not state or any(
        v.dtype != torch.float32
        or v.device.type != "cpu"
        or v.layout != torch.strided
        or not bool(torch.isfinite(v).all())
        or bool((v.abs() > 20).any())
        for v in state.values()
    ):
        raise ValueError("bounded finite CPU float32 model state required")
    plastic: Any = seed.plastic
    protected: Any = seed.anchor
    actor_ids = {id(p) for p in plastic.actor.parameters()}
    other_ids = {id(p) for p in protected.parameters()} | {
        id(p) for p in plastic.critic.parameters()
    }
    actor_storage = {
        v.untyped_storage().data_ptr()
        for name, v in state.items()
        if name.startswith("plastic.actor.")
    }
    other_storage = {
        v.untyped_storage().data_ptr()
        for name, v in state.items()
        if not name.startswith("plastic.actor.")
    }
    if not actor_ids or actor_ids & other_ids or actor_storage & other_storage:
        raise ValueError("plastic actor must not share parameters with protected modules")
    model: Any = copy.deepcopy(seed).eval()
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    before = {name: v.detach().clone() for name, v in model.state_dict().items()}
    model.requires_grad_(False)
    model.plastic.actor.requires_grad_(True)
    parameters = list(model.plastic.actor.parameters())
    x = torch.from_numpy(observation.copy())
    y = torch.from_numpy(raw_action.copy())
    with torch.no_grad():
        anchor, _ = model(x)
        anchor = anchor.detach().clone()
    if anchor.shape != y.shape:
        raise ValueError("demonstration action shape differs from the protected actor")
    rehearsal_x = (
        None if rehearsal_observation is None else torch.from_numpy(rehearsal_observation.copy())
    )
    rehearsal_target = None
    if rehearsal_x is not None:
        with torch.no_grad():
            rehearsal_target = model(rehearsal_x)[0].detach().clone()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    def objective() -> Any:
        predicted, _ = model(x)
        loss = (predicted - y).square().mean() + retention_coefficient * (
            predicted - anchor
        ).square().mean()
        if rehearsal_x is not None:
            loss = (
                loss
                + rehearsal_coefficient * (model(rehearsal_x)[0] - rehearsal_target).square().mean()
            )
        return loss

    initial = float(objective().detach())
    for _ in range(steps):
        loss = objective()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite demonstration loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
    final = float(objective().detach())
    if not math.isfinite(final) or any(
        not bool(torch.isfinite(v).all()) or bool((v.abs() > 20).any())
        for v in model.state_dict().values()
    ):
        raise FloatingPointError("distillation produced invalid weights or loss")
    if any(
        not torch.equal(v, before[name])
        for name, v in model.state_dict().items()
        if not name.startswith("plastic.actor.")
    ):
        raise ValueError("offline distillation modified protected model state")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(flags[name])
    stats = {
        "steps": steps,
        "initial_loss": initial,
        "final_loss": final,
        "supervised_raw_action_samples": len(observation),
        "on_policy": False,
        "frozen_approach_preserved": True,
        "critic_and_exploration_preserved": True,
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
    }
    if rehearsal_x is not None:
        with torch.no_grad():
            drift = float((model(rehearsal_x)[0] - rehearsal_target).square().mean())
        stats.update(
            rehearsal_samples=len(rehearsal_x),
            rehearsal_coefficient=rehearsal_coefficient,
            rehearsal_mean_squared_drift=drift,
            closed_loop_retention_verified=False,
        )
    return model, stats
