"""Bounded numerical Adam proposal with joint parameter AND moment rollback.

Requires a deterministic, side-effect-free tensor loss closure on fixed data.
No model activation, simulator, world, transport or promotion authority.
"""

import copy
import math
from collections.abc import Callable, Sequence
from typing import Any


def bounded_adam_proposal(
    parameters: Sequence[Any],
    optimizer: Any,
    closure: Callable[[], tuple[Any, dict[str, Any]]],
) -> dict[str, Any]:
    import torch

    params = tuple(parameters)
    if not params or any(not p.requires_grad for p in params):
        raise ValueError("explicit nonempty trainable proposal parameters required")
    previous = [p.detach().clone() for p in params]
    previous_optimizer = copy.deepcopy(optimizer.state_dict())

    def restore() -> None:
        with torch.no_grad():
            for p, old in zip(params, previous, strict=True):
                p.copy_(old)
        optimizer.load_state_dict(copy.deepcopy(previous_optimizer))

    try:
        optimizer.zero_grad(set_to_none=True)
        initial, _ = closure()
        if initial.ndim != 0 or not torch.isfinite(initial):
            raise ValueError("finite scalar initial loss required")
        before = float(initial.detach())
        initial.backward()
        gradient = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        proposed = [p.detach().clone() for p in params]
        if any(not torch.isfinite(p).all() for p in proposed):
            raise ValueError("nonfinite Adam proposal")
        rates = [float(g["lr"]) for g in optimizer.param_groups]
        for reduction in range(13):
            alpha = 0.5**reduction
            with torch.no_grad():
                for p, old, new in zip(params, previous, proposed, strict=True):
                    p.copy_(new if reduction == 0 else old + alpha * (new - old))
                loss, stats = closure()
                after, kl = float(loss), float(stats["mean_kl"])
                if (
                    math.isfinite(after)
                    and math.isfinite(kl)
                    and -1e-6 <= kl <= 0.005
                    and after < before - 1e-10
                ):
                    return dict(
                        accepted=True,
                        alpha=alpha,
                        backtracking_reductions=reduction,
                        effective_learning_rates=[r * alpha for r in rates],
                        initial_loss=before,
                        accepted_loss=after,
                        gradient_norm=float(gradient),
                        **{k: float(v) for k, v in stats.items()},
                    )
        restore()
        return dict(
            accepted=False,
            alpha=0.0,
            backtracking_reductions=13,
            initial_loss=before,
            optimizer_state_restored=True,
        )
    except Exception:
        restore()
        raise
