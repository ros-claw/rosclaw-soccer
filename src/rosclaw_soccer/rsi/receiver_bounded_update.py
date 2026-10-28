"""One offline actor proposal with reversible numerical retention gates.

This utility is not a dataset admission, physics evaluation, or policy promotion.
"""

import math
from typing import Any

import torch

from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
from rosclaw_soccer.rsi.motor_sampling_law import history_motor_kl
from rosclaw_soccer.rsi.receiver_actor_critic import ReceiverCritic, receiver_own_history_loss


def bounded_actor_update(
    actor: AnchoredReceivingMotor,
    critic: ReceiverCritic,
    episodes: list[dict[str, Any]],
    log_std: torch.Tensor,
    anchor_observations: torch.Tensor,
    anchor_reference: torch.Tensor,
    *,
    step_size: float = 1e-7,
    max_anchor_delta_rad: float = 0.005,
    max_mean_kl: float = 0.02,
    max_attempts: int = 8,
) -> dict[str, Any]:
    """One episode-weighted actor gradient, accepted only if both gates pass."""
    if (
        not episodes
        or type(max_attempts) is not int
        or not 1 <= max_attempts <= 8
        or not all(
            math.isfinite(v) and v > 0 for v in (step_size, max_anchor_delta_rad, max_mean_kl)
        )
        or anchor_observations.ndim != 2
        or anchor_observations.shape[1] != 112
        or anchor_reference.shape != (len(anchor_observations), 29)
        or not 1 <= len(anchor_observations) <= 8192
        or any(p.requires_grad for p in critic.parameters())
    ):
        raise ValueError("bounded episode set, frozen critic and full anchor data required")
    params = actor.plastic_parameters()
    if not params or any(not p.requires_grad for p in params):
        raise ValueError("all plastic actor tensors must be trainable")
    # A new generation must start from the behavior policy that actually
    # generated its full history, not from the original untrained anchor.
    # Density checks alone cannot establish this: they only validate the
    # stored old means against the stored actions.
    with torch.no_grad():
        max_behavior_error = 0.0
        for episode in episodes:
            current = actor(episode["observations"], episode["reference"])
            old = episode["old_means"]
            if current.shape != old.shape or not torch.isfinite(old).all():
                raise ValueError("finite matching behavior-policy means required")
            error = float((current - old).abs().max())
            max_behavior_error = max(max_behavior_error, error)
            if not math.isfinite(error) or error > 1e-5:
                raise ValueError("actor must match the sampled behavior policy before update")
    original = [p.detach().clone() for p in params]
    for p in params:
        p.grad = None
    for episode in episodes:
        loss, _ = receiver_own_history_loss(actor, critic, episode, log_std, rho=0.9)
        (loss / len(episodes)).backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
        raise ValueError("all plastic gradients must be finite")
    gradients = [p.grad.detach().clone() for p in params]
    if not any(bool(torch.count_nonzero(g)) for g in gradients):
        raise ValueError("nonzero actor gradient required")
    attempts = []
    accepted = False
    try:
        for index in range(max_attempts):
            rate = step_size * (0.5**index)
            with torch.no_grad():
                for p, old, grad in zip(params, original, gradients, strict=True):
                    p.copy_(old - rate * grad)
                finite = all(bool(torch.isfinite(p).all()) for p in params)
                changed = any(
                    not torch.equal(p, old) for p, old in zip(params, original, strict=True)
                )
                if not finite or not changed:
                    attempts.append(dict(step=rate, finite=finite, changed=changed, accepted=False))
                    continue
                _, evidence = actor.forward_with_evidence(anchor_observations, anchor_reference)
                delta = float(evidence["realized_delta_radians"].abs().max())
                kls = []
                for ep in episodes:
                    mean = actor(ep["observations"], ep["reference"])
                    kl = history_motor_kl(
                        "box-truncated-ar1",
                        ep["old_means"],
                        mean,
                        ep["actions"],
                        ep["reference"],
                        ep["action_scale"],
                        log_std,
                        rho=0.9,
                    )
                    kls.append(float(kl))
                mean_kl = sum(kls) / len(kls)
                accepted = (
                    math.isfinite(delta)
                    and math.isfinite(mean_kl)
                    and delta <= max_anchor_delta_rad
                    and 0 <= mean_kl <= max_mean_kl
                )
                attempts.append(
                    dict(
                        step=rate,
                        finite=True,
                        changed=True,
                        anchor_delta_rad=delta,
                        mean_conditional_kl=mean_kl,
                        accepted=accepted,
                    )
                )
                if accepted:
                    break
    finally:
        if not accepted:
            with torch.no_grad():
                for p, old in zip(params, original, strict=True):
                    p.copy_(old)
        for p in params:
            p.grad = None
    return dict(
        accepted=accepted,
        attempts=attempts,
        actor_updates=int(accepted),
        max_behavior_mean_error=max_behavior_error,
        physical_retention_proven=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
