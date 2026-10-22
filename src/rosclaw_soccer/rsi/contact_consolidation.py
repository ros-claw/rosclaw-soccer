"""Offline whole-history plastic motor consolidation, never PPO or inference help."""

from typing import Any

import torch

from rosclaw_soccer.rsi.paid_trainable_actor import PaidTrainableActor

PROTOCOL = "dual256.contact_weighted_offline_consolidation.v1"
STATUS = "DUAL_CONTACT_CONSOLIDATED_PENDING_UNASSISTED_PHYSICAL_EXAM"
KIND = "contact-consolidated"


def consolidation_config(seed: int, device: str) -> dict[str, Any]:
    if seed not in (65023, 65024, 65025) or device not in ("cuda:0", "cpu"):
        raise ValueError("three preregistered sample-order seeds and explicit device required")
    return dict(
        protocol=PROTOCOL,
        seed=seed,
        device=device,
        steps=64,
        actor_lr=0.0001,
        motor_loss_unit_rad=0.02,
        old_anchor_weight=2.0,
        old_anchors_per_step=4,
        whole_history_from_zero=True,
        critic_frozen=True,
        initial_actor_steps=16,
        initial_critic_steps=48,
        final_actor_steps=80,
        final_critic_steps=48,
        learning_phase="offline_supervised_NOT_PPO",
        runtime_teacher=False,
        final_only=True,
    )


def sequence_loss(plastic: PaidTrainableActor, episode: dict[str, torch.Tensor]) -> torch.Tensor:
    """Full BPTT from private zero state; reference is independently precomputed."""
    obs = episode["observations"]
    if obs.ndim != 2 or obs.shape[1] != 160 or not 1 <= len(obs) <= 400:
        raise ValueError("one bounded complete actual history required")
    state = (obs.new_zeros(2, 1, 128), obs.new_zeros(2, 1, 128))
    learned, _, _ = plastic.forward_with_features(obs[:, None], state)
    nominal, prior = episode["reference_nominal"], episode["reference_delta"]
    scale = episode["action_scale"]
    delta = torch.clamp(prior + (learned[:, 0] - nominal) * scale, -0.2, 0.2)
    mean = nominal + delta / scale
    errors = ((mean - episode["targets"]) * scale / 0.02).square().mean(dim=-1)
    weights = episode["loss_weights"]
    if weights.shape != errors.shape or (weights <= 0).any():
        raise ValueError("positive measured contact/prefix/recovery weights required")
    return (errors * weights).sum() / weights.sum()
