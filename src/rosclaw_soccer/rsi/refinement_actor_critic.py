"""Own-trajectory bounded AR PPO math for the WHOLE-parent refinement head.

This is not a collector, trainer, checkpoint admission, or a trained policy.
It cannot turn assisted feedback trajectories into on-policy experience.
Only the plastic head and detached-feature critic receive update gradients.
Frozen parent replay starts at zero on each whole actual observation sequence.
"""

from typing import Any

import torch

from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic
from rosclaw_soccer.rsi.motor_sampling_law import history_motor_kl, history_motor_logp
from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter


def _validate_episode(episode: dict[str, Any]) -> int:
    if (
        episode.get("controller_kind") != "pure-refinement-stochastic-native-v1"
        or episode.get("sampling_law") != "box-truncated-ar1"
        or any(
            episode.get(k) is not False
            for k in ("assisted", "external_teacher", "authority_rejected")
        )
        or any(
            episode.get(k) is not True
            for k in ("all_actor_actions_actually_executed", "natural_terminal_observed")
        )
        or episode.get("termination") not in ("BODY_FAILURE", "FULL_MOTION_PLUS100NN_RECOVERY")
    ):
        raise ValueError("whole pure learner bounded-stochastic native trajectory required")
    obs = episode["observations"]
    if (
        not isinstance(obs, torch.Tensor)
        or obs.ndim != 2
        or obs.shape[1] != 160
        or not 1 <= len(obs) <= 300
        or obs.dtype != torch.float32
        or not torch.isfinite(obs).all()
    ):
        raise ValueError("finite whole unpadded own160 observation sequence required")
    n = len(obs)
    tail, total = episode.get("observed_recovery_controls"), episode.get("total_observed_controls")
    motion = episode.get("motion_reference_controls")
    if (
        type(motion) is not int
        or not n <= motion <= 300
        or type(tail) is not int
        or not 0 <= tail <= 100
        or type(total) is not int
        or total != n + tail
        or episode.get("deterministic_recovery_excluded_from_actor_rows") is not True
        or (tail > 0 and n != motion)
        or (episode["termination"] == "FULL_MOTION_PLUS100NN_RECOVERY" and n != motion)
        or (episode["termination"] == "FULL_MOTION_PLUS100NN_RECOVERY" and tail != 100)
    ):
        raise ValueError("actual terminal recovery clock required; failures cannot be padded")
    shapes = {
        "actions": (n, 29),
        "old_means": (n, 29),
        "reference_nominal": (n, 29),
        "action_scale": (29,),
        "old_logp": (n,),
        "old_values": (n,),
        "returns": (n,),
        "advantages": (n,),
    }
    for key, shape in shapes.items():
        x = episode[key]
        if (
            not isinstance(x, torch.Tensor)
            or x.shape != shape
            or x.dtype != obs.dtype
            or x.device != obs.device
            or not torch.isfinite(x).all()
            or (key == "action_scale" and not (x > 0).all())
        ):
            raise ValueError("finite actual whole-episode field required: " + key)
    return n


def _replay(
    actor: RefinementMotorAdapter, episode: dict[str, Any], log_std: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any], int]:
    n = _validate_episode(episode)
    if (
        type(actor) is not RefinementMotorAdapter
        or actor.parent.training
        or any(p.requires_grad for p in actor.parent.parameters())
        or log_std.shape != (29,)
        or log_std.dtype != torch.float32
        or log_std.device != episode["observations"].device
        or log_std.requires_grad
        or not torch.isfinite(log_std).all()
    ):
        raise ValueError(
            "whole frozen parent, separate plastic head and fixed declared noise required"
        )
    means, _, evidence = actor.forward_with_evidence(
        episode["observations"][:, None], actor.initial_state()
    )
    # These are frozen codecs, not a new bound fitted to the sampled actions.
    expected_scale = actor.parent.memory.parent.get_buffer("action_scale")
    if not torch.equal(expected_scale, episode["action_scale"]) or not torch.allclose(
        evidence["reference_nominal"][:, 0], episode["reference_nominal"], atol=1e-5, rtol=1e-6
    ):
        raise ValueError("actual immutable original reference/scale must match own replay")
    return means[:, 0], evidence, n


@torch.no_grad()
def refinement_conditional_kl(
    actor: RefinementMotorAdapter, episode: dict[str, Any], log_std: torch.Tensor, *, rho: float
) -> float:
    means, _, _ = _replay(actor, episode, log_std)
    return float(
        history_motor_kl(
            "box-truncated-ar1",
            episode["old_means"],
            means,
            episode["actions"],
            episode["reference_nominal"],
            episode["action_scale"],
            log_std,
            rho=rho,
        )
    )


def refinement_own_history_loss(
    actor: RefinementMotorAdapter,
    critic: MotorCritic,
    episode: dict[str, Any],
    log_std: torch.Tensor,
    *,
    rho: float,
    value_only: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """No optimizer step. Failures remain learnable; teacher rescues do not.

    A complete BODY_FAILURE is an observed negative terminal, not missing
    recovery invented as100 controls. Returns must be calculated from actual
    physics by the admitted collector; this function never makes labels.
    """
    if type(value_only) is not bool:
        raise ValueError("explicit actor/critic update scope required")
    with torch.set_grad_enabled(not value_only):
        means, evidence, n = _replay(actor, episode, log_std)
    values = critic(evidence["critic_features"][:, 0].detach())
    old, returns = episode["old_values"], episode["returns"]
    clipped = old + (values - old).clamp(-0.2, 0.2)
    value_loss = (
        0.5 * torch.maximum((values - returns).square(), (clipped - returns).square()).mean()
    )
    with torch.no_grad():
        source_logp = history_motor_logp(
            "box-truncated-ar1",
            episode["old_means"],
            episode["actions"],
            episode["reference_nominal"],
            episode["action_scale"],
            log_std,
            rho=rho,
        )
        if not torch.allclose(source_logp, episode["old_logp"], atol=1e-4, rtol=1e-6):
            raise ValueError("actual bounded AR sampling density required; no Gaussian relabelling")
    if value_only:
        return value_loss, dict(actor_rows=0, critic_rows=n, physical_retention_proven=False)
    logp = history_motor_logp(
        "box-truncated-ar1",
        means,
        episode["actions"],
        episode["reference_nominal"],
        episode["action_scale"],
        log_std,
        rho=rho,
    )
    ratio_log = logp - episode["old_logp"]
    if not torch.isfinite(ratio_log).all() or ratio_log.abs().max() > 40:
        raise ValueError("finite trusted-region whole-history likelihood ratio required")
    ratio = ratio_log.exp()
    advantage = episode["advantages"]
    policy = -torch.minimum(ratio * advantage, ratio.clamp(0.9, 1.1) * advantage).mean()
    anchor = evidence["refinement_parent_mean"][:, 0].detach()
    anchor_penalty = 0.5 * ((means - anchor) / 0.02).square().sum(-1).mean()
    kl = history_motor_kl(
        "box-truncated-ar1",
        episode["old_means"],
        means,
        episode["actions"],
        episode["reference_nominal"],
        episode["action_scale"],
        log_std,
        rho=rho,
    )
    loss = policy + value_loss + 0.02 * anchor_penalty
    if not torch.isfinite(loss):
        raise ValueError("finite separate-head actor/critic loss required")
    return loss, dict(
        actor_rows=n,
        critic_rows=n,
        conditional_kl=kl.detach(),
        policy_loss=policy.detach(),
        value_loss=value_loss.detach(),
        whole_parent_anchor_penalty=anchor_penalty.detach(),
        physical_retention_proven=False,
        optimizer_steps_executed=0,
    )
