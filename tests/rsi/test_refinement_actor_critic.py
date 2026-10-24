import math

import pytest
import torch
from test_refinement_motor_adapter import parent_model

from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic
from rosclaw_soccer.rsi.motor_sampling_law import draw_motor_action, history_motor_logp
from rosclaw_soccer.rsi.refinement_actor_critic import (
    refinement_conditional_kl,
    refinement_own_history_loss,
)
from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter


def fixture(n=12):
    """Probability/gradient unit fixture ONLY; never a native rollout proof."""
    actor = RefinementMotorAdapter(parent_model())
    critic = MotorCritic()
    obs = torch.randn(n, 160) * 0.1
    with torch.no_grad():
        means, _, ev = actor.forward_with_evidence(obs[:, None], actor.initial_state())
    means = means[:, 0]
    reference = ev["reference_nominal"][:, 0]
    scale = actor.parent.memory.parent.get_buffer("action_scale").clone()
    std = torch.full((29,), math.log(0.01), dtype=torch.float32)
    gen = torch.Generator().manual_seed(11)
    actions = []
    for i in range(n):
        action, _ = draw_motor_action(
            "box-truncated-ar1",
            means[i],
            reference[i],
            scale,
            std,
            rho=0.9,
            previous_action=None if not i else actions[-1],
            previous_mean=None if not i else means[i - 1],
            generator=gen,
        )
        actions.append(action)
    actions = torch.stack(actions)
    ep = dict(
        controller_kind="pure-refinement-stochastic-native-v1",
        sampling_law="box-truncated-ar1",
        observations=obs,
        actions=actions,
        old_means=means,
        reference_nominal=reference,
        action_scale=scale,
        old_logp=history_motor_logp(
            "box-truncated-ar1", means, actions, reference, scale, std, rho=0.9
        ),
        old_values=torch.zeros(n),
        returns=torch.ones(n),
        advantages=torch.linspace(-1, 1, n),
        assisted=False,
        external_teacher=False,
        authority_rejected=False,
        all_actor_actions_actually_executed=True,
        natural_terminal_observed=True,
        termination="FULL_MOTION_PLUS100NN_RECOVERY",
        motion_reference_controls=n,
        observed_recovery_controls=100,
        total_observed_controls=n + 100,
        deterministic_recovery_excluded_from_actor_rows=True,
    )
    return actor, critic, ep, std


def test_full389_actual_history_recovery_alone_has_policy_gradient_and_kl_coverage():
    actor, critic, ep, std = fixture(389)
    ep.update(
        controller_kind="pure-refinement-full-recovery-stochastic-native-v2",
        motion_reference_controls=289,
        total_observed_controls=389,
        deterministic_recovery_excluded_from_actor_rows=False,
    )
    ep["advantages"].zero_()
    ep["advantages"][289:] = 1.0
    loss, info = refinement_own_history_loss(
        actor, critic, ep, std, rho=0.9, expose_actor_components=True
    )
    assert torch.isfinite(loss) and info["actor_rows"] == info["critic_rows"] == 389
    gradients = torch.autograd.grad(info["policy_term"], list(actor.head.parameters()))
    assert sum(g.abs().sum() for g in gradients) > 0
    assert refinement_conditional_kl(actor, ep, std, rho=0.9) < 1e-7
    ep["old_logp"][300] += 1
    with pytest.raises(ValueError, match="density"):
        refinement_own_history_loss(actor, critic, ep, std, rho=0.9)


def test_new_head_and_critic_get_gradients_but_entire_acquired_parent_does_not():
    actor, critic, ep, std = fixture()
    loss, info = refinement_own_history_loss(actor, critic, ep, std, rho=0.9)
    assert info["conditional_kl"] < 1e-7
    loss.backward()
    assert actor.head[-1].bias.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in actor.parent.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters())
    assert not info["physical_retention_proven"] and info["optimizer_steps_executed"] == 0


def test_actor_gradient_components_reproduce_actual_backward_without_updates():
    actor, critic, ep, std = fixture()
    parameters = list(actor.head.parameters())
    before = [p.detach().clone() for p in parameters]
    loss, info = refinement_own_history_loss(
        actor, critic, ep, std, rho=0.9, expose_actor_components=True
    )
    pg = torch.autograd.grad(info["policy_term"], parameters, retain_graph=True)
    ag = torch.autograd.grad(info["anchor_term"], parameters, retain_graph=True)
    loss.backward()
    for p, x, y, old in zip(parameters, pg, ag, before, strict=True):
        assert torch.allclose(p.grad, x + y, atol=1e-4, rtol=1e-3)
        assert torch.equal(p, old)
    with pytest.raises(ValueError):
        refinement_own_history_loss(
            actor, critic, ep, std, rho=0.9, value_only=True, expose_actor_components=True
        )


def test_value_only_cannot_update_motor_weights():
    actor, critic, ep, std = fixture()
    loss, info = refinement_own_history_loss(actor, critic, ep, std, rho=0.9, value_only=True)
    loss.backward()
    assert all(p.grad is None for p in actor.parameters())
    assert info["actor_rows"] == 0 and info["critic_rows"] == 12


def test_observed_failure_remains_learnable_without_fabricated_recovery():
    actor, critic, ep, std = fixture()
    ep.update(termination="BODY_FAILURE", observed_recovery_controls=0, total_observed_controls=12)
    ep["returns"].fill_(-1)
    loss, info = refinement_own_history_loss(actor, critic, ep, std, rho=0.9)
    assert torch.isfinite(loss) and info["actor_rows"] == 12


@pytest.mark.parametrize(
    "key,value",
    [
        ("controller_kind", "teacher-feedback"),
        ("sampling_law", "gaussian-ar1"),
        ("assisted", True),
        ("external_teacher", True),
        ("authority_rejected", True),
        ("all_actor_actions_actually_executed", False),
        ("natural_terminal_observed", False),
        ("observed_recovery_controls", 99),
        ("total_observed_controls", 111),
        ("motion_reference_controls", 13),
        ("deterministic_recovery_excluded_from_actor_rows", False),
    ],
)
def test_invalid_onpolicy_or_terminal_contract_rejected(key, value):
    actor, critic, ep, std = fixture()
    ep[key] = value
    with pytest.raises(ValueError):
        refinement_own_history_loss(actor, critic, ep, std, rho=0.9)


def test_fake_source_density_rejected_even_for_value_only():
    actor, critic, ep, std = fixture()
    ep["old_logp"] = ep["old_logp"] + 1
    for value_only in (False, True):
        with pytest.raises(ValueError, match="density"):
            refinement_own_history_loss(actor, critic, ep, std, rho=0.9, value_only=value_only)


def test_original_frozen_reference_cannot_be_replaced_with_wider_action_box():
    actor, critic, ep, std = fixture()
    ep["reference_nominal"] = ep["reference_nominal"] + 0.01
    with pytest.raises(ValueError, match="immutable"):
        refinement_own_history_loss(actor, critic, ep, std, rho=0.9)


def test_kl_responds_to_actual_changed_head_not_borrowed_source_hidden_states():
    actor, _, ep, std = fixture()
    ep["source_hidden_state"] = torch.full((2, 1, 256), float("nan"))
    assert refinement_conditional_kl(actor, ep, std, rho=0.9) < 1e-7
    with torch.no_grad():
        actor.head[-1].bias.add_(0.1)
    assert refinement_conditional_kl(actor, ep, std, rho=0.9) > 0.015
