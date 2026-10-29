import math

import numpy as np
import pytest
import torch
from test_velocity_prior_actor import tensors

from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
from rosclaw_soccer.rsi.precontact_receiving_motor import PrecontactReceivingMotor
from rosclaw_soccer.rsi.receiver_actor_critic import (
    ReceiverCritic,
    episode_balanced_value_loss,
    receiver_own_history_loss,
)
from rosclaw_soccer.rsi.receiver_exploration_stream import ReceiverExplorationStream
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor


def episode():
    n = 5
    obs = torch.randn(n, 112)
    ref = torch.zeros(n, 29)
    means = torch.zeros(n, 29)
    sampler = ReceiverExplorationStream(np.full(29, 0.25, np.float32), seed=71)
    actions, logp = [], []
    for t in range(n):
        action, trace = sampler.propose(means[t].numpy(), ref[t].numpy(), control=t)
        sampler.commit(action)
        actions.append(action)
        logp.append(trace["logp"])
    return dict(
        observations=obs,
        actions=torch.from_numpy(np.stack(actions)),
        old_means=means,
        reference=ref,
        old_logp=torch.from_numpy(np.stack(logp)),
        returns=torch.ones(n),
        advantages=torch.tensor([1.0, -1.0, 0.5, -0.5, 0.25]),
        action_scale=torch.full((29,), 0.25),
        sampling_law="box-truncated-ar1",
        all_actor_actions_actually_executed=True,
        density_audited=True,
        full_task_outcome_observed=True,
        assisted=False,
        external_teacher=False,
        authority_rejected=False,
    )


def test_real_conditioned_loss_updates_actor_and_critic_but_not_anchor():
    actor = AnchoredReceivingMotor(
        VelocityPriorActor(tensors(), reference_features=14), torch.full((29,), 0.25)
    ).train()
    critic, data = ReceiverCritic(), episode()
    loss, metrics = receiver_own_history_loss(
        actor, critic, data, torch.full((29,), math.log(0.02)), rho=0.9
    )
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.plastic_parameters())
    assert all(p.grad is None and not p.requires_grad for p in actor.baseline.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters())
    assert (
        metrics["optimizer_steps_executed"] == 0 and metrics["physical_retention_proven"] is False
    )


def test_incorrect_probability_or_unproven_execution_fails_closed():
    actor = AnchoredReceivingMotor(
        VelocityPriorActor(tensors(), reference_features=14), torch.full((29,), 0.25)
    ).train()
    critic, data = ReceiverCritic(), episode()
    data["old_logp"][0] += 1
    with pytest.raises(ValueError, match="sampling density"):
        receiver_own_history_loss(actor, critic, data, torch.full((29,), math.log(0.02)), rho=0.9)
    data["all_actor_actions_actually_executed"] = False
    with pytest.raises(ValueError, match="audited pure"):
        receiver_own_history_loss(actor, critic, data, torch.full((29,), math.log(0.02)), rho=0.9)


def test_critic_weights_episodes_equally_not_by_length():
    critic = ReceiverCritic()
    episodes = [
        {"observations": torch.zeros(1, 112), "returns": torch.full((1,), 2.0)},
        {"observations": torch.zeros(100, 112), "returns": torch.zeros(100)},
    ]
    assert episode_balanced_value_loss(critic, episodes).item() == pytest.approx(1.0)
    episodes[1]["returns"][0] = float("nan")
    with pytest.raises(ValueError, match="finite complete"):
        episode_balanced_value_loss(critic, episodes)


def test_explicit_precontact_motor_can_learn_on_audited_history():
    actor = PrecontactReceivingMotor(
        VelocityPriorActor(tensors(), reference_features=14), torch.full((29,), 0.25)
    )
    data = episode()
    data["observations"][:, 98:101] = torch.tensor([1.0, 0.0, 0.0])
    data["observations"][:, 101:104] = torch.tensor([-2.0, 0.0, 0.0])
    loss, _ = receiver_own_history_loss(
        actor, ReceiverCritic(), data, torch.full((29,), math.log(0.02)), rho=0.9
    )
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.plastic_parameters())


def test_arbitrary_motor_subclass_is_not_accepted():
    class UnreviewedMotor(AnchoredReceivingMotor):
        pass

    actor = UnreviewedMotor(
        VelocityPriorActor(tensors(), reference_features=14), torch.full((29,), 0.25)
    )
    with pytest.raises(ValueError, match="audited pure"):
        receiver_own_history_loss(
            actor, ReceiverCritic(), episode(), torch.full((29,), math.log(0.02)), rho=0.9
        )
