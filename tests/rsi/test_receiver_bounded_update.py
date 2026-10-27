import math

import pytest
import torch
from test_receiver_actor_critic import episode
from test_velocity_prior_actor import tensors

from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
from rosclaw_soccer.rsi.receiver_actor_critic import ReceiverCritic
from rosclaw_soccer.rsi.receiver_bounded_update import bounded_actor_update
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor


def build():
    torch.manual_seed(9)
    actor = AnchoredReceivingMotor(
        VelocityPriorActor(tensors(), reference_features=14), torch.full((29,), 0.25)
    )
    critic = ReceiverCritic().eval().requires_grad_(False)
    data = episode()
    return actor, critic, data


def test_actor_update_obeys_numerical_gates_and_never_changes_anchor():
    actor, critic, data = build()
    old_anchor = [p.detach().clone() for p in actor.baseline.parameters()]
    result = bounded_actor_update(
        actor,
        critic,
        [data],
        torch.full((29,), math.log(0.02)),
        data["observations"],
        data["reference"],
    )
    assert result["attempts"]
    if result["accepted"]:
        last = result["attempts"][-1]
        assert last["anchor_delta_rad"] <= 0.005 and last["mean_conditional_kl"] <= 0.02
    assert all(
        torch.equal(a, b) for a, b in zip(old_anchor, actor.baseline.parameters(), strict=True)
    )
    assert result["physical_retention_proven"] is False


def test_rejected_update_restores_exact_weights_and_invalid_data_does_not_mutate():
    actor, critic, data = build()
    before = [p.detach().clone() for p in actor.plastic_parameters()]
    result = bounded_actor_update(
        actor,
        critic,
        [data],
        torch.full((29,), math.log(0.02)),
        data["observations"],
        data["reference"],
        step_size=0.1,
        max_anchor_delta_rad=1e-15,
        max_attempts=2,
    )
    assert result["accepted"] is False
    assert all(torch.equal(a, b) for a, b in zip(before, actor.plastic_parameters(), strict=True))
    data["old_logp"][0] += 1
    with pytest.raises(ValueError, match="sampling density"):
        bounded_actor_update(
            actor,
            critic,
            [data],
            torch.full((29,), math.log(0.02)),
            data["observations"],
            data["reference"],
        )
    assert all(torch.equal(a, b) for a, b in zip(before, actor.plastic_parameters(), strict=True))
