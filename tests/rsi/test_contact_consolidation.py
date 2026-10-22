import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.contact_consolidation import consolidation_config, sequence_loss
from rosclaw_soccer.rsi.contact_consolidation_data import (
    BANK_STATUS,
    contact_loss_weights,
    context_witness_hash,
    own_reference_sequence,
)


def test_complete_history_loss_updates_private_recurrent_branch_only():
    from test_dual_recurrent_motor import model

    parent, learner = model()
    obs = np.random.default_rng(17).normal(size=(36, 160)).astype(np.float32)
    nominal, prior = own_reference_sequence(parent, obs)
    episode = {
        "observations": torch.from_numpy(obs),
        "reference_nominal": torch.from_numpy(nominal),
        "reference_delta": torch.from_numpy(prior),
        "action_scale": parent.action_scale,
        "loss_weights": torch.from_numpy(contact_loss_weights(26, 10, 215)),
        "targets": torch.from_numpy(nominal + prior / parent.action_scale.numpy() + 0.05),
    }
    frozen = {k: v.clone() for k, v in parent.state_dict().items()}
    before = learner.plastic.rnn.weight_ih_l0.detach().clone()
    optimizer = torch.optim.Adam(learner.plastic_parameters(), lr=0.0001)
    loss = sequence_loss(learner.plastic, episode)
    assert loss.isfinite() and loss > 0
    loss.backward()
    assert learner.plastic.rnn.weight_ih_l0.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, learner.plastic.rnn.weight_ih_l0)
    assert all(torch.equal(v, frozen[k]) for k, v in parent.state_dict().items())
    assert all(p.grad is None for p in parent.parameters())


def test_three_seeds_have_same_budget_and_distinct_order_not_three_named_identical_runs():
    import random

    orders = []
    for seed in (65023, 65024, 65025):
        config = consolidation_config(seed, "cuda:0")
        assert config["steps"] == 64 and config["final_actor_steps"] == 80
        assert config["final_critic_steps"] == 48 and config["runtime_teacher"] is False
        rng = random.Random(seed)
        orders.append([rng.randrange(48) for _ in range(64)])
    assert len({tuple(o) for o in orders}) == 3
    with pytest.raises(ValueError):
        consolidation_config(42, "cuda:0")


def test_new_bank_does_not_replace_original_exam_context_witness():
    data = dict(status=BANK_STATUS, context_witness_hash="original", teacher_hash="wrong")
    assert context_witness_hash(data, "contact-consolidated") == "original"
    with pytest.raises(ValueError):
        context_witness_hash(data, "ar1-actor-critic")
    with pytest.raises(ValueError):
        context_witness_hash(dict(status="old", teacher_hash="old"), "contact-consolidated")
    assert context_witness_hash(dict(status="old", teacher_hash="old"), "supervised") == "old"
