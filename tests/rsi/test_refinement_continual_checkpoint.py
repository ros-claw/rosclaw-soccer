import copy
from collections import OrderedDict

import pytest
import torch
from test_refinement_actor_critic import fixture as motor_fixture

from rosclaw_soccer.rsi.continuing_refinement_contract import continuing_config, contract_for
from rosclaw_soccer.rsi.refinement_continual_batch import continual_config
from rosclaw_soccer.rsi.refinement_continual_checkpoint import (
    KIND,
    SCHEMA,
    rebased_adam,
    same_nested,
    validate_continual_checkpoint,
)
from rosclaw_soccer.rsi.refinement_onpolicy_learning import tensor_state_hash


def fixture(prior_steps=4, generation=1, full=False):
    """Adam-continuity UNIT fixture, NOT physical or RL-performance evidence."""
    actor, critic, _, _ = motor_fixture()
    cfg = (
        continual_config(25) if generation == 1 else continuing_config(25, generation, prior_steps)
    )
    if full:
        from rosclaw_soccer.rsi.continuing_refinement_contract import full_recovery_config

        cfg = full_recovery_config(25)
    contract = contract_for(cfg)
    parameters = [*actor.head.parameters(), *critic.parameters()]
    optimizer = torch.optim.Adam(
        [
            dict(params=list(actor.head.parameters()), lr=cfg["actor_lr"]),
            dict(params=list(critic.parameters()), lr=cfg["critic_lr"]),
        ]
    )

    def step():
        optimizer.zero_grad(set_to_none=True)
        sum((p + 0.1).square().sum() for p in parameters).backward()
        optimizer.step()

    for _ in range(prior_steps):
        step()
    behavior, value = copy.deepcopy(actor).eval(), copy.deepcopy(critic).eval()
    prior = copy.deepcopy(optimizer.state_dict())
    prior["param_groups"][0]["lr"] /= 32
    prior["param_groups"][1]["lr"] /= 32
    initial = rebased_adam(prior, cfg)
    optimizer.load_state_dict(copy.deepcopy(initial))
    step()
    ids = dict(
        prior_training_hash="parent",
        prepared_hash="batch",
        frozen_parent_hash=tensor_state_hash(dict(behavior.parent.state_dict())),
    )
    state = dict(
        schema=contract.schema,
        kind=contract.kind,
        config=cfg,
        commitments=ids,
        initial_head=copy.deepcopy(dict(behavior.head.state_dict())),
        initial_critic=copy.deepcopy(dict(value.state_dict())),
        head=copy.deepcopy(dict(actor.head.state_dict())),
        critic=copy.deepcopy(dict(critic.state_dict())),
        initial_optimizer=initial,
        optimizer=copy.deepcopy(optimizer.state_dict()),
        rng_state=torch.get_rng_state(),
        new_actor_steps=1,
        new_critic_steps=1,
        actor_steps=prior_steps + 1,
        critic_steps=prior_steps + 1,
    )
    return state, behavior, value, prior, cfg, ids


def test_actual_adam_continuation_and_explicit_lr_rebase():
    s, b, c, p, cfg, ids = fixture()
    validate_continual_checkpoint(s, b, c, p, config=cfg, commitments=ids, steps=1)
    assert s["initial_optimizer"]["param_groups"][0]["lr"] == 3e-5
    assert p["param_groups"][0]["lr"] == 3e-5 / 32
    for k in range(8):
        assert float(s["initial_optimizer"]["state"][k]["step"]) == 4
        assert float(s["optimizer"]["state"][k]["step"]) == 5


def test_full_recovery_v5_continues_actual_twelve_step_weights_critic_and_adam():
    s, b, c, p, cfg, ids = fixture(12, 3, full=True)
    validate_continual_checkpoint(s, b, c, p, config=cfg, commitments=ids, steps=1)
    assert s["kind"] == "refinement-actor-critic-full-recovery-v5"
    assert float(s["optimizer"]["state"][0]["step"]) == 13
    s["kind"] = "refinement-actor-critic-continuing-v4"
    with pytest.raises(ValueError):
        validate_continual_checkpoint(s, b, c, p, config=cfg, commitments=ids, steps=1)


def test_safe_prepared_state_dict_mapping_order_is_not_weight_drift():
    assert same_nested(OrderedDict(weight=torch.ones(2)), dict(weight=torch.ones(2)))
    assert not same_nested(OrderedDict(weight=torch.ones(2)), dict(weight=torch.zeros(2)))


@pytest.mark.parametrize("generation,prior_steps", [(2, 8), (2, 5), (3, 12), (3, 7)])
def test_later_generations_use_actual_cumulative_moments_without_reset(generation, prior_steps):
    s, b, c, p, cfg, ids = fixture(prior_steps, generation)
    validate_continual_checkpoint(s, b, c, p, config=cfg, commitments=ids, steps=1)
    assert s["kind"] != KIND and s["schema"] != SCHEMA
    assert float(s["optimizer"]["state"][0]["step"]) == prior_steps + 1
    s["optimizer"]["state"][0]["step"].fill_(5)
    with pytest.raises(ValueError):
        validate_continual_checkpoint(s, b, c, p, config=cfg, commitments=ids, steps=1)


@pytest.mark.parametrize(
    "fault",
    [
        "reset_moments",
        "reset_counters",
        "wrong_critic",
        "wrong_actor",
        "parent",
        "kind",
        "config",
        "negative_moment",
        "groups",
        "rate",
        "unchanged",
        "rng",
    ],
)
def test_fresh_zeroed_or_incompatible_state_cannot_masquerade_as_continuation(fault):
    s, b, c, p, cfg, ids = fixture()
    if fault == "reset_moments":
        s["initial_optimizer"]["state"][0]["exp_avg"].zero_()
    elif fault == "reset_counters":
        s["optimizer"]["state"][0]["step"].fill_(1)
    elif fault == "wrong_critic":
        next(iter(s["initial_critic"].values())).zero_()
    elif fault == "wrong_actor":
        next(iter(s["initial_head"].values())).zero_()
    elif fault == "parent":
        ids = ids | dict(frozen_parent_hash="wrong")
    elif fault == "kind":
        s["kind"] = "refinement-actor-critic-v1"
    elif fault == "config":
        s["config"] = s["config"] | dict(generation_index=0)
    elif fault == "negative_moment":
        s["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif fault == "groups":
        s["optimizer"]["param_groups"][0]["params"] = [4, 5, 6, 7]
    elif fault == "rate":
        s["optimizer"]["param_groups"][0]["lr"] = float("nan")
    elif fault == "unchanged":
        s["head"] = copy.deepcopy(s["initial_head"])
    else:
        s["rng_state"] = torch.ones(1)
    with pytest.raises(ValueError):
        validate_continual_checkpoint(s, b, c, p, config=cfg, commitments=ids, steps=1)
