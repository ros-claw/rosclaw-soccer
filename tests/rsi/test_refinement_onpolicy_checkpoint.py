import copy

import pytest
import torch
from test_refinement_actor_critic import fixture as trajectory

from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic
from rosclaw_soccer.rsi.refinement_actor_critic import refinement_own_history_loss
from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import validate_checkpoint
from rosclaw_soccer.rsi.refinement_onpolicy_learning import (
    COMMITMENTS,
    KIND,
    SCHEMA,
    learning_config,
    tensor_state_hash,
)


def fixture():
    actor, _, episode, std = trajectory()
    config = learning_config(25)
    torch.manual_seed(config["seed"])
    critic = MotorCritic()
    behavior = copy.deepcopy(actor).eval().requires_grad_(False)
    initial_value = copy.deepcopy(critic)
    initial_head = {k: v.clone() for k, v in actor.head.state_dict().items()}
    initial_critic = {k: v.clone() for k, v in critic.state_dict().items()}
    optimizer = torch.optim.Adam(
        [
            {"params": list(actor.head.parameters()), "lr": config["actor_lr"]},
            {"params": list(critic.parameters()), "lr": config["critic_lr"]},
        ]
    )
    loss, _ = refinement_own_history_loss(actor, critic, episode, std, rho=0.9)
    loss.backward()
    optimizer.step()
    commitments = {k: "sha256:" + "1" * 64 for k in COMMITMENTS}
    commitments["frozen_parent_hash"] = tensor_state_hash(dict(behavior.parent.state_dict()))
    state = dict(
        schema=SCHEMA,
        kind=KIND,
        config=config,
        commitments=commitments,
        initial_head=initial_head,
        head={k: v.clone() for k, v in actor.head.state_dict().items()},
        initial_critic=initial_critic,
        critic={k: v.clone() for k, v in critic.state_dict().items()},
        optimizer=optimizer.state_dict(),
        rng_state=torch.get_rng_state(),
        actor_steps=1,
        critic_steps=1,
    )
    return state, behavior, initial_value, config, commitments


def test_actual_tensor_optimizer_step_has_new_kind_and_frozen_whole_parent():
    s, b, c, cfg, ids = fixture()
    validate_checkpoint(s, b, c, config=cfg, commitments=ids, steps=1)
    assert s["kind"] != "consolidated-refinement-v1"


@pytest.mark.parametrize(
    "fault", ["kind", "zero", "head", "nan", "parent", "moment", "step", "rate", "rng"]
)
def test_corrupt_stale_or_unupdated_checkpoint_rejected(fault):
    s, b, c, cfg, ids = fixture()
    if fault == "kind":
        s["kind"] = "consolidated-refinement-v1"
    elif fault == "zero":
        s["head"] = copy.deepcopy(s["initial_head"])
    elif fault == "head":
        s["head"].pop(next(iter(s["head"])))
    elif fault == "nan":
        next(iter(s["critic"].values())).fill_(float("nan"))
    elif fault == "parent":
        ids = ids | dict(frozen_parent_hash="sha256:" + "2" * 64)
    elif fault == "moment":
        s["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif fault == "step":
        s["optimizer"]["state"][0]["step"].fill_(2)
    elif fault == "rate":
        s["optimizer"]["param_groups"][0]["lr"] = float("nan")
    else:
        s["rng_state"] = torch.zeros(1)
    with pytest.raises(ValueError):
        validate_checkpoint(s, b, c, config=cfg, commitments=ids, steps=1)
