import pytest
import torch

from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import (
    same_training_state,
    validate_adam_slots,
    validate_critic,
)


def opt_fixture():
    p = [torch.nn.Parameter(torch.ones(2)) for _ in range(20)]
    optimizer = torch.optim.Adam([dict(params=p[:16]), dict(params=p[16:])], lr=0.001)
    sum(x.sum() for x in p).backward()
    optimizer.step()
    return p, optimizer.state_dict()


def test_real_actor_critic_adam_slots_are_distinct_from_256_step_supervised():
    p, opt = opt_fixture()
    validate_adam_slots(opt, p, 1)
    with pytest.raises(ValueError):
        validate_adam_slots(opt, p, 256)


@pytest.mark.parametrize("fault", ["step", "nan", "negative", "duplicate", "groups"])
def test_bad_ppo_ownership_or_optimizer_moments_rejected(fault):
    p, opt = opt_fixture()
    if fault == "step":
        opt["state"][0]["step"].fill_(2)
    if fault == "nan":
        opt["state"][0]["exp_avg"].fill_(float("nan"))
    if fault == "negative":
        opt["state"][0]["exp_avg_sq"].fill_(-1)
    if fault == "duplicate":
        opt["param_groups"][1]["params"][0] = 0
    if fault == "groups":
        opt["param_groups"] = opt["param_groups"][:1]
    with pytest.raises(ValueError):
        validate_adam_slots(opt, p, 1)


def test_non_tensor_critic_is_never_deserialized_as_a_module():
    with pytest.raises(ValueError):
        validate_critic(torch.nn.Linear(2, 1), {})


def test_continual_adam_carries_moments_counters_and_effective_rates():
    import copy

    p, opt = opt_fixture()
    previous = copy.deepcopy(opt)
    parameters = [torch.nn.Parameter(x.detach().clone()) for x in p]
    successor = torch.optim.Adam(
        [dict(params=parameters[:16]), dict(params=parameters[16:])], lr=0.02
    )
    successor.load_state_dict(copy.deepcopy(previous))
    assert same_training_state(successor.state_dict(), previous)
    sum(x.sum() for x in parameters).backward()
    successor.step()
    validate_adam_slots(successor.state_dict(), parameters, 2)
    assert not same_training_state(successor.state_dict(), previous)
    changed = copy.deepcopy(previous)
    changed["param_groups"][0]["lr"] *= 2
    assert not same_training_state(changed, previous)
    changed = copy.deepcopy(previous)
    changed["state"][0]["exp_avg"].add_(0.001)
    assert not same_training_state(changed, previous)


def test_separate_critic_updates_have_honest_distinct_step_counters():
    p, opt = opt_fixture()
    optimizer = torch.optim.Adam([dict(params=p[:16]), dict(params=p[16:])])
    optimizer.load_state_dict(opt)
    optimizer.zero_grad(set_to_none=True)
    for x in p[16:]:
        x.grad = torch.ones_like(x)
    optimizer.step()
    validate_adam_slots(optimizer.state_dict(), p, 1, 2)
    with pytest.raises(ValueError):
        validate_adam_slots(optimizer.state_dict(), p, 1)
