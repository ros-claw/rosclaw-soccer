import copy

import pytest
import torch

from rosclaw_soccer.rsi.actor_step_schedule import (
    apply_actor_step_schedule,
    validate_actor_step_schedule,
)
from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import same_training_state
from rosclaw_soccer.rsi.paid_recurrent_ppo import backtracked_adam_update


def fixture():
    ps = [torch.nn.Parameter(torch.ones(2)) for _ in range(20)]
    opt = torch.optim.Adam([dict(params=ps[:16], lr=9.375e-7), dict(params=ps[16:], lr=3.125e-5)])
    sum(p.sum() for p in ps).backward()
    opt.step()
    return ps, opt


def trial():
    ps, opt = fixture()
    initial = copy.deepcopy(opt.state_dict())
    schedule = apply_actor_step_schedule(opt, "recover-base-lr")
    epochs = []
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        sum(p.sum() for p in ps).backward()
        calls = []

        def kl(calls=calls):
            calls.append(1)
            return 1.0 if len(calls) == 1 else 0.003

        before = [g["lr"] for g in opt.param_groups]
        decision = backtracked_adam_update(opt, ps, kl)
        epochs.append(
            dict(
                learning_rates_before=before,
                decision=decision,
                learning_rates_after=[g["lr"] for g in opt.param_groups],
            )
        )
    return initial, opt.state_dict(), epochs, schedule


def test_actor_rate_intervention_does_not_reset_weights_moments_or_critic_rate():
    ps, opt = fixture()
    weights = [p.detach().clone() for p in ps]
    initial = copy.deepcopy(opt.state_dict())
    apply_actor_step_schedule(opt, "recover-base-lr")
    expected = copy.deepcopy(initial)
    expected["param_groups"][0]["lr"] = 3e-5
    assert same_training_state(opt.state_dict(), expected)
    assert all(torch.equal(p.detach(), old) for p, old in zip(ps, weights, strict=True))
    initial, final, epochs, schedule = trial()
    validate_actor_step_schedule(initial, final, epochs, schedule, "recover-base-lr")
    assert all(float(s["step"]) == 5 for s in final["state"].values())


@pytest.mark.parametrize("fault", ["schedule", "before", "after", "final", "kl", "false_reject"])
def test_lr_or_trust_region_metadata_cannot_be_silently_changed(fault):
    initial, final, epochs, schedule = trial()
    if fault == "schedule":
        schedule["optimizer_state_reset"] = True
    elif fault == "before":
        epochs[1]["learning_rates_before"][0] *= 2
    elif fault == "after":
        epochs[1]["learning_rates_after"][1] *= 2
    elif fault == "final":
        final["param_groups"][0]["lr"] *= 2
    elif fault == "kl":
        epochs[0]["decision"]["attempts"][-1]["kl"] = 0.02
    else:
        epochs[0]["decision"]["attempts"][0]["kl"] = 0.001
    with pytest.raises(ValueError):
        validate_actor_step_schedule(initial, final, epochs, schedule, "recover-base-lr")


def test_rejected_step_restores_rates_and_slots_without_new_learning():
    ps, opt = fixture()
    before = copy.deepcopy(opt.state_dict())
    result = backtracked_adam_update(opt, ps, lambda: 1.0)
    assert result["applied"] is False and len(result["attempts"]) == 8
    assert same_training_state(opt.state_dict(), before)
    schedule = apply_actor_step_schedule(opt, "inherit")
    assert schedule["initial_actor_lr"] == schedule["scheduled_actor_lr"]
