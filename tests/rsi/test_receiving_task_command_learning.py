import copy

import pytest
import torch

from rosclaw_soccer.rsi.receiving_command_learning import fit_command_bc
from rosclaw_soccer.rsi.receiving_task_command_learning import continue_task_command


@pytest.fixture(scope="module")
def fixture():
    """Synthetic Torch optimization only, never receiving physics evidence."""
    g = torch.Generator().manual_seed(23)
    x = torch.randn(96, 118, generator=g)
    y = torch.tensor([0.25, 0, 0]) + torch.tensor([0.75, 0.5, 1]) * torch.tanh(x[:, :3] * 0.3)
    tr = torch.arange(96) < 72
    fitted = fit_command_bc(x[:, :112].contiguous(), y, tr, ~tr, seed=123, steps=400)
    parent = dict(
        actor=fitted["actor"].state_dict(),
        optimizer=fitted["optimizer"],
        batch_rng=fitted["batch_rng"],
        optimizer_updates=400,
        seed=123,
    )
    own = torch.randn(64, 118, generator=g)
    advice = torch.tensor([0.25, 0, 0]) + torch.tensor([0.75, 0.5, 1]) * torch.tanh(
        own[:, :3] * 0.3 + own[:, 112:115] * 0.5
    )
    return parent, x, y, tr, ~tr, own, advice


def test_parent_function_adam_and_rng_are_retained_while_new_projection_learns(fixture):
    p, x, y, tr, va, own, advice = fixture
    progress = []
    r = continue_task_command(
        p,
        x,
        y,
        tr,
        va,
        own,
        advice,
        own[:, 112] > 0,
        steps=25,
        on_step=lambda step, loss, norm: progress.append(step),
    )
    assert progress == list(range(1, 26))
    assert r["exact_parent_function_before_updates"]
    assert torch.equal(r["initial_batch_rng"], p["batch_rng"])
    assert r["optimizer_parameter_steps"] == [425] * 6 + [25]
    assert len(r["changed_parameter_tensors"]) == 7
    assert r["after"]["own_advice_mse"] < r["before"]["own_advice_mse"]
    assert p["optimizer"]["param_groups"][0]["params"] == list(range(6))
    assert all(float(s["step"]) == 400 for s in p["optimizer"]["state"].values())


@pytest.mark.parametrize("fault", ["reset", "parent_steps", "overlap", "bounds", "future_grad"])
def test_counterfeit_continuation_or_invalid_advice_rejected_before_updates(fixture, fault):
    p, x, y, tr, va, own, advice = copy.deepcopy(fixture)
    if fault == "reset":
        p["optimizer"]["state"][0]["step"].fill_(0)
    elif fault == "parent_steps":
        p["optimizer_updates"] = 0
    elif fault == "overlap":
        va[0] = True
    elif fault == "bounds":
        advice[0, 0] = 1.1
    else:
        own.requires_grad_(True)
    with pytest.raises(ValueError):
        continue_task_command(p, x, y, tr, va, own, advice, own[:, 112] > 0, steps=1)
