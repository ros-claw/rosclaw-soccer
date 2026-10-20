import copy

import pytest

from rosclaw_soccer.rsi.actor_trust_region import bounded_adam_proposal
from rosclaw_soccer.rsi.optimizer_lineage_audit import exact_tensor_tree


def test_backtracking_respects_kl_and_keeps_moment_clock():
    import torch

    p = torch.nn.Parameter(torch.tensor([0.0]))
    opt = torch.optim.Adam([p], lr=0.1)

    def loss():
        return (p - 1).square().sum(), dict(mean_kl=(p.square() * 100).sum())

    result = bounded_adam_proposal([p], opt, loss)
    assert result["accepted"] and result["alpha"] < 1
    assert result["mean_kl"] <= 0.005
    assert float(opt.state[p]["step"]) == 1
    assert opt.param_groups[0]["lr"] == 0.1
    assert 0 < float(p.detach()) < 0.01


def test_rejection_restores_existing_weights_and_adam_moments():
    import torch

    p = torch.nn.Parameter(torch.tensor([0.1]))
    opt = torch.optim.Adam([p], lr=0.01)
    (p - 1).square().sum().backward()
    opt.step()
    saved = p.detach().clone(), copy.deepcopy(opt.state_dict())

    def no_room():
        return (p - 1).square().sum(), dict(mean_kl=torch.tensor(0.1))

    result = bounded_adam_proposal([p], opt, no_room)
    assert not result["accepted"]
    exact_tensor_tree(p.detach(), saved[0])
    exact_tensor_tree(opt.state_dict(), saved[1])


def test_bad_candidate_exception_also_restores_transaction():
    import torch

    p = torch.nn.Parameter(torch.tensor([0.1]))
    opt = torch.optim.Adam([p], lr=0.1)
    original = p.detach().clone(), copy.deepcopy(opt.state_dict())
    calls = 0

    def broken():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("fixture candidate failure")
        return (p - 1).square().sum(), dict(mean_kl=torch.tensor(0.0))

    with pytest.raises(ValueError):
        bounded_adam_proposal([p], opt, broken)
    exact_tensor_tree(p.detach(), original[0])
    exact_tensor_tree(opt.state_dict(), original[1])
