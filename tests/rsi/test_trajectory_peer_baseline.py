import pytest
import torch

from rosclaw_soccer.rsi.trajectory_peer_baseline import peer_return_baselines


def test_own_return_is_not_in_own_baseline_and_groups_do_not_mix():
    returns = [torch.tensor([x, 2 * x], dtype=torch.float32) for x in (1.0, 3.0, 7.0, 9.0)]
    groups = ["a", "a", "b", "b"]
    b, a = peer_return_baselines(returns, groups)
    assert torch.equal(b[0], returns[1]) and torch.equal(b[2], returns[3])
    assert torch.equal(a[0], torch.tensor([-2.0, -4.0]))
    changed = [r.clone() for r in returns]
    changed[0] += 20
    b2, a2 = peer_return_baselines(changed, groups)
    assert torch.equal(b2[0], b[0])
    assert torch.equal(a2[0], a[0] + 20)
    assert torch.equal(b2[2], b[2])
    assert torch.equal(returns[0], torch.tensor([1.0, 2.0]))


def test_shared_time_discount_is_not_a_spurious_actor_advantage():
    observed = torch.linspace(0.1, 0.9, 289)
    _, advantage = peer_return_baselines([observed.clone() for _ in range(8)], ["same"] * 8)
    assert all(float(a.abs().max()) < 2e-7 for a in advantage)


def test_negative_terminal_is_kept_and_missing_peer_tail_never_padded():
    good = torch.tensor([0.1, 0.2, 0.3])
    failure = torch.tensor([-3.0])
    b, a = peer_return_baselines([good, failure], ["a", "a"])
    assert torch.equal(b[0], torch.tensor([-3.0, 0.0, 0.0]))
    assert torch.equal(b[1], torch.tensor([0.1]))
    assert len(a[1]) == 1 and float(a[1][0]) < -3
    assert len(a[0]) == 3 and torch.equal(good, torch.tensor([0.1, 0.2, 0.3]))


@pytest.mark.parametrize("fault", ["singleton", "length", "dtype", "nan", "grad", "long", "empty"])
def test_bad_or_unbound_return_groups_rejected(fault):
    returns, groups = [torch.ones(3), torch.ones(3)], ["a", "a"]
    if fault == "singleton":
        groups[1] = "b"
    elif fault == "length":
        groups.pop()
    elif fault == "dtype":
        returns[0] = returns[0].double()
    elif fault == "nan":
        returns[0][0] = float("nan")
    elif fault == "grad":
        returns[0].requires_grad_(True)
    elif fault == "long":
        returns[0] = torch.ones(301)
    else:
        returns[0] = torch.ones(0)
    with pytest.raises(ValueError):
        peer_return_baselines(returns, groups)
