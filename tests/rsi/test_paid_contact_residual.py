import pytest
import torch
from torch import nn

from rosclaw_soccer.rsi.paid_contact_residual import ContactResidual, FrozenContactResidualActor


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("mean", torch.zeros(1, 160))
        self.register_buffer("denominator", torch.ones(1, 160))

    def forward_with_features(self, obs, state):
        hidden = obs[..., :128] * self.gain
        return obs[..., :29] * self.gain, state, hidden


def test_zero_residual_preserves_base_actions_memory_and_freezes_all_base_parameters():
    base, residual = TinyBase(), ContactResidual()
    model = FrozenContactResidualActor(base, residual, torch.ones(29))
    obs = torch.randn(5, 1, 160)
    state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
    action, after = model(obs, state)
    assert torch.equal(action, obs[..., :29])
    assert all(torch.equal(x, y) for x, y in zip(after, state, strict=True))
    assert not base.gain.requires_grad
    action.sum().backward()
    assert base.gain.grad is None
    assert residual.net[-1].weight.grad is not None


def test_learned_residual_is_bounded_in_joint_radians_not_action_units():
    base, residual = TinyBase(), ContactResidual()
    model = FrozenContactResidualActor(base, residual, torch.full((29,), 0.1))
    with torch.no_grad():
        residual.net[-1].bias.fill_(100)
    obs = torch.zeros(1, 1, 160)
    action, _ = model(obs, (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128)))
    assert torch.equal(action, torch.full((1, 1, 29), 2.0))


@pytest.mark.parametrize("fault", ["nan", "shape", "dtype"])
def test_invalid_residual_observation_rejected(fault):
    obs = torch.zeros(1, 288)
    if fault == "nan":
        obs[0, 0] = float("nan")
    elif fault == "shape":
        obs = torch.zeros(1, 160)
    else:
        obs = obs.double()
    with pytest.raises(ValueError):
        ContactResidual()(obs)


def test_independent_players_do_not_share_mutable_memory():
    model = FrozenContactResidualActor(TinyBase(), ContactResidual(), torch.ones(29))
    a, b = torch.randn(1, 1, 160), torch.randn(1, 1, 160)
    sa = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
    sb = (torch.ones(2, 1, 128), torch.ones(2, 1, 128))
    first, _ = model(a, sa)
    model(b, sb)
    again, _ = model(a, sa)
    assert torch.equal(first, again)
    assert all((s == 1).all() for s in sb)
