import pytest
import torch

from rosclaw_soccer.rsi.paid_ancestor_router import AncestralResidualRouter
from rosclaw_soccer.rsi.paid_contact_residual import DecoupledContactResidual
from rosclaw_soccer.rsi.paid_full_entry_contact import FullEntryContactResidual
from rosclaw_soccer.rsi.paid_progressive_contact import ProgressiveContactResidual


def model():
    old, new = DecoupledContactResidual(), DecoupledContactResidual()
    with torch.no_grad():
        old.net[-1].bias.fill_(0.5)
        old.gate[-1].bias.fill_(1.0)
    parent = AncestralResidualRouter(ProgressiveContactResidual(old, new))
    return FullEntryContactResidual(parent, DecoupledContactResidual())


def test_fresh_model_preserves_entire_learned_parent_in_train_and_eval():
    m = model()
    x = torch.randn(9, 288)
    for mode in (True, False):
        m.train(mode)
        assert torch.equal(m(x), m.parent(x))
        assert not m.parent.training
        assert all(not p.requires_grad for p in m.parent.parameters())


def test_new_action_is_not_limited_to_either_old_motor_but_authority_stays_bounded():
    m = model().eval()
    with torch.no_grad():
        m.plastic.gate[-1].bias.fill_(1.0)
        m.plastic.net[-1].bias.fill_(-10.0)
    x = torch.randn(5, 288)
    result = m(x)
    assert result.abs().max() <= 0.2 and (result < 0).all()
    assert not torch.equal(result, m.parent(x)) and result.count_nonzero() > 0


def test_rejection_is_exact_and_new_optimizer_never_contains_ancestral_parameters():
    m = model().train()
    before = {k: v.clone() for k, v in m.parent.state_dict().items()}
    x = torch.randn(5, 288)
    assert torch.equal(m(x), m.parent(x))
    with torch.no_grad():
        m.plastic.gate[-1].bias.fill_(1.0)
    optimizer = torch.optim.Adam(m.plastic_parameters(), lr=0.0003)
    m(x).square().mean().backward()
    optimizer.step()
    assert all(torch.equal(v, before[k]) for k, v in m.parent.state_dict().items())
    assert all(p.grad is None for p in m.parent.parameters())
    assert set(map(id, m.plastic_parameters())).isdisjoint(map(id, m.parent.parameters()))


def test_arbitrary_recursive_parent_is_not_implicitly_admitted():
    with pytest.raises(ValueError):
        FullEntryContactResidual(DecoupledContactResidual(), DecoupledContactResidual())
