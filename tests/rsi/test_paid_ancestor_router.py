import pytest
import torch

from rosclaw_soccer.rsi.paid_ancestor_router import AncestralResidualRouter
from rosclaw_soccer.rsi.paid_contact_residual import DecoupledContactResidual
from rosclaw_soccer.rsi.paid_progressive_contact import ProgressiveContactResidual


def parent():
    old, new = DecoupledContactResidual(), DecoupledContactResidual()
    with torch.no_grad():
        old.net[-1].bias.fill_(0.7)
        old.gate[-1].bias.fill_(1.0)
    return ProgressiveContactResidual(old, new).eval()


def test_default_reject_preserves_current_learned_parent_bitwise():
    p = parent()
    x = torch.randn(7, 288)
    expected = p(x).detach().clone()
    router = AncestralResidualRouter(p).train()
    assert torch.equal(router(x), expected) and expected.count_nonzero() > 0
    assert not p.training and not any(v.requires_grad for v in p.parameters())


def test_admitted_route_is_exact_ancestor_zero_not_approximate_cancellation():
    model = AncestralResidualRouter(parent())
    with torch.no_grad():
        model.gate[-1].bias.fill_(0.0)
    x = torch.randn(3, 288)
    assert model(x).count_nonzero() == 0 and model.parent(x).abs().max() <= 0.2


def test_open_gate_can_be_functionally_identical_without_any_new_motion():
    p = ProgressiveContactResidual(DecoupledContactResidual(), DecoupledContactResidual())
    model = AncestralResidualRouter(p)
    with torch.no_grad():
        model.gate[-1].bias.fill_(2.0)
    pred, logits = model.forward_with_gate(torch.zeros(2, 288))
    assert (logits >= 0).all() and torch.equal(pred, model.parent(torch.zeros(2, 288)))


def test_optimizer_changes_gate_only_not_parent_muscle_memory():
    model = AncestralResidualRouter(parent()).train()
    before = {k: v.clone() for k, v in model.parent.state_dict().items()}
    optimizer = torch.optim.Adam(model.gate.parameters(), lr=0.0003)
    logits = model.forward_with_gate(torch.randn(5, 288))[1]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    loss.backward()
    optimizer.step()
    assert all(torch.equal(v, before[k]) for k, v in model.parent.state_dict().items())
    assert all(p.grad is None for p in model.parent.parameters())


@pytest.mark.parametrize("fault", ["shape", "dtype", "nan"])
def test_invalid_own_features_are_rejected(fault):
    model = AncestralResidualRouter(parent())
    x = torch.zeros(2, 288)
    if fault == "shape":
        x = x[:, :160]
    elif fault == "dtype":
        x = x.double()
    else:
        x[0, 0] = float("nan")
    with pytest.raises(ValueError):
        model(x)
