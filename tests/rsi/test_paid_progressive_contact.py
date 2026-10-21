import pytest
import torch

from rosclaw_soccer.rsi.paid_contact_residual import ContactResidual, DecoupledContactResidual
from rosclaw_soccer.rsi.paid_progressive_contact import ProgressiveContactResidual


def head(cls, value, logit):
    model = cls()
    with torch.no_grad():
        model.net[-1].weight.zero_()
        model.net[-1].bias.fill_(value)
        model.gate[-1].weight.zero_()
        model.gate[-1].bias.fill_(logit)
    return model


@pytest.mark.parametrize("cls", [ContactResidual, DecoupledContactResidual])
@pytest.mark.parametrize("training", [True, False])
def test_rejection_preserves_acquired_parent_exactly_not_original_zero(cls, training):
    parent = head(cls, 0.7, 2.0).eval()
    plastic = head(DecoupledContactResidual, -9.0, -0.01)
    x = torch.randn(17, 288)
    expected = parent(x).detach().clone()
    model = ProgressiveContactResidual(parent, plastic).train(training)
    assert torch.equal(model(x), expected)
    assert model(x).count_nonzero() == 17 * 29
    assert not parent.training and not any(p.requires_grad for p in parent.parameters())


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_plastic_head_can_reverse_parent_without_expanding_total_motor_authority(sign):
    parent = head(DecoupledContactResidual, -sign * 10, 1)
    plastic = head(DecoupledContactResidual, sign * 10, 0)
    model = ProgressiveContactResidual(parent, plastic).eval()
    value = model(torch.zeros(4, 288))
    assert (value.abs() <= 0.2).all() and (sign * value > 0.199).all()


def test_only_plastic_parameters_update_and_parent_stays_bitwise_frozen():
    parent = head(ContactResidual, 0.4, 1)
    plastic = head(DecoupledContactResidual, 0.01, 1)
    model = ProgressiveContactResidual(parent, plastic).train()
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    plastic_before = [p.clone() for p in model.plastic_parameters()]
    optimizer = torch.optim.Adam(model.plastic_parameters(), lr=0.01)
    loss = model(torch.randn(8, 288)).square().mean()
    loss.backward()
    optimizer.step()
    assert all(torch.equal(v, before[k]) for k, v in parent.state_dict().items())
    assert all(p.grad is None for p in parent.parameters())
    assert any(
        not torch.equal(p, b)
        for p, b in zip(model.plastic_parameters(), plastic_before, strict=True)
    )
    assert not ({id(p) for p in parent.parameters()} & {id(p) for p in model.plastic_parameters()})


@pytest.mark.parametrize("fault", ["shared_head", "shared_tensor", "wrong_plastic", "bound"])
def test_ambiguous_or_expanded_parent_plastic_authority_is_rejected(fault):
    parent = DecoupledContactResidual()
    plastic = DecoupledContactResidual()
    if fault == "shared_head":
        plastic = parent
    elif fault == "shared_tensor":
        plastic.net[0].weight = parent.net[0].weight
    elif fault == "wrong_plastic":
        plastic = ContactResidual()
    else:
        parent.bound_radians = 0.4
    with pytest.raises(ValueError):
        ProgressiveContactResidual(parent, plastic)


@pytest.mark.parametrize("fault", ["nan", "shape", "dtype"])
def test_invalid_features_rejected_before_output(fault):
    model = ProgressiveContactResidual(ContactResidual(), DecoupledContactResidual())
    x = torch.zeros(3, 288)
    if fault == "nan":
        x[0, 0] = float("nan")
    elif fault == "shape":
        x = x[:, :160]
    else:
        x = x.double()
    with pytest.raises(ValueError):
        model(x)
