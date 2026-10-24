import pytest
import torch
from test_dual_recurrent_motor import model

from rosclaw_soccer.rsi.consolidated_motor_adapter import ConsolidatedMotorAdapter
from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter


def parent_model():
    _, memory = model()
    with torch.no_grad():
        memory.plastic.head.bias.add_(0.015)
    parent = ConsolidatedMotorAdapter(memory, head_bound_rad=0.2)
    with torch.no_grad():
        parent.head[-1].bias.fill_(0.8)
    return parent


def test_zero_refinement_keeps_learned_parent_and_private_history_bitwise():
    parent = parent_model()
    motor = RefinementMotorAdapter(parent)
    p, s = parent.initial_state(batch=2), motor.initial_state(batch=2)
    with torch.no_grad():
        for _ in range(12):
            obs = torch.randn(1, 2, 160)
            expected, p = parent(obs, p)
            actual, s, evidence = motor.forward_with_evidence(obs, s)
            assert torch.equal(expected, actual)
            assert all(torch.equal(a, b) for a, b in zip(p, s, strict=True))
            assert not evidence["refinement_delta_radians"].any()
            assert evidence["physical_retention_proven"] is False


@pytest.mark.parametrize("sign", [-1, 1])
def test_can_reach_opposite_old_boundary_without_extra_total_authority(sign):
    motor = RefinementMotorAdapter(parent_model())
    with torch.no_grad():
        motor.head[-1].bias.fill_(sign * 50)
    for old in (-0.2, -0.19, 0.0, 0.19, 0.2):
        _, total = motor.adapt_features(torch.zeros(1, 1, 288), torch.full((1, 1, 29), old))
        assert (total.abs() <= 0.2).all()
        assert torch.equal(total, torch.full_like(total, sign * 0.2))


@pytest.mark.parametrize("old", [-0.2, 0.2])
def test_gradient_can_leave_either_saturated_parent_boundary_at_zero(old):
    motor = RefinementMotorAdapter(parent_model()).train()
    before = {k: v.clone() for k, v in motor.parent.state_dict().items()}
    opt = torch.optim.SGD(motor.plastic_parameters(), lr=0.1)
    features, delta = torch.zeros(1, 1, 288), torch.full((1, 1, 29), old)
    _, total = motor.adapt_features(features, delta)
    loss = (total + old).square().mean()
    loss.backward()
    assert motor.head[-1].bias.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in motor.parent.parameters())
    opt.step()
    _, updated = motor.adapt_features(features, delta)
    assert (updated + old).square().mean() < loss.detach()
    assert all(torch.equal(v, before[k]) for k, v in motor.parent.state_dict().items())


def test_constructor_does_not_alias_or_freeze_caller_and_train_keeps_parent_eval():
    parent = parent_model()
    motor = RefinementMotorAdapter(parent).train()
    assert any(p.requires_grad for p in parent.head.parameters())
    ptrs = {p.untyped_storage().data_ptr() for p in parent.parameters()}
    assert all(p.untyped_storage().data_ptr() not in ptrs for p in motor.parameters())
    assert not motor.parent.training and not motor.parent.memory.training
    assert len(motor.plastic_parameters()) == 4


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_inputs_or_weights_rejected(bad):
    motor = RefinementMotorAdapter(parent_model())
    features, delta = torch.zeros(1, 1, 288), torch.zeros(1, 1, 29)
    features[0, 0, 0] = bad
    with pytest.raises(ValueError):
        motor.adapt_features(features, delta)
    features.zero_()
    with torch.no_grad():
        motor.head[-1].bias.fill_(bad)
    with pytest.raises(ValueError):
        motor.adapt_features(features, delta)
