import pytest
import torch

from rosclaw_soccer.rsi.feature_support_gate import FeatureSupportGate
from rosclaw_soccer.rsi.protected_feature_geometry import protected_feature_geometry


def gate():
    p = torch.zeros(5, 288)
    p[:, 0] = torch.arange(5)
    a = torch.zeros(2, 288)
    a[:, 0] = torch.tensor([0.5, 2.5])
    geometry = protected_feature_geometry(p, a)
    return FeatureSupportGate(a, geometry["scale"], geometry["prospective_radius"]), p, a


def test_protected_zero_and_rescue_one_are_exact_not_soft_anchor_losses():
    g, protected, anchors = gate()
    assert torch.equal(g(protected), torch.zeros(5, 1))
    assert torch.equal(g(anchors), torch.ones(2, 1))
    assert torch.equal(g(protected[:, None]), torch.zeros(5, 1, 1))
    assert list(g.parameters()) == []


def test_boundary_is_continuous_without_switching_motion_scripts():
    g, _, a = gate()
    x = a[:1].repeat(101, 1)
    x[:, 0] += torch.linspace(0, 0.5, 101)
    y = g(x).flatten()
    assert (y[:-1] >= y[1:]).all()
    assert y[0] == 1 and y[-1] == 0
    assert (y[1:] - y[:-1]).abs().max() < 0.05


def test_buffers_do_not_alias_caller_and_lower_precision_memory_is_rejected():
    g, _, _ = gate()
    g.float()
    with pytest.raises(ValueError):
        g(torch.zeros(1, 288))


def test_caller_gradients_and_storage_cannot_mutate_fixed_applicability_memory():
    a = torch.ones(1, 288, requires_grad=True)
    scale = torch.ones(288, dtype=torch.float64, requires_grad=True)
    radii = torch.ones(1, dtype=torch.float64, requires_grad=True)
    g = FeatureSupportGate(a, scale, radii)
    with torch.no_grad():
        a.fill_(10)
        scale.fill_(10)
        radii.fill_(10)
    assert all(not x.requires_grad for x in g.buffers())
    assert g(torch.ones(1, 288)).item() == 1
    assert g(torch.full((1, 288), 10.0)).item() == 0


@pytest.mark.parametrize("fault", ["NaN", "empty", "wrong_width"])
def test_invalid_observation_cannot_silently_activate_support(fault):
    g, _, _ = gate()
    x = torch.zeros(1, 288)
    if fault == "NaN":
        x[0, 0] = float("nan")
    elif fault == "empty":
        x = x[:0]
    else:
        x = torch.zeros(1, 287)
    with pytest.raises(ValueError):
        g(x)
