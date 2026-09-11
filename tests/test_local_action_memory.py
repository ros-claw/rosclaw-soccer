import math

import pytest

from rosclaw_soccer.training.local_action_memory import build_phase_action_memory


def bank():
    torch = pytest.importorskip("torch")
    torch.manual_seed(573)
    observations = torch.randn(3, 4, 6) * 0.1
    phase = torch.arange(3) * (2 * math.pi / 100)
    observations[:, :, 4] = phase.sin()[:, None]
    observations[:, :, 5] = phase.cos()[:, None]
    actions = torch.randn(3, 4, 2) * 0.1
    return torch, observations, actions


def build(observations, actions, **kwargs):
    return build_phase_action_memory(
        observations, actions, neighbors=2, phase_indices=(4, 5), phase_period=100, **kwargs
    )


def test_matches_explicit_neighbor_mean_without_aliasing_banks():
    torch, observations, actions = bank()
    model = build(observations, actions)
    query = observations[:, 0].clone()
    actual = model(query)
    for phase in range(3):
        scale = observations[phase].std(0).clamp_min(0.01)
        distance = ((observations[phase] - query[phase]) / scale).square().mean(1)
        nearest = distance.topk(2, largest=False).indices
        assert torch.equal(actual[phase], actions[phase, nearest].mean(0))
    observations.zero_()
    actions.zero_()
    assert torch.equal(actual, model(query))


def test_learning_is_local_and_bounded_while_centers_remain_frozen():
    torch, observations, actions = bank()
    model = build(observations, actions)
    original = model(observations[:, 0]).detach()
    assert list(dict(model.named_parameters())) == ["correction"]
    model(observations[1, :1]).sum().backward()
    assert model.correction.grad[1].abs().sum() > 0
    assert model.correction.grad[[0, 2]].abs().sum() == 0
    assert torch.equal(model.centers, observations)
    assert torch.equal(model.actions, actions)
    with torch.no_grad():
        model.correction.fill_(1e6)
    assert torch.allclose(model(observations[:, 0]) - original, torch.full_like(original, 0.05))


@pytest.mark.parametrize("fault", ["phase", "nan", "shape", "grad", "integer"])
def test_invalid_banks_rejected(fault):
    torch, observations, actions = bank()
    if fault == "phase":
        observations[1, :, 4:6] = observations[0, :, 4:6]
    elif fault == "nan":
        actions[0, 0, 0] = float("nan")
    elif fault == "shape":
        actions = actions[:, :3]
    elif fault == "grad":
        actions.requires_grad_(True)
    else:
        actions = actions.to(torch.int64)
    with pytest.raises(ValueError):
        build(observations, actions)


@pytest.mark.parametrize("fault", ["phase", "nan", "normalization", "empty"])
def test_unrepresented_or_invalid_query_fails_closed(fault):
    _, observations, actions = bank()
    model = build(observations, actions)
    query = observations[0, :1].clone()
    if fault == "phase":
        query[:, 4] = math.sin(2 * math.pi * 10 / 100)
        query[:, 5] = math.cos(2 * math.pi * 10 / 100)
    elif fault == "nan":
        query[:, 0] = float("nan")
    elif fault == "normalization":
        query[:, 4:6] = 0
    else:
        query = query[:0]
    with pytest.raises(ValueError):
        model(query)


@pytest.mark.parametrize("kwargs", [{"maximum_correction": True}, {"scale_floor": 0}])
def test_invalid_plasticity_configuration_rejected(kwargs):
    _, observations, actions = bank()
    with pytest.raises(ValueError):
        build(observations, actions, **kwargs)
