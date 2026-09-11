"""Redistribution must not discard late motor actions or alter discounted return."""

import pytest

from rosclaw_soccer.training.delayed_component_credit import redistribute_delayed_component

torch = pytest.importorskip("torch")


def case():
    reward = torch.ones(10, 2, dtype=torch.float64)
    component = torch.zeros_like(reward)
    destination = torch.full(reward.shape, -1, dtype=torch.int64)
    alive = torch.ones_like(reward)
    return reward, component, destination, alive


def test_signed_multiple_components_same_destination_preserve_return():
    reward, component, destination, alive = case()
    component[7, 0], component[9, 0], component[8, 1] = 5, -3, 2
    destination[7, 0], destination[9, 0], destination[8, 1] = 1, 1, 2
    copies = [a.clone() for a in (reward, component, destination, alive)]
    result, receipt = redistribute_delayed_component(
        reward, component, destination, alive, gamma=0.995
    )
    assert result[1, 0] == pytest.approx(1 + 5 * 0.995**6 - 3 * 0.995**8)
    assert result[9, 0] == 4
    assert receipt.moved_components == 3 and receipt.maximum_delay_steps == 8
    assert receipt.numeric_return_preserved
    for actual, expected in zip((reward, component, destination, alive), copies, strict=True):
        assert torch.equal(actual, expected)
    assert torch.equal(alive, torch.ones_like(alive))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_randomized_component_accounting(dtype):
    torch.manual_seed(69601)
    reward = torch.randn(200, 32, dtype=dtype)
    component = torch.zeros_like(reward)
    component[100:] = torch.randn(100, 32, dtype=dtype)
    destination = torch.full(reward.shape, -1, dtype=torch.int64)
    destination[100:] = torch.randint(0, 100, (100, 32))
    result, receipt = redistribute_delayed_component(
        reward, component, destination, torch.ones_like(reward), gamma=0.995
    )
    weights = 0.995 ** torch.arange(200, dtype=torch.float64)[:, None]
    assert torch.allclose(
        (weights * result).sum(0), (weights * reward).sum(0), atol=1e-5, rtol=1e-5
    )
    assert receipt.moved_components == 3200


def test_empty_and_same_step_are_exact_identity():
    args = case()
    result, receipt = redistribute_delayed_component(*args, gamma=0.995)
    assert torch.equal(result, args[0]) and receipt.moved_components == 0
    args[1][3, 0] = 5000
    args[2][3, 0] = 3
    result, receipt = redistribute_delayed_component(*args, gamma=0.995)
    assert torch.equal(result, args[0]) and receipt.moved_components == 0


@pytest.mark.parametrize(
    "kind",
    ["future", "missing", "zero_event", "inactive", "restart", "nan", "large", "grad", "dtype"],
)
def test_invalid_assignments_fail_closed(kind):
    reward, component, destination, alive = case()
    component[7, 0] = 2
    destination[7, 0] = 2
    if kind == "future":
        destination[7, 0] = 8
    if kind == "missing":
        destination[7, 0] = -1
    if kind == "zero_event":
        destination[5, 0] = 0
    if kind == "inactive":
        alive[6:, 0] = 0
    if kind == "restart":
        alive[1, 0] = 0
    if kind == "nan":
        component[7, 0] = float("nan")
    if kind == "large":
        component[7, 0] = 1e7
    if kind == "grad":
        reward.requires_grad_(True)
    if kind == "dtype":
        destination = destination.float()
    with pytest.raises(ValueError):
        redistribute_delayed_component(reward, component, destination, alive, gamma=0.995)


@pytest.mark.parametrize("gamma", [True, float("nan"), 1.0, 0.5])
def test_invalid_gamma(gamma):
    with pytest.raises(ValueError):
        redistribute_delayed_component(*case(), gamma=gamma)


def test_complex_mask_and_non_tensor_input_rejected():
    reward, component, destination, alive = case()
    with pytest.raises(ValueError):
        redistribute_delayed_component(reward, component, destination, alive.cfloat(), gamma=0.995)
    with pytest.raises(ValueError):
        redistribute_delayed_component(reward.tolist(), component, destination, alive, gamma=0.995)
