import pytest

from rosclaw_soccer.training.episode_group_advantage import group_relative_advantages


def test_groups_are_explicit_and_preserve_failure_members_and_input_arrays():
    torch = pytest.importorskip("torch")
    returns = torch.tensor([-5.0, 10.0, -1.0, 10.0, 3.0, 10.0])
    groups = torch.tensor([20, 7, 20, 7, 20, 7])
    before = returns.clone(), groups.clone()
    result = group_relative_advantages(returns, groups)
    torch.testing.assert_close(result[::2], torch.tensor([-1.2247449, 0.0, 1.2247449]))
    assert torch.equal(result[1::2], torch.zeros(3))
    assert torch.equal(returns, before[0]) and torch.equal(groups, before[1])
    assert not result.requires_grad


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_permutation_and_positive_affine_outcome_transform(dtype):
    torch = pytest.importorskip("torch")
    returns = torch.tensor([1.0, 2.0, 5.0, -3.0, 0.0], dtype=getattr(torch, dtype))
    groups = torch.tensor([9, 9, 9, 200, 200])
    order = torch.tensor([4, 0, 2, 3, 1])
    expected = group_relative_advantages(returns, groups)
    torch.testing.assert_close(
        group_relative_advantages(returns[order], groups[order]), expected[order]
    )
    torch.testing.assert_close(group_relative_advantages(returns * 3 + 4, groups), expected)
    assert expected.dtype == returns.dtype


@pytest.mark.parametrize("value", [0.1, 999999.9375, -999999.9375])
def test_large_constant_groups_are_exactly_zero(value):
    torch = pytest.importorskip("torch")
    returns = torch.full((3001,), value)
    groups = torch.zeros(3001, dtype=torch.int64)
    assert torch.equal(group_relative_advantages(returns, groups), torch.zeros_like(returns))


@pytest.mark.parametrize(
    "fault",
    [
        "nan",
        "inf",
        "large",
        "grad",
        "integer",
        "shape",
        "empty",
        "singleton",
        "negative_group",
        "float_group",
        "group_shape",
    ],
)
def test_invalid_group_contract_fails_closed(fault):
    torch = pytest.importorskip("torch")
    returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
    groups = torch.tensor([0, 0, 7, 7])
    if fault in ("nan", "inf", "large"):
        returns[0] = {"nan": float("nan"), "inf": float("inf"), "large": 1e7}[fault]
    elif fault == "grad":
        returns.requires_grad_()
    elif fault == "integer":
        returns = returns.long()
    elif fault == "shape":
        returns = returns[:, None]
    elif fault == "empty":
        returns, groups = returns[:0], groups[:0]
    elif fault == "singleton":
        groups[0] = 1
    elif fault == "negative_group":
        groups[0] = -1
    elif fault == "float_group":
        groups = groups.float()
    else:
        groups = groups[:2]
    with pytest.raises(ValueError):
        group_relative_advantages(returns, groups)
