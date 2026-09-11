"""Learned offset export must preserve the incumbent and unrelated weights."""

import pytest

from rosclaw_soccer.training.actor_output_offset import fuse_actor_output_offset
from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic

torch = pytest.importorskip("torch")


def test_export_changes_only_final_bias_and_preserves_predictions():
    torch.manual_seed(592)
    source = build_ball_residual_actor_critic()
    before = {k: v.clone() for k, v in source.state_dict().items()}
    offset = torch.linspace(-0.5, 0.5, 29)
    exported = fuse_actor_output_offset(source, offset)
    query = torch.randn(32, 133).clamp(-10, 10)
    mean, value = source(query)
    actual, actual_value = exported(query)
    torch.testing.assert_close(actual, mean + offset, rtol=1e-6, atol=1e-6)
    assert torch.equal(actual_value, value)
    assert all(torch.equal(v, before[k]) for k, v in source.state_dict().items())
    assert [k for k, v in exported.state_dict().items() if not torch.equal(v, before[k])] == [
        "actor.4.bias"
    ]
    assert all(
        original.data_ptr() != clone.data_ptr()
        for original, clone in zip(source.parameters(), exported.parameters(), strict=True)
    )
    offset.fill_(0)
    assert not torch.equal(exported.actor[-1].bias, source.actor[-1].bias)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 2.01])
def test_invalid_offsets_fail_without_mutating_source(value):
    source = build_ball_residual_actor_critic()
    before = source.actor[-1].bias.clone()
    with pytest.raises(ValueError, match="detached finite bounded"):
        fuse_actor_output_offset(source, torch.full((29,), value))
    assert torch.equal(source.actor[-1].bias, before)


@pytest.mark.parametrize(
    "offset",
    [
        torch.zeros(28),
        torch.zeros(1, 29),
        torch.zeros(29, dtype=torch.float64),
        torch.zeros(29, requires_grad=True),
    ],
)
def test_shape_dtype_and_autograd_must_match(offset):
    with pytest.raises(ValueError, match="detached finite bounded"):
        fuse_actor_output_offset(build_ball_residual_actor_critic(), offset)


@pytest.mark.parametrize("bound", [True, 0, 3, float("nan")])
def test_invalid_declared_bound_is_rejected(bound):
    with pytest.raises(ValueError):
        fuse_actor_output_offset(
            build_ball_residual_actor_critic(), torch.zeros(29), maximum_absolute_offset=bound
        )


def test_source_corruption_is_rejected_even_outside_actor():
    source = build_ball_residual_actor_critic()
    with torch.no_grad():
        source.critic[-1].bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="source policy"):
        fuse_actor_output_offset(source, torch.zeros(29))


def test_missing_linear_output_is_rejected():
    source = build_ball_residual_actor_critic()
    source.actor.add_module("unexpected", torch.nn.Tanh())
    with pytest.raises(ValueError, match="final biased linear"):
        fuse_actor_output_offset(source, torch.zeros(29))
