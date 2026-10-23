import pytest
import torch
from test_dual_recurrent_motor import model

from rosclaw_soccer.rsi.consolidated_motor_adapter import ConsolidatedMotorAdapter


def learned_memory():
    _, memory = model()
    with torch.no_grad():
        # Identity must hold for LEARNED plastic memory, not only its initial
        # duplicate of the old frozen foundation.
        memory.plastic.head.bias.add_(0.015)
    return memory


def test_zero_head_preserves_entire_learned_memory_and_private_history_bitwise():
    memory = learned_memory()
    adapter = ConsolidatedMotorAdapter(memory)
    old, own = memory.initial_state(batch=2), adapter.initial_state(batch=2)
    with torch.no_grad():
        for _ in range(20):
            obs = torch.randn(1, 2, 160)
            expected, old = memory(obs, old)
            actual, own, e = adapter.forward_with_evidence(obs, own)
            assert torch.equal(expected, actual)
            assert all(torch.equal(a, b) for a, b in zip(old, own, strict=True))
            assert not e["adapter_delta_radians"].any()
            assert e["physical_retention_proven"] is False


def test_new_head_has_gradient_at_zero_but_slow_memory_has_none_and_no_alias():
    memory = learned_memory()
    before = {k: v.clone() for k, v in memory.state_dict().items()}
    adapter = ConsolidatedMotorAdapter(memory).train()
    source_storage = {p.untyped_storage().data_ptr() for p in memory.parameters()}
    assert all(p.untyped_storage().data_ptr() not in source_storage for p in adapter.parameters())
    assert memory.plastic_parameters()  # constructor did not freeze caller
    assert not adapter.memory.training and not adapter.memory.plastic.training
    obs = torch.randn(9, 1, 160)
    optimizer = torch.optim.Adam(adapter.plastic_parameters(), lr=0.001)
    output, _ = adapter(obs, adapter.initial_state())
    (output - 0.15).square().mean().backward()
    assert adapter.head[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in adapter.memory.parameters())
    optimizer.step()
    assert adapter.head[-1].weight.abs().sum() > 0
    assert all(torch.equal(v, before[k]) for k, v in adapter.memory.state_dict().items())
    assert all(torch.equal(v, before[k]) for k, v in memory.state_dict().items())


@pytest.mark.parametrize("sign", [-1, 1])
def test_fast_head_does_not_expand_original_motor_authority(sign):
    adapter = ConsolidatedMotorAdapter(learned_memory())
    with torch.no_grad():
        adapter.head[-1].bias.fill_(sign * 1e6)
    for old_value in (-0.2, -0.19, 0.0, 0.19, 0.2):
        added, total = adapter.adapt_features(
            torch.zeros(2, 1, 288), torch.full((2, 1, 29), old_value), torch.ones(29)
        )
        assert (added.abs() <= 0.04).all()
        assert (total.abs() <= 0.2).all()


def test_features_are_own_actual_history_not_static_or_borrowed_teacher_state():
    adapter = ConsolidatedMotorAdapter(learned_memory())
    obs = torch.randn(6, 1, 160)
    _, _, whole = adapter.own_features(obs, adapter.initial_state())
    _, state, _ = adapter.own_features(obs[:3], adapter.initial_state())
    _, _, tail = adapter.own_features(obs[3:], state)
    assert torch.allclose(whole["critic_features"][3:], tail["critic_features"], atol=1e-7)
    _, _, reset = adapter.own_features(obs[3:], adapter.initial_state())
    assert not torch.equal(tail["critic_features"], reset["critic_features"])
    with pytest.raises(ValueError):
        adapter(obs, (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128)))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_inputs_and_head_rejected(bad):
    adapter = ConsolidatedMotorAdapter(learned_memory())
    obs = torch.zeros(1, 1, 160)
    obs[0, 0, 0] = bad
    with pytest.raises(ValueError):
        adapter(obs, adapter.initial_state())
    with torch.no_grad():
        adapter.head[-1].bias.fill_(bad)
    with pytest.raises(ValueError):
        adapter(torch.zeros_like(obs), adapter.initial_state())
