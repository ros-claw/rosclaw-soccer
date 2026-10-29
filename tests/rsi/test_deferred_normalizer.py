import pytest
import torch
from rsl_rl.modules.normalization import EmpiricalNormalization

from scripts.rsi_deferred_normalizer import DeferredNormalization


def test_transform_frozen_during_rollout_then_matches_global_batch():
    a = DeferredNormalization(2)
    reference = EmpiricalNormalization(2)
    batches = [torch.full((16, 2), float(i)) for i in range(24)]
    first = a(torch.ones(1, 2)).clone()
    for x in batches:
        a.update(x)
        assert torch.equal(first, a(torch.ones(1, 2)))
    assert a.count.item() == 0
    a.flush()
    reference.update(torch.cat(batches))
    assert all(torch.equal(v, a.state_dict()[k]) for k, v in reference.state_dict().items())
    assert a.count.item() == 384
    with pytest.raises(ValueError, match="complete 24-step"):
        a.flush()


def test_incomplete_or_oversized_rollout_fails_closed():
    a = DeferredNormalization(2)
    with pytest.raises(ValueError):
        a.update(torch.zeros(15, 2))
    a.update(torch.zeros(16, 2))
    with pytest.raises(ValueError):
        a.flush()
    for _ in range(23):
        a.update(torch.zeros(16, 2))
    with pytest.raises(ValueError):
        a.update(torch.zeros(16, 2))
