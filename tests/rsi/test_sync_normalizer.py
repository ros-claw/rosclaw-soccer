import pytest
import torch
from rsl_rl.modules.normalization import EmpiricalNormalization

from scripts.rsi_sync_normalizer import DeferredFourRankNormalization, FourRankNormalization


def test_global_batch_matches_reference_over_multiple_updates(monkeypatch):
    import scripts.rsi_sync_normalizer as module

    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 4)
    batches = [
        torch.arange(32, dtype=torch.float32).reshape(16, 2) + rank * 20 for rank in range(4)
    ]

    def gather(outputs, _local):
        for output, batch in zip(outputs, batches, strict=True):
            output.copy_(batch)

    monkeypatch.setattr(module.dist, "all_gather", gather)
    reference = EmpiricalNormalization(2)
    actual = FourRankNormalization(2)
    for offset in (0, 2, 5):
        batches[:] = [batch + offset for batch in batches]
        actual.update(batches[0])
        reference.update(torch.cat(batches))
    for key, value in reference.state_dict().items():
        assert torch.equal(value, actual.state_dict()[key])
    assert actual.count.item() == 192


def test_refuses_wrong_rank_count_or_batch(monkeypatch):
    import scripts.rsi_sync_normalizer as module

    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 3)
    with pytest.raises(ValueError):
        FourRankNormalization(2).update(torch.zeros(16, 2))
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 4)
    with pytest.raises(ValueError):
        FourRankNormalization(2).update(torch.zeros(15, 2))


def test_deferred_four_rank_matches_global_rollout_batch(monkeypatch):
    import scripts.rsi_sync_normalizer as module

    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 4)
    batch = torch.arange(24 * 16 * 2, dtype=torch.float32).reshape(24 * 16, 2)

    def gather(outputs, local):
        for rank, output in enumerate(outputs):
            output.copy_(local + rank)

    monkeypatch.setattr(module.dist, "all_gather", gather)
    actual = DeferredFourRankNormalization(2)
    for chunk in batch.split(16):
        actual.update(chunk)
    assert actual.count.item() == 0
    actual.flush()
    expected = EmpiricalNormalization(2)
    expected.update(torch.cat([batch + rank for rank in range(4)]))
    assert all(torch.equal(v, actual.state_dict()[k]) for k, v in expected.state_dict().items())
    assert actual.count.item() == 1536
