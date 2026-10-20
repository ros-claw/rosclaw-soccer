"""Checkpoint weight predicates only, not a real optimizer training run."""

import pytest
import torch

from scripts.rsi_mjlab_motion_learning_smoke import changed_weight_names, cloned_tensors


def test_weight_change_does_not_count_only_normalizer_or_distribution_updates():
    before = {
        "mlp.0.weight": torch.ones(2, 3),
        "obs_normalizer.mean": torch.zeros(3),
        "distribution.std_param": torch.ones(1),
    }
    old = cloned_tensors(before)
    before["obs_normalizer.mean"] += 1
    before["distribution.std_param"] += 0.1
    assert not changed_weight_names(old, cloned_tensors(before))
    before["mlp.0.weight"][0, 0] += 0.01
    assert changed_weight_names(old, cloned_tensors(before)) == ["mlp.0.weight"]
    assert old["mlp.0.weight"][0, 0] == 1


def test_nonfinite_or_changed_checkpoint_schema_is_not_learning_success():
    before = {"mlp.weight": torch.zeros(2)}
    with pytest.raises(ValueError):
        changed_weight_names(before, {})
    with pytest.raises(ValueError):
        changed_weight_names(before, {"mlp.weight": torch.full((2,), float("nan"))})
    with pytest.raises(ValueError):
        changed_weight_names(before, {"mlp.weight": torch.zeros(3)})
    with pytest.raises(ValueError):
        changed_weight_names(before, {"mlp.weight": torch.zeros(2, dtype=torch.float64)})
