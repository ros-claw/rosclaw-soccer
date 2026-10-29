import pytest
import torch

from scripts.rsi_mjlab_mlp_transfer import expand_mlp_state


def state(n):
    return {
        "mlp.0.weight": torch.arange(4 * n, dtype=torch.float32).reshape(4, n),
        "mlp.0.bias": torch.arange(4, dtype=torch.float32),
        "obs_normalizer._mean": torch.zeros(1, n),
        "obs_normalizer._var": torch.ones(1, n),
        "obs_normalizer._std": torch.ones(1, n),
        "obs_normalizer.count": torch.tensor(12),
    }


def test_expanded_mlp_exactly_preserves_first_layer_outputs():
    source, destination = state(3), state(5)
    source["obs_normalizer._mean"] = torch.tensor([[2.0, -1.0, 3.0]])
    source["obs_normalizer._std"] = torch.tensor([[1.2, 2.0, 3.0]])
    result = expand_mlp_state(
        source,
        destination,
        old_features=("a", "b", "c"),
        new_features=("a", "b", "c", "ball", "goal"),
    )
    x = torch.tensor([[1.0, 2.0, 5.0, 123.0, -999.0]])
    old = (
        (x[:, :3] - source["obs_normalizer._mean"]) / (source["obs_normalizer._std"] + 0.01)
    ) @ source["mlp.0.weight"].T + source["mlp.0.bias"]
    new = ((x - result["obs_normalizer._mean"]) / (result["obs_normalizer._std"] + 0.01)) @ result[
        "mlp.0.weight"
    ].T + result["mlp.0.bias"]
    assert torch.equal(old, new)
    assert torch.count_nonzero(result["mlp.0.weight"][:, 3:]) == 0
    assert torch.equal(destination["mlp.0.weight"], state(5)["mlp.0.weight"])


@pytest.mark.parametrize(
    "old,new",
    [(("a", "b"), ("b", "a", "c")), (("a", "a"), ("a", "a", "b")), (("a", "b"), ("a", "b"))],
)
def test_rejects_unproven_or_duplicate_feature_schemas(old, new):
    with pytest.raises(ValueError):
        expand_mlp_state(state(len(old)), state(len(new)), old_features=old, new_features=new)
