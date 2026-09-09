import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_torch import FrozenSonicG1Torch


def shell():
    torch = pytest.importorskip("torch")
    model = object.__new__(FrozenSonicG1Torch)
    model._torch = torch
    model.device = torch.device("cpu")
    return model, torch


@pytest.mark.parametrize("shape", [(0, 994), (4097, 994), (994,), (2, 993)])
def test_frozen_decoder_rejects_wrong_batch_before_inference(shape):
    model, _ = shell()
    with pytest.raises(ValueError, match="batch"):
        model.decode(np.zeros(shape))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1001.0])
def test_frozen_encoder_rejects_nonfinite_or_unbounded_inputs(value):
    model, _ = shell()
    with pytest.raises(ValueError, match="batch"):
        model.encode_g1(np.full((1, 640), value))


def test_encoder_reproduces_exported_reshape_not_invented_feature_order():
    model, torch = shell()
    model._encoder = []
    model._quant = [torch.zeros(64), torch.ones(64), torch.zeros(64), torch.ones(64)]
    captured = []

    def inspect(features, layers):
        captured.append(features.clone())
        return torch.zeros((len(features), 64))

    model._mlp = inspect
    features = torch.arange(640, dtype=torch.float32).reshape(1, 640)
    model.encode_g1(features)
    expected = torch.cat(
        (features[:, :580].reshape(1, 10, 58), features[:, 580:].reshape(1, 10, 6)), 2
    )
    torch.testing.assert_close(captured[0], expected.reshape(1, 640))
    torch.testing.assert_close(features, torch.arange(640, dtype=torch.float32).reshape(1, 640))


def test_mlp_is_swish_except_final_layer_and_preserves_batch_independence():
    model, torch = shell()
    identity = torch.eye(2)
    layers = [(identity, torch.zeros(2)), (identity * 2, torch.ones(2))]
    x = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    torch.testing.assert_close(model._mlp(x, layers), 2 * x * torch.sigmoid(x) + 1)
    torch.testing.assert_close(model._mlp(x, layers)[0], model._mlp(x[:1], layers)[0])


def test_mlp_does_not_return_nonfinite_outputs():
    model, torch = shell()
    with pytest.raises(FloatingPointError):
        model._mlp(torch.ones((1, 2)), [(torch.full((2, 2), float("inf")), torch.zeros(2))])
