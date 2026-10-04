from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.foundation_onnx_capture_audit import recompute_calls


class Session:
    def __init__(self, width, function):
        self.input = SimpleNamespace(name="input", shape=[1, width])
        self.function = function
        self.calls = []

    def get_inputs(self):
        return [self.input]

    def run(self, outputs, inputs):
        assert outputs is None
        value = inputs["input"]
        self.calls.append(value.copy())
        return [self.function(value)]


def sample():
    rng = np.random.default_rng(460)
    features = rng.normal(size=(3, 2, 640)).astype(np.float32)
    token = features[:, :, :64].copy()
    observation = rng.normal(size=(3, 2, 994)).astype(np.float32)
    observation[:, :, :64] = token
    action = observation[:, :, :29] * 0.1
    encoder = Session(1247, lambda x: x[:, 4:68])
    decoder = Session(994, lambda x: x[:, :29] * 0.1)
    return (
        encoder,
        decoder,
        {
            "foundation_neural_encoder_features": features,
            "foundation_neural_latent_token": token,
            "foundation_neural_decoder_input": observation,
            "foundation_neural_raw_action_isaac": action,
        },
    )


def test_all_calls_use_original_single_lane_shapes_and_exact_g1_packing():
    encoder, decoder, arrays = sample()
    result = recompute_calls(encoder, decoder, arrays)
    assert result["recomputed_foundation_calls"] == 6
    assert result["encoder_max_abs_error"] == result["decoder_max_abs_error"] == 0
    for i, packed in enumerate(encoder.calls):
        frame, lane = divmod(i, 2)
        features = arrays["foundation_neural_encoder_features"][frame, lane]
        assert packed.shape == (1, 1247)
        np.testing.assert_array_equal(packed[:, :4], 0)
        np.testing.assert_array_equal(packed[0, 4:584], features[:580])
        np.testing.assert_array_equal(packed[:, 584:590], 0)
        np.testing.assert_array_equal(packed[0, 590:650], features[580:])
        np.testing.assert_array_equal(packed[:, 650:], 0)
        assert decoder.calls[i].shape == (1, 994)


@pytest.mark.parametrize("kind", ["token", "action", "nonfinite", "shape", "input_shape"])
def test_mismatching_original_call_fails_closed(kind):
    encoder, decoder, arrays = sample()
    if kind == "token":
        arrays["foundation_neural_latent_token"][0, 0, 0] += 0.1
    elif kind == "action":
        arrays["foundation_neural_raw_action_isaac"][0, 0, 0] += 0.001
    elif kind == "nonfinite":
        decoder.function = lambda x: np.full((1, 29), np.nan)
    elif kind == "shape":
        decoder.function = lambda x: np.zeros((1, 28))
    else:
        encoder.input.shape[-1] = 1751
    with pytest.raises(ValueError, match="ONNX"):
        recompute_calls(encoder, decoder, arrays)
