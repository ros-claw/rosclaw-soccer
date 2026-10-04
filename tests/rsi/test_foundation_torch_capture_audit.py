"""Synthetic arithmetic/tamper tests, not source-graph qualification."""

from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.foundation_torch_capture_audit import TorchCallAuditor


class Session:
    def __init__(self, function):
        self.function = function

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, _, inputs):
        return self.function(inputs["input"])


@pytest.fixture
def sample():
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    auditor = TorchCallAuditor.__new__(TorchCallAuditor)
    auditor._torch = torch
    auditor._paths, auditor._hashes = [], []
    bias = torch.zeros(64)
    bias[52] = torch.atanh(torch.tensor(1.5000005 / 2))
    auditor._enc = [(torch.zeros(640, 64), bias)]
    weight = torch.zeros(994, 29)
    weight[:29] = torch.eye(29) * 0.1
    auditor._dec = [(weight, torch.zeros(29))]
    auditor._quant = [
        torch.zeros(64),
        torch.full((64,), 2.0),
        torch.zeros(64),
        torch.full((64,), 16.0),
    ]
    bounded = torch.tanh(bias) * 2
    token = (bounded + (torch.round(bounded) - bounded)) / 16
    original_bounded = bounded.numpy().copy()
    original_bounded[52] = np.float32(1.4999993)
    original_token = np.round(original_bounded) / 16
    auditor._original = Session(lambda _: [original_token[None]])
    auditor._instrumented = Session(
        lambda _: [original_token[None], original_bounded.reshape(2, 32)]
    )
    auditor._decoder = Session(lambda x: [(torch.tensor(x) @ weight).numpy()])
    rng = np.random.default_rng(480)
    obs = rng.normal(size=(3, 2, 994)).astype(np.float32)
    obs[:, :, :64] = token.numpy()
    arrays = dict(
        foundation_neural_encoder_features=np.zeros((3, 2, 640), dtype=np.float32),
        foundation_neural_latent_token=np.broadcast_to(token.numpy(), (3, 2, 64)).copy(),
        foundation_neural_decoder_input=obs,
        foundation_neural_raw_action_isaac=(torch.tensor(obs.reshape(-1, 994)) @ weight)
        .numpy()
        .reshape(3, 2, 29),
    )
    yield auditor, arrays
    torch.set_num_threads(previous)


def test_executed_arithmetic_exact_but_original_boundary_difference_is_not_hidden(sample):
    auditor, arrays = sample
    result = auditor.recompute(arrays)
    assert result["recomputed_executed_calls"] == 6
    assert result["torch_encoder_tolerance"] == result["torch_decoder_tolerance"] == 0
    assert result["original_encoder_bit_parity_passed"] is False
    assert len(result["original_token_differences"]) == 6
    assert result["activation_ceiling"] == "SIM_ONLY"
    assert result["hardware_authorized"] is False


@pytest.mark.parametrize(
    "fault", ["token", "action", "nonfinite", "instrumentation", "continuous", "threads"]
)
def test_fail_closed_without_widening_executed_policy_tolerance(sample, fault):
    auditor, arrays = sample
    if fault == "token":
        arrays["foundation_neural_latent_token"][0, 0, 0] += 0.0625
    elif fault == "action":
        arrays["foundation_neural_raw_action_isaac"][0, 0, 0] += 1e-5
    elif fault == "nonfinite":
        arrays["foundation_neural_encoder_features"][0, 0, 0] = np.nan
    elif fault == "instrumentation":
        auditor._instrumented = Session(lambda _: [np.ones((1, 64)), np.zeros((2, 32))])
    elif fault == "continuous":
        original = auditor._original.run(None, {"input": None})[0]
        auditor._instrumented = Session(lambda _: [original, np.ones((2, 32))])
    else:
        auditor._torch.set_num_threads(2)
    with pytest.raises(ValueError):
        auditor.recompute(arrays)
