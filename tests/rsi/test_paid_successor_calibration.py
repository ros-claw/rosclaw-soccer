import pytest
import torch

from rosclaw_soccer.rsi.paid_contact_residual import ContactResidual
from rosclaw_soccer.rsi.paid_successor_calibration import calibrate_successor_gate
from rosclaw_soccer.rsi.paid_successor_features import (
    staged_successor_config,
    successor_student_config,
)


def test_training_threshold_shift_preserves_features_and_score_ordering_only_bias_changes():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        gate = ContactResidual().gate
        with torch.no_grad():
            gate[2].weight.fill_(0.1)
            gate[2].bias.fill_(1.0)
        x = torch.randn(17, 288)
    before = {k: v.clone() for k, v in gate.state_dict().items()}
    scores = gate(x).detach()
    result = calibrate_successor_gate(gate, x)
    assert result["maximum_negative_after_CPU"] <= -0.1
    assert (gate(x) < 0).all() and result["not_physical_qualification"]
    for k, v in before.items():
        if k != "2.bias":
            assert torch.equal(gate.state_dict()[k], v)
    assert torch.allclose(gate(x).detach(), scores - result["output_bias_shift"], atol=1e-6)


@pytest.mark.parametrize("fault", ["nan", "empty", "dtype", "shape", "margin"])
def test_invalid_support_never_calibrates_a_candidate(fault):
    gate = ContactResidual().gate
    x = torch.zeros(4, 288)
    margin = 0.1
    if fault == "nan":
        x[0, 0] = float("nan")
    elif fault == "empty":
        x = x[:0]
    elif fault == "dtype":
        x = x.double()
    elif fault == "shape":
        x = x[:, :160]
    else:
        margin = float("nan")
    before = {k: v.clone() for k, v in gate.state_dict().items()}
    with pytest.raises(ValueError):
        calibrate_successor_gate(gate, x, margin=margin)
    assert all(torch.equal(gate.state_dict()[k], v) for k, v in before.items())


def test_staged_protocol_is_explicit_and_does_not_rewrite_v1():
    v1, v2 = successor_student_config(0), staged_successor_config(0)
    assert v1["gate_projection"] and v1["protocol"].endswith("v1")
    assert not v2["gate_projection"] and v2["gate_frozen_during_head_training"]
    assert v2["DEV_episodes_fit"] == 0 and v2["head_training_steps"] == 1024
